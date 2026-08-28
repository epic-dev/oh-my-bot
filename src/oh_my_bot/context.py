import json
import logging
import re

logger = logging.getLogger(__name__)

# Characters per token before anything has been measured. Roughly right for English prose in most
# tokenizers, and corrected from the backend's own accounting after the first reply.
DEFAULT_RATIO = 4.0

# How much of each observation to fold in. Low enough that one odd reply cannot swing the estimate,
# high enough to converge within a handful of turns.
_SMOOTHING = 0.3

# Floor on the learned ratio, so a nonsense observation can never cause a divide-by-zero or an
# absurdly large token estimate.
_MIN_RATIO = 0.5


def message_chars(messages) -> int:
    # Counts the characters that will actually be serialized into the request body, including
    # tool-call JSON and role keys, not just the visible content.
    return sum(len(json.dumps(m, default=str)) for m in messages)


def estimate_tokens(messages, ratio: float) -> int:
    # Estimates the prompt size in tokens using the learned characters-per-token ratio.
    return int(message_chars(messages) / max(ratio, _MIN_RATIO))


def update_ratio(store, model: str, messages, usage: dict) -> float:
    # Corrects the characters-per-token ratio from the backend's own reported prompt_tokens, so the
    # estimate converges on whatever model is actually loaded. Returns the ratio now in force.
    # Backends that omit usage (some Ollama configurations) leave it untouched rather than
    # poisoning it with a guess.
    current = store.get_token_ratio(model, DEFAULT_RATIO)
    prompt_tokens = (usage or {}).get("prompt_tokens")
    if not prompt_tokens:
        return current
    observed = message_chars(messages) / prompt_tokens
    updated = max(current * (1 - _SMOOTHING) + observed * _SMOOTHING, _MIN_RATIO)
    store.set_token_ratio(model, updated)
    return updated


# Marks a tool result whose body has been dropped, and recognises one already dropped so that
# compacting an already-compact history is a no-op.
_ELIDED_RE = re.compile(r"^\[\d+ characters elided\]$")

# Tool results left intact by tier 1. The model is usually still reasoning about the most recent
# ones, so eliding them would break the turn in progress.
KEEP_RECENT_TOOL_RESULTS = 2

# Messages never evicted: the system prompt and the first user message of the session.
PINNED_PREFIX = 2

SUMMARY_PREFIX = "Summary of earlier conversation:"

# Compaction triggers at the threshold but compacts to this fraction of it. Without the gap,
# compaction lands exactly on the budget and the very next message crosses it again, so every
# turn would pay for a compaction (and a summarization model call).
COMPACT_TO_FRACTION = 0.7


def _elide(content: str) -> str:
    # Renders the placeholder that replaces a dropped tool-result body.
    return f"[{len(content)} characters elided]"


def safe_cut_points(messages, pinned: int = PINNED_PREFIX) -> list:
    # Returns indices where the history can be cut without orphaning a tool call from its results.
    # Only immediately before a user message: an assistant message carrying tool_calls and the
    # tool messages answering it must be dropped together, or the request is malformed and strict
    # backends reject the whole thing.
    return [i for i in range(pinned, len(messages)) if messages[i].get("role") == "user"]


def _squeeze_tool_outputs(messages, budget: int, ratio: float) -> int:
    # Tier 1: replace the bodies of old tool results with a placeholder, oldest first, keeping the
    # command that produced them. No model call, no lost structure, and tool output is where the
    # bulk of the growth is. Returns how many were elided.
    tool_indices = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    if KEEP_RECENT_TOOL_RESULTS:
        tool_indices = tool_indices[:-KEEP_RECENT_TOOL_RESULTS]
    elided = 0
    for index in tool_indices:
        content = messages[index].get("content") or ""
        if not content or _ELIDED_RE.match(content):
            continue
        messages[index] = dict(messages[index], content=_elide(content))
        elided += 1
        if estimate_tokens(messages, ratio) <= budget:
            break
    return elided


def _trim(messages, budget: int, ratio: float) -> list:
    # Tier 2: drop the oldest whole turns, cutting only at safe points so tool calls keep their
    # results. Returns the messages that were dropped, for tier 3 to summarize.
    dropped = []
    while estimate_tokens(messages, ratio) > budget:
        cuts = [c for c in safe_cut_points(messages) if c > PINNED_PREFIX]
        if not cuts:
            break
        cut = cuts[0]
        dropped.extend(messages[PINNED_PREFIX:cut])
        del messages[PINNED_PREFIX:cut]
    return dropped


def compact(messages, budget: int, ratio: float, summarize=None):
    # Brings a history under `budget` estimated tokens in three tiers, stopping as soon as it
    # fits: elide old tool outputs, drop the oldest turns, then — only if turns were actually
    # lost — summarize them back in. Returns (messages, note); note is None if nothing was done.
    messages = [dict(m) for m in messages]
    if estimate_tokens(messages, ratio) <= budget:
        return messages, None

    notes = []
    elided = _squeeze_tool_outputs(messages, budget, ratio)
    if elided:
        notes.append(f"elided {elided} old tool output{'s' if elided != 1 else ''}")
    if estimate_tokens(messages, ratio) <= budget:
        return messages, ", ".join(notes)

    dropped = _trim(messages, budget, ratio)
    if dropped:
        notes.append(f"dropped {len(dropped)} old message{'s' if len(dropped) != 1 else ''}")

    if dropped and summarize is not None:
        summary = _summarize_dropped(dropped, summarize)
        if summary:
            candidate = list(messages)
            candidate.insert(PINNED_PREFIX, {"role": "system", "content": f"{SUMMARY_PREFIX} {summary}"})
            # Only keep the summary if it does not push the history back over budget; a summary
            # that causes an overflow is worse than the history it replaced.
            if estimate_tokens(candidate, ratio) <= budget:
                messages = candidate
                notes.append("summarized what was dropped")
    return messages, ", ".join(notes) if notes else None


def _summarize_dropped(dropped, summarize):
    # Asks the caller-supplied summarizer to condense the dropped span. Never allowed to break
    # compaction: losing the summary is survivable, losing the turn is not.
    try:
        return summarize(dropped)
    except Exception:
        logger.exception("Summarization failed; compacting without a summary")
        return None

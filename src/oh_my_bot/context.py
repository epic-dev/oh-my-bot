import json
import logging

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

import json
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import requests

# ChatML-style special tokens: <|im_end|>, <|endoftext|>, <|im_start|>, <|eot_id|>. A tokenizer
# should consume these, but some servers decode them into the reply as plain text. Real prose
# never contains this bracket form, so anything matching it is a leaked control token.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>\s]{0,64}\|>")

# Matches a ```tool ...``` (or plain ```json ...```) fence, used when the backend or model
# ignores the native tools API and writes the call into the message body instead.
_FENCE_RE = re.compile(r"```(?:tool|json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass
class ToolCall:
    # One tool invocation requested by the model, normalized across native and text-fallback forms.
    id: str
    name: str
    arguments: dict = field(default_factory=dict)

    def to_wire(self) -> dict:
        # Renders this call back into OpenAI chat-completions shape for storage in the history.
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments)},
        }


@dataclass
class AssistantMessage:
    # One assistant turn: free text, tool calls, or both, plus the backend's token accounting and
    # why generation stopped. finish_reason == "length" means the model was cut off mid-sentence,
    # which for a reasoning model usually means it never got past its own chain of thought.
    content: Optional[str] = None
    tool_calls: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    finish_reason: Optional[str] = None
    # The unparsed response body, carried back across the subprocess boundary so the parent can
    # trace it. The child cannot write to the store: connections are thread-local to the parent.
    raw: dict = field(default_factory=dict)


@lru_cache(maxsize=8)
def _paired_pattern(tags: tuple):
    # Builds (and caches) the regex matching a complete <tag>...</tag> pair for any configured tag.
    alternation = "|".join(re.escape(tag) for tag in tags)
    return re.compile(rf"<({alternation})>.*?</\1>", re.DOTALL | re.IGNORECASE)


def _last_close(text: str, tags: tuple):
    # Finds the end offset of the last closing tag of any configured tag, or None if there is none.
    lowered = text.lower()
    end = None
    for tag in tags:
        idx = lowered.rfind(f"</{tag.lower()}>")
        if idx != -1:
            candidate = idx + len(tag) + 3
            if end is None or candidate > end:
                end = candidate
    return end


def _first_open(text: str, tags: tuple):
    # Finds the offset of the earliest opening tag of any configured tag, or None if there is none.
    lowered = text.lower()
    start = None
    for tag in tags:
        idx = lowered.find(f"<{tag.lower()}>")
        if idx != -1 and (start is None or idx < start):
            start = idx
    return start


def truncate_at_stop(content, stop_sequences: tuple):
    # Cuts a reply at the first end-of-turn marker. If a marker survives into the text the server
    # failed to stop there, which means anything after it is a turn the model hallucinated
    # (typically a fabricated user message) rather than part of its answer.
    if not content or not stop_sequences:
        return content
    cut = None
    for sequence in stop_sequences:
        idx = content.find(sequence)
        if idx != -1 and (cut is None or idx < cut):
            cut = idx
    return content if cut is None else content[:cut]


def strip_special_tokens(content):
    # Removes any leaked <|...|> control tokens left over after truncation. Unconditional, unlike
    # truncation and reasoning stripping: a control token is never legitimate answer text, whatever
    # the model. The unmodified reply is still recoverable from the traces table.
    if not content:
        return content
    return _SPECIAL_TOKEN_RE.sub("", content)


def strip_reasoning(content, tags: tuple):
    # Removes chain-of-thought from a model reply, given the tag names to strip (they are
    # model-specific, so the caller supplies them from config rather than this module assuming a
    # default). Tolerates the three shapes reasoning arrives in:
    # a complete <tag>...</tag> pair; a dangling close tag (the chat template pre-filled the
    # opening tag, so the reply starts mid-thought); and an unclosed opening tag (generation hit
    # the token limit inside the block). An empty tag tuple disables stripping entirely.
    if not content or not tags:
        return content
    tags = tuple(tags)
    text = _paired_pattern(tags).sub("", content)
    end = _last_close(text, tags)
    if end is not None:
        text = text[end:]
    start = _first_open(text, tags)
    if start is not None:
        text = text[:start]
    return text.strip()


def _normalize_tool_call(raw: dict) -> Optional[ToolCall]:
    # Converts one raw native tool_call entry into a ToolCall, tolerating backends that omit
    # the id or send arguments as an object instead of a JSON string.
    function = raw.get("function") or {}
    name = function.get("name")
    if not name:
        return None
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    return ToolCall(id=raw.get("id") or uuid.uuid4().hex, name=name, arguments=arguments)


def parse_text_tool_calls(content):
    # Extracts tool calls the model wrote into its message body as a fenced JSON block, and
    # returns them alongside the content with those fences removed, so a call the model wrote as
    # prose is not also shown to the user as prose.
    if not content:
        return [], content
    calls = []
    spans = []
    for match in _FENCE_RE.finditer(content):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        name = payload.get("name") or payload.get("tool")
        if not name:
            continue
        arguments = payload.get("arguments") or payload.get("parameters") or {}
        if not isinstance(arguments, dict):
            continue
        calls.append(ToolCall(id=uuid.uuid4().hex, name=name, arguments=arguments))
        spans.append(match.span())
    if not calls:
        return [], content
    remaining = content
    for start, end in reversed(spans):
        remaining = remaining[:start] + remaining[end:]
    return calls, remaining.strip()


class LLMConnector(ABC):
    @abstractmethod
    def complete(self, messages: list, tools: Optional[list] = None) -> AssistantMessage:
        # Sends chat messages plus tool schemas to a backend and returns its assistant turn.
        raise NotImplementedError


class OpenAICompatConnector(LLMConnector):
    def __init__(
        self, base_url: str, model: str, reasoning_tags: tuple, stop_sequences: tuple, max_tokens: int
    ):
        # Stores the backend's base URL, model name, the reasoning tags to strip from replies,
        # the end-of-turn markers to stop on, and the generation budget per reply.
        self.base_url = base_url
        self.model = model
        self.reasoning_tags = tuple(reasoning_tags)
        self.stop_sequences = tuple(stop_sequences)
        self.max_tokens = max_tokens

    def complete(self, messages: list, tools: Optional[list] = None) -> AssistantMessage:
        # POSTs messages (and any tool schemas) to {base_url}/chat/completions and returns the
        # assistant turn. Content is cleaned of everything that is not the answer — text past an
        # end-of-turn marker, leaked control tokens, and chain-of-thought — before tool calls are
        # read from it, so a tool call the model only *considered* inside its reasoning is never
        # executed. Native tool_calls win; a fenced block in the body is the fallback for backends
        # or models that ignore the tools parameter.
        url = f"{self.base_url}/chat/completions"
        payload = {"model": self.model, "messages": messages}
        if self.max_tokens:
            # Servers default this low (mlx_lm.server caps at 512), which is not enough for a
            # reasoning model to finish thinking AND answer — generation stops mid-thought and
            # the reply cleans down to nothing.
            payload["max_tokens"] = self.max_tokens
        if tools:
            payload["tools"] = tools
        if self.stop_sequences:
            # Asking the server to stop is the actual fix; the client-side cleaning below is a
            # fallback for servers that ignore this parameter.
            payload["stop"] = list(self.stop_sequences)
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        message = choice["message"]

        content = truncate_at_stop(message.get("content"), self.stop_sequences)
        content = strip_reasoning(strip_special_tokens(content), self.reasoning_tags)

        raw_calls = message.get("tool_calls") or []
        tool_calls = [c for c in (_normalize_tool_call(r) for r in raw_calls) if c]
        if not tool_calls:
            tool_calls, content = parse_text_tool_calls(content)
        return AssistantMessage(
            content=content,
            tool_calls=tool_calls,
            usage=data.get("usage") or {},
            finish_reason=choice.get("finish_reason"),
            raw=data,
        )

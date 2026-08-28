import re
from abc import ABC, abstractmethod
from functools import lru_cache

import requests

# ChatML-style special tokens: <|im_end|>, <|endoftext|>, <|im_start|>, <|eot_id|>. A tokenizer
# should consume these, but some servers decode them into the reply as plain text. Real prose
# never contains this bracket form, so anything matching it is a leaked control token.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>\s]{0,64}\|>")


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


class LLMConnector(ABC):
    @abstractmethod
    def complete(self, messages: list[dict]) -> str:
        # Sends chat messages to a backend and returns its text reply. Implemented per backend.
        raise NotImplementedError


class OpenAICompatConnector(LLMConnector):
    def __init__(self, base_url: str, model: str, reasoning_tags: tuple, stop_sequences: tuple):
        # Stores the backend's base URL, model name, the reasoning tags to strip from replies,
        # and the end-of-turn markers to stop on.
        self.base_url = base_url
        self.model = model
        self.reasoning_tags = tuple(reasoning_tags)
        self.stop_sequences = tuple(stop_sequences)

    def complete(self, messages: list[dict]) -> str:
        # POSTs messages to {base_url}/chat/completions and returns the reply text, cleaned of
        # anything that is not the answer: content past an end-of-turn marker, leaked control
        # tokens, and chain-of-thought. Backends that return reasoning in a separate field
        # (reasoning_content) are handled for free: only `content` is ever read.
        url = f"{self.base_url}/chat/completions"
        payload = {"model": self.model, "messages": messages}
        if self.stop_sequences:
            # Asking the server to stop is the actual fix; the client-side cleaning below is a
            # fallback for servers that ignore this parameter.
            payload["stop"] = list(self.stop_sequences)
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        content = truncate_at_stop(data["choices"][0]["message"]["content"], self.stop_sequences)
        content = strip_special_tokens(content)
        return strip_reasoning(content, self.reasoning_tags)

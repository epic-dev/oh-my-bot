import re
from abc import ABC, abstractmethod
from functools import lru_cache

import requests


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
    def __init__(self, base_url: str, model: str, reasoning_tags: tuple):
        # Stores the backend's base URL, model name, and which reasoning tags to strip from replies.
        self.base_url = base_url
        self.model = model
        self.reasoning_tags = tuple(reasoning_tags)

    def complete(self, messages: list[dict]) -> str:
        # POSTs messages to {base_url}/chat/completions and returns the reply text with any
        # chain-of-thought removed. Backends that return reasoning in a separate field
        # (reasoning_content) are handled for free: only `content` is ever read.
        url = f"{self.base_url}/chat/completions"
        resp = requests.post(
            url,
            json={"model": self.model, "messages": messages},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return strip_reasoning(data["choices"][0]["message"]["content"], self.reasoning_tags)

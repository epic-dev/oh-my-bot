import re
from abc import ABC, abstractmethod

import requests

# Reasoning models (Qwen3 among them) wrap their chain of thought in <think>...</think> inside
# the normal content field. It is not an answer, it is often longer than the answer, and carrying
# it forward in the history wastes the context window — so it is stripped at the source.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"


def strip_thinking(content):
    # Removes chain-of-thought from a model reply, tolerating the three shapes it arrives in:
    # a complete <think>...</think> pair, a dangling close tag (the chat template pre-filled the
    # opening tag, so the reply starts mid-thought), and an unclosed opening tag (generation hit
    # the token limit inside the block).
    if not content:
        return content
    text = _THINK_BLOCK_RE.sub("", content)
    lowered = text.lower()
    if _CLOSE_TAG in lowered:
        text = text[lowered.rindex(_CLOSE_TAG) + len(_CLOSE_TAG):]
        lowered = text.lower()
    if _OPEN_TAG in lowered:
        text = text[: lowered.index(_OPEN_TAG)]
    return text.strip()


class LLMConnector(ABC):
    @abstractmethod
    def complete(self, messages: list[dict]) -> str:
        # Sends chat messages to a backend and returns its text reply. Implemented per backend.
        raise NotImplementedError


class OpenAICompatConnector(LLMConnector):
    def __init__(self, base_url: str, model: str):
        # Stores the backend's base URL and model name for later chat-completion calls.
        self.base_url = base_url
        self.model = model

    def complete(self, messages: list[dict]) -> str:
        # POSTs messages to {base_url}/chat/completions and returns the reply text with any
        # chain-of-thought removed.
        url = f"{self.base_url}/chat/completions"
        resp = requests.post(
            url,
            json={"model": self.model, "messages": messages},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return strip_thinking(data["choices"][0]["message"]["content"])

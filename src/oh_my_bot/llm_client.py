from abc import ABC, abstractmethod

import requests


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
        # POSTs messages to {base_url}/chat/completions and returns the reply text.
        url = f"{self.base_url}/chat/completions"
        resp = requests.post(
            url,
            json={"model": self.model, "messages": messages},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def build_messages(chat_id: int, text: str) -> list[dict]:
    # Builds the chat-completion messages list for one incoming user message (no history yet).
    return [{"role": "user", "content": text}]

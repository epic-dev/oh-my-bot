import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    # Holds every runtime setting the app needs, loaded once at startup.
    telegram_bot_token: str
    llm_base_url: str
    llm_model: str
    max_workers: int
    llm_timeout_seconds: int
    poll_timeout_seconds: int


def load_config() -> Config:
    # Reads .env plus the real environment and builds a validated Config, raising if the token is missing.
    load_dotenv()
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required but not set (check .env)")
    return Config(
        telegram_bot_token=token,
        llm_base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1"),
        llm_model=os.environ.get("LLM_MODEL", "qwen3:1.7b"),
        max_workers=int(os.environ.get("MAX_WORKERS", "4")),
        llm_timeout_seconds=int(os.environ.get("LLM_TIMEOUT_SECONDS", "60")),
        poll_timeout_seconds=int(os.environ.get("POLL_TIMEOUT_SECONDS", "30")),
    )

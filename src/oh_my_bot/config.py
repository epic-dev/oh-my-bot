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
    allowed_user_ids: frozenset
    db_path: str
    workspace_root: str
    skills_dir: str
    max_loop_iterations: int
    max_llm_retries: int
    max_consecutive_tool_failures: int
    exec_timeout_seconds: int
    exec_max_output_bytes: int
    approval_timeout_seconds: int
    llm_context_tokens: int
    compact_threshold_pct: int


def _parse_user_ids(raw: str) -> frozenset:
    # Parses the comma-separated ALLOWED_USER_IDS value into a set of ints, rejecting empty/invalid input.
    ids = {part.strip() for part in raw.split(",") if part.strip()}
    if not ids:
        raise RuntimeError("ALLOWED_USER_IDS is required but empty (check .env)")
    try:
        return frozenset(int(i) for i in ids)
    except ValueError as exc:
        raise RuntimeError(f"ALLOWED_USER_IDS must be comma-separated integers: {exc}") from exc


def load_config() -> Config:
    # Reads .env plus the real environment and builds a validated Config, raising if required values are missing.
    load_dotenv()
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required but not set (check .env)")
    allowed = os.environ.get("ALLOWED_USER_IDS")
    if not allowed:
        raise RuntimeError("ALLOWED_USER_IDS is required but not set (check .env)")
    return Config(
        telegram_bot_token=token,
        llm_base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1"),
        llm_model=os.environ.get("LLM_MODEL", "qwen3:1.7b"),
        max_workers=int(os.environ.get("MAX_WORKERS", "4")),
        llm_timeout_seconds=int(os.environ.get("LLM_TIMEOUT_SECONDS", "60")),
        poll_timeout_seconds=int(os.environ.get("POLL_TIMEOUT_SECONDS", "30")),
        allowed_user_ids=_parse_user_ids(allowed),
        db_path=os.environ.get("DB_PATH", "./oh-my-bot.db"),
        workspace_root=os.environ.get("WORKSPACE_ROOT", "./workspaces"),
        skills_dir=os.environ.get("SKILLS_DIR", "./skills"),
        max_loop_iterations=int(os.environ.get("MAX_LOOP_ITERATIONS", "5")),
        max_llm_retries=int(os.environ.get("MAX_LLM_RETRIES", "3")),
        max_consecutive_tool_failures=int(os.environ.get("MAX_CONSECUTIVE_TOOL_FAILURES", "3")),
        exec_timeout_seconds=int(os.environ.get("EXEC_TIMEOUT_SECONDS", "30")),
        exec_max_output_bytes=int(os.environ.get("EXEC_MAX_OUTPUT_BYTES", "8192")),
        approval_timeout_seconds=int(os.environ.get("APPROVAL_TIMEOUT_SECONDS", "600")),
        llm_context_tokens=int(os.environ.get("LLM_CONTEXT_TOKENS", "4096")),
        compact_threshold_pct=int(os.environ.get("COMPACT_THRESHOLD_PCT", "75")),
    )

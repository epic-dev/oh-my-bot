import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Chain-of-thought tags stripped from model replies unless REASONING_TAGS overrides them. The tag
# is model-specific (<think> for Qwen3 and DeepSeek-R1, <reasoning> or <thought> elsewhere), which
# is why it is a setting and not a constant in the connector.
DEFAULT_REASONING_TAGS = ("think", "thinking", "reasoning", "thought", "reflection", "scratchpad")

# End-of-turn markers from the common chat templates (ChatML/Qwen, GPT-style, Llama 3, Gemma).
# They are sent to the server as stop sequences and, if one still shows up in the reply, the text
# is truncated there — everything after an end-of-turn marker belongs to a turn the model
# hallucinated, not to its answer.
DEFAULT_STOP_SEQUENCES = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<end_of_turn>")


@dataclass(frozen=True)
class Config:
    # Holds every runtime setting the app needs, loaded once at startup.
    telegram_bot_token: str
    llm_base_url: str
    llm_model: str
    llm_max_tokens: int
    reasoning_tags: tuple
    stop_sequences: tuple
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


def _parse_reasoning_tags(raw) -> tuple:
    # Parses the comma-separated REASONING_TAGS value into a tuple of tag names. An explicitly
    # empty value disables reasoning stripping; an unset value keeps the built-in defaults.
    if raw is None:
        return DEFAULT_REASONING_TAGS
    return tuple(tag.strip().lower() for tag in raw.split(",") if tag.strip())


def _parse_stop_sequences(raw) -> tuple:
    # Parses the comma-separated STOP_SEQUENCES value. An explicitly empty value disables stop
    # handling; an unset value keeps the built-in defaults. Case is preserved: these are literal
    # tokenizer strings, not tag names.
    if raw is None:
        return DEFAULT_STOP_SEQUENCES
    return tuple(seq.strip() for seq in raw.split(",") if seq.strip())


def _parse_user_ids(raw: str) -> frozenset:
    # Parses the comma-separated ALLOWED_USER_IDS value into a set of ints, rejecting empty/invalid input.
    ids = {part.strip() for part in raw.split(",") if part.strip()}
    if not ids:
        raise RuntimeError("ALLOWED_USER_IDS is required but empty (check .env)")
    try:
        return frozenset(int(i) for i in ids)
    except ValueError as exc:
        raise RuntimeError(f"ALLOWED_USER_IDS must be comma-separated integers: {exc}") from exc


def _check_token_budget(config: "Config") -> None:
    # Warns when compaction would allow a prompt that leaves too little room for the reply.
    # LLM_CONTEXT_TOKENS is compared against the *prompt*, so whatever the threshold leaves over
    # has to cover generation. Violating this truncates replies mid-sentence, which looks like a
    # model problem rather than a configuration one.
    headroom = config.llm_context_tokens * (100 - config.compact_threshold_pct) // 100
    if config.llm_max_tokens and headroom < config.llm_max_tokens:
        logger.warning(
            "LLM_CONTEXT_TOKENS=%s at COMPACT_THRESHOLD_PCT=%s%% leaves only %s tokens for a reply, "
            "but LLM_MAX_TOKENS=%s. Replies may be truncated. Raise LLM_CONTEXT_TOKENS to at least "
            "%s, lower LLM_MAX_TOKENS, or lower the threshold.",
            config.llm_context_tokens,
            config.compact_threshold_pct,
            headroom,
            config.llm_max_tokens,
            config.llm_max_tokens * 100 // max(100 - config.compact_threshold_pct, 1),
        )


def load_config() -> Config:
    # Reads .env plus the real environment and builds a validated Config, raising if required values are missing.
    load_dotenv()
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required but not set (check .env)")
    allowed = os.environ.get("ALLOWED_USER_IDS")
    if not allowed:
        raise RuntimeError("ALLOWED_USER_IDS is required but not set (check .env)")
    config = Config(
        telegram_bot_token=token,
        llm_base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1"),
        llm_model=os.environ.get("LLM_MODEL", "qwen3:1.7b"),
        llm_max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "2048")),
        reasoning_tags=_parse_reasoning_tags(os.environ.get("REASONING_TAGS")),
        stop_sequences=_parse_stop_sequences(os.environ.get("STOP_SEQUENCES")),
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
    _check_token_budget(config)
    return config

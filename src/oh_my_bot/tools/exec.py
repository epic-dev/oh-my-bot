import logging
import os
import re
import signal
import subprocess

from dotenv import dotenv_values, find_dotenv

logger = logging.getLogger(__name__)

# Environment keys never passed to a command the model runs. Three sources, because any one of
# them alone leaves a gap:
#   1. every key defined in .env — the app's own secrets;
#   2. a hardcoded critical set, so scrubbing still works if .env cannot be located;
#   3. a name pattern, which catches credentials exported in the shell rather than written to .env.
# Without this, one `env` or `printenv` prints the bot token straight into a chat.
_CRITICAL_KEYS = frozenset({"TELEGRAM_BOT_TOKEN"})
_SECRET_NAME_RE = re.compile(r"(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|API_KEY|_KEY)$", re.IGNORECASE)


def _dotenv_keys() -> frozenset:
    # Reads the keys (never the values) defined in .env, so they can be scrubbed by name.
    try:
        return frozenset(dotenv_values(find_dotenv(usecwd=True)) or {})
    except OSError:
        logger.warning("Could not read .env to determine which keys to scrub")
        return frozenset()


SECRET_KEYS = _dotenv_keys() | _CRITICAL_KEYS


def is_secret_key(key: str) -> bool:
    # Reports whether an environment variable must be withheld from commands the model runs.
    return key in SECRET_KEYS or bool(_SECRET_NAME_RE.search(key))


def clean_env() -> dict:
    # Returns the current environment minus every secret, for use by commands the model runs.
    return {k: v for k, v in os.environ.items() if not is_secret_key(k)}


def _truncate(text: str, max_bytes: int) -> str:
    # Caps captured output, marking the cut so the model knows the result is incomplete.
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    kept = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return kept + f"\n[output truncated, {len(encoded) - max_bytes} more bytes]"


def run_command(command: str, cwd, timeout: int, max_bytes: int) -> str:
    # Runs one shell command to completion in its own process group, merging stderr into stdout,
    # killing the whole group on timeout, and truncating the output to max_bytes.
    try:
        process = subprocess.Popen(
            ["bash", "-c", command],
            cwd=str(cwd),
            env=clean_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            start_new_session=True,
        )
    except OSError as exc:
        return f"Could not start command: {exc}"
    try:
        output = process.communicate(timeout=timeout)[0]
    except subprocess.TimeoutExpired:
        # start_new_session put the child in its own process group, so this also kills anything
        # it spawned — a plain process.kill() would leave orphans behind.
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            process.kill()
        output = process.communicate()[0] or ""
        return _truncate(output, max_bytes) + f"\n[killed after {timeout}s timeout]"
    result = _truncate(output or "", max_bytes)
    if process.returncode != 0:
        result += f"\n[exit code {process.returncode}]"
    return result or "[no output]"


def exec_tool(ctx, command: str) -> str:
    # The `exec` tool: runs a shell command in the session workspace.
    # NOTE: the approval gate lives in tools/dispatch (Task 8), not here, so this function must
    # never be called directly from anywhere that bypasses dispatch.
    return run_command(
        command,
        cwd=ctx.session.workspace(),
        timeout=ctx.config.exec_timeout_seconds,
        max_bytes=ctx.config.exec_max_output_bytes,
    )

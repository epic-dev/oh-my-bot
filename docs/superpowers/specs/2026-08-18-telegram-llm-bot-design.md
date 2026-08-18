# Telegram → Local LLM Bot — Design Spec

## Purpose

A tiny Python app that bridges Telegram users and a locally-running LLM
(qwen3:1.7b). A user messages the bot; the bot forwards the message to a
local LLM over HTTP and sends the reply back. No conversation memory yet,
but the code should make adding it later a small, isolated change.

## Non-functional requirements

- LLM backend is swappable (Ollama / vLLM / MLX) via config, not code changes.
- No SDKs: Telegram and LLM calls are both raw HTTP.
- Robust to connection/query errors — a single failure never crashes the
  polling loop or the whole app.
- Each LLM call is killable/interruptible (runs in its own OS process).
- Bounded concurrency via a thread pool.
- Dependencies managed with `uv`.

## Non-goals

- Persistence or restart-safety of in-flight messages.
- Multi-bot support.
- Markdown/rich message formatting.
- Rate-limit handling beyond basic retry/backoff.
- Automated test suite (manual verification only, see Testing).

## Configuration

All config comes from environment variables, loaded from a gitignored
`.env` file (already present, already in `.gitignore`) via `python-dotenv`.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `TELEGRAM_BOT_TOKEN` | yes | — | Telegram Bot API token |
| `LLM_BASE_URL` | no | `http://localhost:8080/v1` | Base URL of the OpenAI-compatible LLM server |
| `LLM_MODEL` | no | `qwen3:1.7b` | Model name sent in the chat completion request |
| `MAX_WORKERS` | no | `4` | Thread pool size (max concurrent in-flight updates) |
| `LLM_TIMEOUT_SECONDS` | no | `60` | Time to wait for an LLM reply before killing the call |
| `POLL_TIMEOUT_SECONDS` | no | `30` | Telegram long-poll timeout |

## Dependencies

Managed via `uv` (`uv add <package>`, `uv run oh-my-bot`). Expected:
`requests`, `python-dotenv`. No Telegram or LLM SDK.

## Project layout

Standard Python `src`-layout, packaged as an installable app with a console
entry point (`uv_build` backend):

```
oh-my-bot/
├── .env                  # gitignored, already present
├── pyproject.toml        # package config + `oh-my-bot` console script
└── src/
    └── oh_my_bot/
        ├── __init__.py
        ├── config.py
        ├── telegram_client.py
        ├── llm_client.py
        ├── worker.py
        └── main.py
```

Run via `uv run oh-my-bot` (the console script, from `[project.scripts]`)
or equivalently `uv run python -m oh_my_bot.main`. Modules within the
package import each other with relative imports (e.g. `from .config import
load_config`).

Every function/method gets a one-line comment above it describing what it
does (not what each line does — just its purpose).

## Components

### `src/oh_my_bot/config.py`
Loads and validates the environment variables listed above (via
`python-dotenv` + `os.environ`) into a small `Config` object/namedtuple.
Raises a clear error at startup if `TELEGRAM_BOT_TOKEN` is missing.

### `src/oh_my_bot/telegram_client.py`
Raw HTTP calls to `https://api.telegram.org/bot<token>/...`.

- `get_updates(offset)` — long-polls Telegram for new updates starting
  after `offset`, using `POLL_TIMEOUT_SECONDS`. Returns the list of
  updates (empty list on no new messages).
- `send_message(chat_id, text)` — sends a text reply to a chat. Failures
  are caught, logged, and swallowed (nothing else to do if send fails).

### `src/oh_my_bot/llm_client.py`
- `LLMConnector` (ABC) — defines `complete(messages) -> str`, the one
  method any backend must implement.
- `OpenAICompatConnector(LLMConnector)` — implements `complete` by
  POSTing to `{LLM_BASE_URL}/chat/completions` in OpenAI chat-completions
  shape. Works unchanged against MLX (`mlx_lm.server`), and later
  Ollama/vLLM, since all three speak this API — switching backends is a
  config change (`LLM_BASE_URL`/`LLM_MODEL`), not a code change.
- `build_messages(chat_id, text)` — builds the `messages` list passed to
  `complete`. Currently always `[{"role": "user", "content": text}]`;
  this is the single seam where a future history/memory store plugs in.

### `src/oh_my_bot/worker.py`
The concurrency core.

- `ChatLocks` — a small helper holding one `threading.Lock` per
  `chat_id` (created on first use), used to serialize a single user's
  messages while different users still run concurrently.
- `run_llm_call(connector, messages, timeout)` — spawns a fresh
  `multiprocessing.Process` that calls `connector.complete(messages)` and
  puts the result (or an error) on a `multiprocessing.Queue`; joins with
  `timeout`; if the process is still alive, calls `.terminate()` and
  returns a timeout error instead. This is what makes each LLM call
  genuinely killable — thread cancellation alone can't do this in Python.
- `handle_update(update, config, connector)` — the per-message pipeline:
  acquire the chat's lock → build messages → `run_llm_call` → send the
  result (or a user-facing error message) back via `telegram_client` →
  release the lock. Any unexpected exception is caught, logged, and
  turned into a generic error reply so it never kills the worker thread.

### `src/oh_my_bot/main.py`
- `main()` — loads config, constructs the connector, creates the
  `ThreadPoolExecutor(MAX_WORKERS)`, then loops forever: poll
  `get_updates(offset)`, submit each update to the pool via
  `handle_update`, advance `offset` to the latest update id + 1
  immediately (does not wait for replies to finish generating).
  Poll-level exceptions (network errors etc.) are caught, logged, and
  retried with capped exponential backoff so the loop never dies.

## Data flow

1. `main()` polls `get_updates(offset)`.
2. Each update is submitted to the thread pool as `handle_update`.
3. `offset` advances immediately after the poll batch, regardless of how
   far processing has gotten (acking is decoupled from LLM completion).
4. Inside `handle_update`: acquire per-chat lock → `build_messages` →
   `run_llm_call` (own process, timeout-bounded) → `send_message` with
   either the LLM's reply or an error message → release lock.

## Error handling summary

| Failure | Behavior |
|---|---|
| `get_updates` network error | Log, backoff, retry — loop keeps running |
| LLM unreachable/HTTP error | Caught in subprocess, returned as error result → user gets "couldn't reach the AI service" reply |
| LLM call exceeds timeout | Process terminated → user gets "took too long" reply |
| Any other exception in `handle_update` | Caught, logged, generic error reply → worker pool keeps running |
| `send_message` fails | Logged and swallowed |

## Testing (manual)

- Normal round trip: send a message, confirm a reply from the local MLX
  server.
- Timeout path: stop the MLX server mid-request, confirm the user gets
  the timeout message and the app keeps running.
- Concurrency: message from two different Telegram accounts at once,
  confirm both get replies without blocking each other; send two
  messages quickly from the *same* account and confirm replies arrive
  in order (serialized).

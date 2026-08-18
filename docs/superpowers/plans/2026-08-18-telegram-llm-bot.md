# Telegram → Local LLM Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tiny Python app that long-polls Telegram, forwards each message to a local OpenAI-compatible LLM server (MLX now, swappable later), and sends the reply back — with a bounded thread pool, per-user serialization, and a genuinely killable per-call subprocess.

**Architecture:** Five small modules (`config`, `telegram_client`, `llm_client`, `worker`, `main`), each with one responsibility. `main.py` runs the poll loop and owns a `ThreadPoolExecutor`; `worker.py` serializes per chat and runs each LLM call in its own `multiprocessing.Process` so it can be killed on timeout.

**Tech Stack:** Python, `uv` for dependency management, `requests` for all HTTP, `python-dotenv` for `.env` loading. No Telegram or LLM SDK.

**Spec:** `docs/superpowers/specs/2026-08-18-telegram-llm-bot-design.md`

## Global Constraints

- Dependencies only via `uv` (`uv add`, `uv run`); only `requests` and `python-dotenv` — no Telegram/LLM SDKs.
- All configuration comes from environment variables loaded from the existing, gitignored `.env` (never commit secrets).
- Every function/method gets a one-line comment above it describing its purpose.
- No persisted/automated test suite (this is a spec non-goal) — verification happens via throwaway scripts run with `uv run python ...` (never saved to the repo) or, where noted, real manual interaction with Telegram. Delete any scratch verification file after use.
- `LLM_BASE_URL` / `LLM_MODEL` must be the only thing that changes to swap LLM backends — no backend-specific code branches.

---

### Task 1: Project scaffolding with uv

**Files:**
- Create: `pyproject.toml`, `uv.lock`, `.python-version` (all via `uv init`/`uv add`)

**Interfaces:**
- Produces: a `uv run` environment with `requests` and `python-dotenv` installed, used by every later task.

- [ ] **Step 1: Initialize the uv project**

Run: `uv init --name oh-my-bot --no-readme .`
Expected: creates `pyproject.toml`, `.python-version`, and a placeholder `main.py` (Task 6 will overwrite `main.py`). It will not touch the existing `.env`/`.gitignore`.

- [ ] **Step 2: Add dependencies**

Run: `uv add requests python-dotenv`
Expected: `pyproject.toml` gains `requests` and `python-dotenv`; `uv.lock` is created/updated.

- [ ] **Step 3: Verify the environment**

Run:
```bash
uv run python -c "import requests, dotenv; print('deps ok')"
```
Expected: prints `deps ok` with no errors.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock .python-version
git commit -m "chore: scaffold project with uv"
```

---

### Task 2: `config.py` — configuration loading

**Files:**
- Create: `config.py`

**Interfaces:**
- Produces: `Config` (dataclass with fields `telegram_bot_token: str`, `llm_base_url: str`, `llm_model: str`, `max_workers: int`, `llm_timeout_seconds: int`, `poll_timeout_seconds: int`) and `load_config() -> Config`. Every later task that needs configuration calls `load_config()`.

- [ ] **Step 1: Write `config.py`**

```python
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
```

- [ ] **Step 2: Verify it loads from the real `.env`**

Run:
```bash
uv run python -c "
from config import load_config
c = load_config()
assert c.telegram_bot_token, 'token missing'
assert c.max_workers == 4
print('OK:', c.llm_base_url, c.llm_model, c.max_workers)
"
```
Expected: prints `OK: http://localhost:8080/v1 qwen3:1.7b 4` (or your `.env` overrides) with no traceback.

- [ ] **Step 3: Verify the missing-token error path**

Run (uses an empty temp dir so no `.env` is picked up, and clears the shell var):
```bash
mkdir -p /tmp/oh-my-bot-config-check && cd /tmp/oh-my-bot-config-check && \
TELEGRAM_BOT_TOKEN= uv run --project /Users/py/projects/oh-my-bot python -c "
import os
os.environ.pop('TELEGRAM_BOT_TOKEN', None)
import sys; sys.path.insert(0, '/Users/py/projects/oh-my-bot')
from config import load_config
try:
    load_config()
    print('FAIL: expected RuntimeError')
except RuntimeError as e:
    print('OK:', e)
"
cd /Users/py/projects/oh-my-bot
```
Expected: prints `OK: TELEGRAM_BOT_TOKEN is required but not set (check .env)`.

- [ ] **Step 4: Commit**

```bash
git add config.py
git commit -m "feat: add config loading from environment/.env"
```

---

### Task 3: `llm_client.py` — LLM connector

**Files:**
- Create: `llm_client.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `LLMConnector` (ABC with `complete(messages: list[dict]) -> str`), `OpenAICompatConnector(base_url: str, model: str)` implementing it, and `build_messages(chat_id: int, text: str) -> list[dict]`. `worker.py` (Task 5) calls both `build_messages` and `connector.complete`.

- [ ] **Step 1: Write `llm_client.py`**

```python
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
```

- [ ] **Step 2: Verify `build_messages`**

Run:
```bash
uv run python -c "
from llm_client import build_messages
assert build_messages(1, 'hi') == [{'role': 'user', 'content': 'hi'}]
print('OK')
"
```
Expected: `OK`.

- [ ] **Step 3: Verify `OpenAICompatConnector` against a local stub server (no real LLM needed)**

Write this to a throwaway file `/tmp/verify_llm_client.py`:
```python
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import sys
sys.path.insert(0, "/Users/py/projects/oh-my-bot")


class StubHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        assert body["model"] == "test-model", body
        assert body["messages"] == [{"role": "user", "content": "hi"}], body
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"choices": [{"message": {"content": "hello back"}}]}).encode())

    def log_message(self, *args):
        pass


server = HTTPServer(("localhost", 8765), StubHandler)
threading.Thread(target=server.serve_forever, daemon=True).start()

from llm_client import OpenAICompatConnector

conn = OpenAICompatConnector("http://localhost:8765", "test-model")
result = conn.complete([{"role": "user", "content": "hi"}])
assert result == "hello back", result
print("OK:", result)
server.shutdown()
```

Run: `uv run python /tmp/verify_llm_client.py`
Expected: `OK: hello back`

- [ ] **Step 4: Delete the scratch file**

Run: `rm /tmp/verify_llm_client.py`

- [ ] **Step 5: Commit**

```bash
git add llm_client.py
git commit -m "feat: add OpenAI-compatible LLM connector"
```

---

### Task 4: `telegram_client.py` — Telegram HTTP calls

**Files:**
- Create: `telegram_client.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (takes a raw token string).
- Produces: `get_updates(token: str, offset: int, timeout: int) -> list[dict]` and `send_message(token: str, chat_id: int, text: str) -> None`. `main.py` (Task 6) calls `get_updates`; `worker.py` (Task 5) calls `send_message`.

- [ ] **Step 1: Write `telegram_client.py`**

```python
import logging

import requests

logger = logging.getLogger(__name__)


def get_updates(token: str, offset: int, timeout: int) -> list[dict]:
    # Long-polls Telegram for updates after `offset`; returns [] if there are none yet.
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    params = {"offset": offset, "timeout": timeout}
    resp = requests.get(url, params=params, timeout=timeout + 10)
    resp.raise_for_status()
    return resp.json()["result"]


def send_message(token: str, chat_id: int, text: str) -> None:
    # Sends a text reply to a chat; logs and swallows failures since there's nothing else to do.
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=10)
        resp.raise_for_status()
    except requests.RequestException:
        logger.exception("Failed to send message to chat %s", chat_id)
```

- [ ] **Step 2: Verify `get_updates` against the real Telegram API**

Run:
```bash
uv run python -c "
from config import load_config
from telegram_client import get_updates
c = load_config()
result = get_updates(c.telegram_bot_token, 0, 5)
assert isinstance(result, list)
print('OK, updates:', result)
"
```
Expected: `OK, updates: []` (or a list of any pending updates) — proves the token is valid and the HTTP call works. No traceback.

- [ ] **Step 3: Get your chat_id for the next step (manual, human required)**

Open Telegram, start a chat with your bot, and send it any message (e.g. "hello"). Then run the Step 2 command again — this time `result` will contain your message; note the `chat_id` field at `result[0]["message"]["chat"]["id"]`.

- [ ] **Step 4: Verify `send_message` against the real Telegram API (manual, human required)**

Run (replace `<your_chat_id>`):
```bash
uv run python -c "
from config import load_config
from telegram_client import send_message
c = load_config()
send_message(c.telegram_bot_token, <your_chat_id>, 'test message from telegram_client')
print('sent, check Telegram')
"
```
Expected: the message "test message from telegram_client" appears in your Telegram chat with the bot.

- [ ] **Step 5: Commit**

```bash
git add telegram_client.py
git commit -m "feat: add Telegram getUpdates/sendMessage HTTP client"
```

---

### Task 5: `worker.py` — concurrency core

**Files:**
- Create: `worker.py`

**Interfaces:**
- Consumes: `LLMConnector.complete` and `build_messages` from `llm_client.py` (Task 3); `send_message` from `telegram_client.py` (Task 4); `Config` from `config.py` (Task 2, specifically `.llm_timeout_seconds`).
- Produces: `ChatLocks` (with `.get(chat_id: int) -> threading.Lock`), `run_llm_call(connector, messages: list[dict], timeout: int) -> str`, and `handle_update(update: dict, config: Config, connector: LLMConnector, chat_locks: ChatLocks, telegram_token: str) -> None`. `main.py` (Task 6) submits `handle_update` to its thread pool.

- [ ] **Step 1: Write `worker.py`**

```python
import logging
import multiprocessing
import threading

from llm_client import build_messages
from telegram_client import send_message

logger = logging.getLogger(__name__)


class ChatLocks:
    def __init__(self):
        # Holds one Lock per chat_id, created on first use, so each user's messages are serialized.
        self._locks: dict[int, threading.Lock] = {}
        self._registry_lock = threading.Lock()

    def get(self, chat_id: int) -> threading.Lock:
        # Returns the Lock for this chat_id, creating it the first time it's requested.
        with self._registry_lock:
            if chat_id not in self._locks:
                self._locks[chat_id] = threading.Lock()
            return self._locks[chat_id]


def _llm_call_target(connector, messages, result_queue):
    # Runs inside the child process: calls the connector and puts the outcome on the queue.
    try:
        result_queue.put(("ok", connector.complete(messages)))
    except Exception as exc:
        result_queue.put(("error", str(exc)))


def run_llm_call(connector, messages: list[dict], timeout: int) -> str:
    # Runs connector.complete in its own process; kills it and returns an error string on timeout.
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_llm_call_target, args=(connector, messages, result_queue))
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return "Sorry, that took too long. Please try again."
    if result_queue.empty():
        return "Sorry, I couldn't reach the AI service. Please try again shortly."
    status, payload = result_queue.get()
    if status == "error":
        logger.error("LLM call failed: %s", payload)
        return "Sorry, I couldn't reach the AI service. Please try again shortly."
    return payload


def handle_update(update, config, connector, chat_locks, telegram_token):
    # Processes one Telegram update end-to-end: serialize per chat, call the LLM, send the reply.
    message = update.get("message")
    if not message or "text" not in message:
        return
    chat_id = message["chat"]["id"]
    text = message["text"]
    lock = chat_locks.get(chat_id)
    with lock:
        try:
            messages = build_messages(chat_id, text)
            reply = run_llm_call(connector, messages, config.llm_timeout_seconds)
        except Exception:
            logger.exception("Unexpected error handling update for chat %s", chat_id)
            reply = "Sorry, something went wrong. Please try again."
        send_message(telegram_token, chat_id, reply)
```

- [ ] **Step 2: Verify `ChatLocks`**

Run:
```bash
uv run python -c "
from worker import ChatLocks
locks = ChatLocks()
a1 = locks.get(1)
a2 = locks.get(1)
b1 = locks.get(2)
assert a1 is a2
assert a1 is not b1
print('OK')
"
```
Expected: `OK`.

- [ ] **Step 3: Verify `run_llm_call`'s success, timeout-kill, and error paths**

Write this to a throwaway file `/tmp/verify_worker.py` (the fake connectors must be at module level so `multiprocessing`'s spawn method can pickle them):
```python
import sys
import time

sys.path.insert(0, "/Users/py/projects/oh-my-bot")

from worker import run_llm_call


class FakeConnector:
    def complete(self, messages):
        return "hello"


class SlowConnector:
    def complete(self, messages):
        time.sleep(5)
        return "too slow"


class BrokenConnector:
    def complete(self, messages):
        raise ConnectionError("boom")


if __name__ == "__main__":
    assert run_llm_call(FakeConnector(), [], timeout=10) == "hello"
    print("OK: success path")

    start = time.monotonic()
    result = run_llm_call(SlowConnector(), [], timeout=1)
    elapsed = time.monotonic() - start
    assert "took too long" in result, result
    assert elapsed < 3, f"expected fast return, took {elapsed:.1f}s"
    print(f"OK: timeout/kill path returned in {elapsed:.1f}s")

    result = run_llm_call(BrokenConnector(), [], timeout=10)
    assert "couldn't reach" in result, result
    print("OK: error path")
```

Run: `uv run python /tmp/verify_worker.py`
Expected:
```
OK: success path
OK: timeout/kill path returned in 1.0s
OK: error path
```
The timeout line proves the subprocess was actually killed (elapsed ≈ 1s, not 5s) — this is the core "killable LLM call" requirement from the spec.

- [ ] **Step 4: Delete the scratch file**

Run: `rm /tmp/verify_worker.py`

- [ ] **Step 5: Commit**

```bash
git add worker.py
git commit -m "feat: add thread-pool worker with per-chat locking and killable LLM calls"
```

---

### Task 6: `main.py` — poll loop and wiring

**Files:**
- Modify: `main.py` (overwrite the `uv init` placeholder)

**Interfaces:**
- Consumes: `load_config` (Task 2), `OpenAICompatConnector` (Task 3), `get_updates` (Task 4), `ChatLocks`/`handle_update` (Task 5).
- Produces: the runnable app (`uv run python main.py`). Nothing later depends on this module.

- [ ] **Step 1: Write `main.py`**

```python
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from config import load_config
from llm_client import OpenAICompatConnector
from telegram_client import get_updates
from worker import ChatLocks, handle_update

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Loads config, starts the thread pool, and long-polls Telegram forever, dispatching updates to workers.
    config = load_config()
    connector = OpenAICompatConnector(config.llm_base_url, config.llm_model)
    chat_locks = ChatLocks()
    offset = 0
    backoff = 1

    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        while True:
            try:
                updates = get_updates(config.telegram_bot_token, offset, config.poll_timeout_seconds)
                backoff = 1
            except Exception:
                logger.exception("Failed to poll Telegram, retrying in %ss", backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue

            for update in updates:
                offset = max(offset, update["update_id"] + 1)
                pool.submit(handle_update, update, config, connector, chat_locks, config.telegram_bot_token)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Start a local LLM server**

Start `mlx_lm.server` (or whichever OpenAI-compatible MLX server you use) bound to the host/port matching your `.env`'s `LLM_BASE_URL` (default `http://localhost:8080/v1`), serving `qwen3:1.7b` (or your `.env`'s `LLM_MODEL`).

- [ ] **Step 3: Run the app**

Run: `uv run python main.py`
Expected: logs show the app running with no traceback; it blocks, long-polling Telegram.

- [ ] **Step 4: End-to-end verification (manual, human required)**

With the app running and the LLM server up:
- From your Telegram account, send the bot a normal message. Confirm you get back a real LLM-generated reply.
- Stop the LLM server, send another message, confirm you get the "couldn't reach the AI service" reply, and that the app keeps running (check the terminal — no crash).
- Restart the LLM server. From a second Telegram account (or ask a friend), message the bot at the same time as sending two quick messages from your first account. Confirm: both accounts get replies, and your two messages come back in the order you sent them (serialized), not interleaved or reordered.

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat: wire up polling loop and thread pool in main.py"
```

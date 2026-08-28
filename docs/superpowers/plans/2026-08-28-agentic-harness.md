# Agentic Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing single-call Telegram bot into an agent: a bounded model/tool loop with an approval-gated `exec` tool that runs commands on the host, workspace-scoped file tools, persistent SQLite-backed sessions, tiered context compaction, and progressively-disclosed skills.

**Architecture:** The thread pool is replaced by one long-lived actor thread per chat, so a chat blocked on a user's approval tap never delays another chat. `main.py` becomes a router (messages → actor queues, callback queries → pending approvals). `agent.py` runs the loop under three independent circuit breakers. Every LLM call and every `exec` still runs in its own killable subprocess.

**Tech Stack:** Python, `uv`, `requests`, `python-dotenv`. No new third-party dependencies — `sqlite3`, `subprocess`, `threading`, `pathlib` are stdlib, and `SKILL.md` frontmatter is parsed by hand rather than adding PyYAML.

**Spec:** `docs/superpowers/specs/2026-08-28-agentic-harness-design.md`

## Global Constraints

- Carried forward: dependencies only via `uv`; all config from the gitignored `.env`; a one-line comment above every function describing its purpose; no persisted test suite (verification via throwaway scripts under the scratch dir, deleted after use, or real manual Telegram interaction); `LLM_BASE_URL`/`LLM_MODEL` remain the only thing that changes to swap backends.
- Modules live in `src/oh_my_bot/` and import each other relatively (`from .config import load_config`). Verification scripts import the installed package (`from oh_my_bot.config import load_config`) under `uv run`.
- **The workspace guard in `tools/files.py` (Task 6) is the load-bearing security control of this design.** `exec` is confirmed per command, so it may touch anything; the file tools are never confirmed, so they must be provably unable to escape the workspace. Do not weaken it, and do not add an unconfirmed tool that writes outside the workspace.
- Never let a secret from `.env` reach an `exec` child's environment.
- Each phase leaves the bot in a working, runnable state.

## Decisions taken since the spec

The spec left eight questions open. Each is resolved below so the plan is executable; every one is a small, local change if you want it the other way.

| # | Question | Decision | Rationale |
|---|---|---|---|
| 1 | Deny semantics | Deny returns `"The user denied this command."` to the model as the tool result, charged against the tool-failure budget | Lets the model try a different approach instead of throwing away the turn; the breaker still stops a model that keeps asking |
| 2 | Non-allowlisted users | Silent drop, logged | Leaks nothing about the bot to anyone who finds it |
| 3 | Group chats | DMs only — actors key on `chat_id`, but any update where `chat.type != "private"` is dropped | A host-unrestricted `exec` bot in a group is a much larger trust surface; revisit deliberately |
| 4 | Native-tools detection | Opportunistic — always send `tools`, read `tool_calls` if present, else try parsing a fenced block from the content | No startup probe to maintain, and it also catches the common case where the backend accepts `tools` but the small model answers in prose |
| 5 | `exec` limits | 30s timeout, 8 KiB output cap, stderr merged into stdout | One stream is simpler for the model to read; the cap keeps a stray `find /` from eating the window |
| 6 | Compaction trigger | 75% of `LLM_CONTEXT_TOKENS`, checked before **every** model call, including mid-loop | A tool-heavy turn can overflow without the user sending anything; the check costs nothing |
| 7 | Skill frontmatter | `name` and `description` only; unknown keys parsed and ignored | Nothing enforces `allowed-tools` yet, and ignoring unknown keys leaves room to add it |
| 8 | Actor lifecycle | Threads live for the process lifetime, no reaping | The allowlist is a handful of people; a reaper is complexity with no payoff at this scale |

---

## Phase 1 — Sessions

The bot becomes conversational. No loop, no tools, no new attack surface.

### Task 1: Extend `config.py` and `.gitignore`

**Files:**
- Modify: `src/oh_my_bot/config.py`, `.gitignore`, `.env`

**Interfaces:**
- Produces: `Config` gains `allowed_user_ids: frozenset[int]`, `db_path`, `workspace_root`, `skills_dir`, `max_loop_iterations`, `max_llm_retries`, `max_consecutive_tool_failures`, `exec_timeout_seconds`, `exec_max_output_bytes`, `approval_timeout_seconds`, `llm_context_tokens`, `compact_threshold_pct`. Every later task reads from it.

- [ ] **Step 1: Rewrite `src/oh_my_bot/config.py`**

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
```

- [ ] **Step 2: Add your Telegram user id to `.env`**

Find your numeric user id by messaging [@userinfobot](https://t.me/userinfobot) on Telegram, then add to `.env`:

```bash
ALLOWED_USER_IDS=123456789
```

Note this is your **user** id, not a chat id. Multiple ids are comma-separated.

- [ ] **Step 3: Extend `.gitignore`**

Append:
```
workspaces/
*.db
```

- [ ] **Step 4: Verify config loads and rejects bad input**

Run:
```bash
uv run python -c "
from oh_my_bot.config import load_config, _parse_user_ids
c = load_config()
assert c.allowed_user_ids, 'allowlist empty'
assert c.max_loop_iterations == 5
assert c.compact_threshold_pct == 75
print('OK:', sorted(c.allowed_user_ids), c.db_path, c.llm_context_tokens)
for bad in ('', '  ', 'abc', '1,notanum'):
    try:
        _parse_user_ids(bad)
        raise AssertionError(f'expected rejection of {bad!r}')
    except RuntimeError as e:
        print('OK rejected', repr(bad), '-', e)
"
```
Expected: an `OK:` line with your id, then four `OK rejected` lines.

- [ ] **Step 5: Commit**

```bash
git add src/oh_my_bot/config.py .gitignore
git commit -m "feat: add agent configuration and user allowlist"
```

---

### Task 2: `store.py` — SQLite persistence

**Files:**
- Create: `src/oh_my_bot/store.py`

**Interfaces:**
- Consumes: `Config` (Task 1).
- Produces: `Store(db_path)` with `connect()`, `init_schema()`, `get_or_create_session(chat_id)`, `new_session(chat_id)`, `append_message(session_id, role, content, tool_calls, tool_call_id)`, `load_messages(session_id)`, `set_auto_approve(chat_id, on)`, `add_approval_pattern(session_id, pattern)`, `load_approval_patterns(session_id)`, `get_token_ratio(model)`, `set_token_ratio(model, ratio)`, `append_trace(session_id, request, response)`. Used by `session.py` (Task 3), `approvals.py` (Task 8), `context.py` (Task 13).

- [ ] **Step 1: Write `src/oh_my_bot/store.py`**

```python
import json
import sqlite3
import threading
import time
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT PRIMARY KEY,
    chat_id       INTEGER NOT NULL,
    active        INTEGER NOT NULL DEFAULT 1,
    auto_approve  INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_chat ON sessions(chat_id, active);

CREATE TABLE IF NOT EXISTS messages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    role          TEXT NOT NULL,
    content       TEXT,
    tool_calls    TEXT,
    tool_call_id  TEXT,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS approvals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    pattern       TEXT NOT NULL,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_approvals_session ON approvals(session_id);

CREATE TABLE IF NOT EXISTS token_ratios (
    model         TEXT PRIMARY KEY,
    ratio         REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS traces (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    request       TEXT,
    response      TEXT,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session_id, id);
"""


class Store:
    def __init__(self, db_path: str):
        # Holds the database path and a thread-local slot for per-thread connections.
        self.db_path = db_path
        self._local = threading.local()

    def connect(self) -> sqlite3.Connection:
        # Returns this thread's connection, opening it in WAL mode on first use.
        # sqlite3 connections are not shareable across threads and every chat actor is its own thread.
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def init_schema(self) -> None:
        # Creates every table and index if they don't already exist.
        self.connect().executescript(SCHEMA)
        self.connect().commit()

    def get_or_create_session(self, chat_id: int) -> str:
        # Returns the chat's active session id, creating a first session if it has none.
        conn = self.connect()
        row = conn.execute(
            "SELECT session_id FROM sessions WHERE chat_id = ? AND active = 1", (chat_id,)
        ).fetchone()
        if row:
            return row["session_id"]
        return self.new_session(chat_id)

    def new_session(self, chat_id: int) -> str:
        # Deactivates any current session for the chat and creates a fresh one, returning its id.
        conn = self.connect()
        conn.execute("UPDATE sessions SET active = 0 WHERE chat_id = ?", (chat_id,))
        session_id = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO sessions (session_id, chat_id, active, auto_approve, created_at) VALUES (?, ?, 1, 0, ?)",
            (session_id, chat_id, time.time()),
        )
        conn.commit()
        return session_id

    def append_message(self, session_id, role, content=None, tool_calls=None, tool_call_id=None) -> None:
        # Persists one message (user, assistant, tool, or system) at the end of a session's history.
        conn = self.connect()
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls, tool_call_id, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, role, content, json.dumps(tool_calls) if tool_calls else None, tool_call_id, time.time()),
        )
        conn.commit()

    def load_messages(self, session_id: str) -> list:
        # Loads a session's full history in order, as chat-completion message dicts.
        rows = self.connect().execute(
            "SELECT role, content, tool_calls, tool_call_id FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        messages = []
        for row in rows:
            msg = {"role": row["role"]}
            if row["content"] is not None:
                msg["content"] = row["content"]
            if row["tool_calls"]:
                msg["tool_calls"] = json.loads(row["tool_calls"])
            if row["tool_call_id"]:
                msg["tool_call_id"] = row["tool_call_id"]
            messages.append(msg)
        return messages

    def replace_messages(self, session_id: str, messages: list) -> None:
        # Overwrites a session's history wholesale; used by compaction to persist the compacted form.
        conn = self.connect()
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        now = time.time()
        conn.executemany(
            "INSERT INTO messages (session_id, role, content, tool_calls, tool_call_id, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    session_id,
                    m["role"],
                    m.get("content"),
                    json.dumps(m["tool_calls"]) if m.get("tool_calls") else None,
                    m.get("tool_call_id"),
                    now,
                )
                for m in messages
            ],
        )
        conn.commit()

    def get_auto_approve(self, session_id: str) -> bool:
        # Reports whether the session has confirmation suspended via /auto.
        row = self.connect().execute(
            "SELECT auto_approve FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return bool(row and row["auto_approve"])

    def set_auto_approve(self, session_id: str, on: bool) -> None:
        # Turns per-command confirmation off (or back on) for the rest of the session.
        conn = self.connect()
        conn.execute("UPDATE sessions SET auto_approve = ? WHERE session_id = ?", (1 if on else 0, session_id))
        conn.commit()

    def add_approval_pattern(self, session_id: str, pattern: str) -> None:
        # Records a command pattern the user chose to always allow.
        conn = self.connect()
        conn.execute(
            "INSERT INTO approvals (session_id, pattern, created_at) VALUES (?, ?, ?)",
            (session_id, pattern, time.time()),
        )
        conn.commit()

    def load_approval_patterns(self, session_id: str) -> list:
        # Returns every always-allow pattern recorded for a session.
        rows = self.connect().execute(
            "SELECT pattern FROM approvals WHERE session_id = ?", (session_id,)
        ).fetchall()
        return [r["pattern"] for r in rows]

    def get_token_ratio(self, model: str, default: float = 4.0) -> float:
        # Returns the learned characters-per-token ratio for a model, or the default if unlearned.
        row = self.connect().execute("SELECT ratio FROM token_ratios WHERE model = ?", (model,)).fetchone()
        return row["ratio"] if row else default

    def set_token_ratio(self, model: str, ratio: float) -> None:
        # Stores the updated characters-per-token ratio for a model.
        conn = self.connect()
        conn.execute(
            "INSERT INTO token_ratios (model, ratio) VALUES (?, ?)"
            " ON CONFLICT(model) DO UPDATE SET ratio = excluded.ratio",
            (model, ratio),
        )
        conn.commit()

    def append_trace(self, session_id: str, request, response) -> None:
        # Records one raw LLM request/response pair for after-the-fact debugging.
        conn = self.connect()
        conn.execute(
            "INSERT INTO traces (session_id, request, response, created_at) VALUES (?, ?, ?, ?)",
            (session_id, json.dumps(request, default=str), json.dumps(response, default=str), time.time()),
        )
        conn.commit()
```

- [ ] **Step 2: Verify the store, including cross-thread use**

Write to `$SCRATCH/verify_store.py` (use your scratch dir; delete it after):
```python
import threading

from oh_my_bot.store import Store

store = Store("/tmp/oh-my-bot-verify.db")
store.init_schema()

sid = store.get_or_create_session(42)
assert store.get_or_create_session(42) == sid, "should reuse the active session"

store.append_message(sid, "user", "hello")
store.append_message(sid, "assistant", None, tool_calls=[{"id": "c1", "function": {"name": "exec"}}])
store.append_message(sid, "tool", "output", tool_call_id="c1")
msgs = store.load_messages(sid)
assert [m["role"] for m in msgs] == ["user", "assistant", "tool"], msgs
assert msgs[1]["tool_calls"][0]["id"] == "c1", msgs
assert msgs[2]["tool_call_id"] == "c1", msgs
print("OK: history round-trips")

sid2 = store.new_session(42)
assert sid2 != sid
assert store.load_messages(sid2) == [], "new session starts empty"
assert len(store.load_messages(sid)) == 3, "old session history is retained"
print("OK: /new isolates history")

assert store.get_auto_approve(sid2) is False
store.set_auto_approve(sid2, True)
assert store.get_auto_approve(sid2) is True
store.add_approval_pattern(sid2, "ls")
assert store.load_approval_patterns(sid2) == ["ls"]
print("OK: session flags and patterns")

assert store.get_token_ratio("m") == 4.0
store.set_token_ratio("m", 3.2)
store.set_token_ratio("m", 3.4)
assert store.get_token_ratio("m") == 3.4, "upsert should overwrite"
print("OK: token ratio upsert")

errors = []


def worker(n):
    try:
        s = store.get_or_create_session(1000 + n)
        for i in range(20):
            store.append_message(s, "user", f"m{i}")
        assert len(store.load_messages(s)) == 20
    except Exception as exc:
        errors.append(exc)


threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
for t in threads:
    t.start()
for t in threads:
    t.join()
assert not errors, errors
print("OK: 8 threads wrote concurrently with no sqlite thread errors")
```

Run: `uv run python "$SCRATCH/verify_store.py"` (with `SCRATCH` set to your scratch dir)
Expected: six `OK:` lines. The last one is the important one — it proves the thread-local connection strategy works under the actor model.

- [ ] **Step 3: Delete the scratch files**

Run: `rm "$SCRATCH/verify_store.py" /tmp/oh-my-bot-verify.db*`

- [ ] **Step 4: Commit**

```bash
git add src/oh_my_bot/store.py
git commit -m "feat: add SQLite store for sessions, messages, approvals, and traces"
```

---

### Task 3: `session.py` — session objects and commands

**Files:**
- Create: `src/oh_my_bot/session.py`

**Interfaces:**
- Consumes: `Store` (Task 2), `Config` (Task 1).
- Produces: `Session` (holding `chat_id`, `session_id`, `store`, `config`) with `history()`, `add_user(text)`, `add_assistant(content, tool_calls)`, `add_tool_result(tool_call_id, content)`, `workspace()`, and module-level `handle_command(text, session, ...) -> str | None`. Used by `agent.py` (Task 10) and `actors.py` (Task 9).

- [ ] **Step 1: Write `src/oh_my_bot/session.py`**

```python
import logging
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant operating on the user's computer via a Telegram bot.

You have tools available. Use them when the task requires acting on the system;
answer directly when it does not. After a tool returns, use its result to
continue. When you have the final answer, reply with plain text and no tool call.

Working directory for file tools: {workspace}
"""


class Session:
    def __init__(self, chat_id: int, store, config):
        # Binds a chat to its active session id and the store that persists it.
        self.chat_id = chat_id
        self.store = store
        self.config = config
        self.session_id = store.get_or_create_session(chat_id)

    def workspace(self) -> Path:
        # Returns this session's workspace directory, creating it on first use.
        path = Path(self.config.workspace_root).resolve() / str(self.chat_id) / self.session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def system_prompt(self) -> str:
        # Builds the system message for this session.
        return SYSTEM_PROMPT.format(workspace=self.workspace())

    def history(self) -> list:
        # Returns the full message list to send to the model: system prompt plus persisted history.
        return [{"role": "system", "content": self.system_prompt()}] + self.store.load_messages(self.session_id)

    def add_user(self, text: str) -> None:
        # Appends the user's message to the session history.
        self.store.append_message(self.session_id, "user", text)

    def add_assistant(self, content=None, tool_calls=None) -> None:
        # Appends an assistant turn, which may carry content, tool calls, or both.
        self.store.append_message(self.session_id, "assistant", content, tool_calls=tool_calls)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        # Appends the result of one tool call, linked back to the call that produced it.
        self.store.append_message(self.session_id, "tool", content, tool_call_id=tool_call_id)

    def reset(self) -> None:
        # Starts a fresh session: new context, new empty workspace, auto-approve off, patterns cleared.
        old_workspace = Path(self.config.workspace_root).resolve() / str(self.chat_id) / self.session_id
        self.session_id = self.store.new_session(self.chat_id)
        _archive_workspace(old_workspace)

    def auto_approve(self) -> bool:
        # Reports whether the user suspended per-command confirmation for this session.
        return self.store.get_auto_approve(self.session_id)


def _archive_workspace(path: Path) -> None:
    # Renames a finished session's workspace out of the way instead of deleting it.
    # A recursive delete on a path built from ids is not worth getting wrong on a bot
    # that can run arbitrary commands on this host.
    if not path.exists():
        return
    archive = path.with_name(f"{path.name}.archived.{int(time.time())}")
    try:
        shutil.move(str(path), str(archive))
    except OSError:
        logger.exception("Could not archive workspace %s", path)


def handle_command(text: str, session: Session) -> str:
    # Handles a /-prefixed message, returning the reply to send, or None if it isn't a command.
    if not text.startswith("/"):
        return None
    command = text.split()[0].lower()
    if command == "/new":
        session.reset()
        return "Started a new session. Fresh context, fresh workspace, confirmations back on."
    if command == "/auto":
        session.store.set_auto_approve(session.session_id, True)
        return "Auto-approve on: commands will run without confirmation until /new."
    if command == "/status":
        messages = session.store.load_messages(session.session_id)
        return (
            f"Session: {session.session_id[:8]}\n"
            f"Messages: {len(messages)}\n"
            f"Auto-approve: {'on' if session.auto_approve() else 'off'}\n"
            f"Workspace: {session.workspace()}"
        )
    return f"Unknown command: {command}"
```

`/stop`, `/skills`, `/skill`, and `/compact` are added in later tasks; `handle_command` grows a branch each time.

- [ ] **Step 2: Verify sessions, reset, and workspace isolation**

Write to `$SCRATCH/verify_session.py`:
```python
from pathlib import Path

from oh_my_bot.config import load_config
from oh_my_bot.session import Session, handle_command
from oh_my_bot.store import Store

config = load_config()
config = type(config)(**{**config.__dict__, "db_path": "/tmp/oh-my-bot-sess.db", "workspace_root": "/tmp/oh-my-bot-ws"})
store = Store(config.db_path)
store.init_schema()

s = Session(7, store, config)
ws1 = s.workspace()
assert ws1.is_dir(), "workspace should be created"
(ws1 / "leftover.txt").write_text("from session 1")

s.add_user("hello")
s.add_assistant("hi there")
history = s.history()
assert history[0]["role"] == "system" and str(ws1) in history[0]["content"], history[0]
assert [m["role"] for m in history[1:]] == ["user", "assistant"], history
print("OK: history includes system prompt with workspace path")

reply = handle_command("/status", s)
assert "Auto-approve: off" in reply and "Messages: 2" in reply, reply
assert handle_command("/auto", s) and s.auto_approve() is True
print("OK: /status and /auto")

old_id = s.session_id
assert handle_command("/new", s)
assert s.session_id != old_id
assert s.store.load_messages(s.session_id) == [], "new session starts empty"
assert s.auto_approve() is False, "/new must turn auto-approve back off"
ws2 = s.workspace()
assert ws2 != ws1 and not (ws2 / "leftover.txt").exists(), "new session gets a clean workspace"
assert not ws1.exists(), "old workspace should have been moved aside"
assert list(Path("/tmp/oh-my-bot-ws/7").glob("*.archived.*")), "old workspace should be archived, not deleted"
print("OK: /new resets context, workspace, and toggles; old workspace archived")

assert handle_command("plain text", s) is None
print("OK: non-commands pass through")
```

Run: `uv run python "$SCRATCH/verify_session.py"`
Expected: four `OK:` lines. The archive assertion matters — it proves `/new` does not recursively delete.

- [ ] **Step 3: Delete the scratch files**

Run: `rm "$SCRATCH/verify_session.py" /tmp/oh-my-bot-sess.db*; rm -rf /tmp/oh-my-bot-ws`

- [ ] **Step 4: Commit**

```bash
git add src/oh_my_bot/session.py
git commit -m "feat: add persistent sessions with /new, /auto, and /status"
```

---

### Task 4: Wire sessions into the existing worker

**Files:**
- Modify: `src/oh_my_bot/worker.py`, `src/oh_my_bot/main.py`, `src/oh_my_bot/llm_client.py`

**Interfaces:**
- Consumes: `Session`, `handle_command` (Task 3), `Store` (Task 2).
- Produces: a conversational bot. `build_messages()` is deleted from `llm_client.py` — `Session.history()` replaces it. Still one LLM call per message, still the thread pool; Task 9 replaces the pool.

- [ ] **Step 1: Delete `build_messages` from `src/oh_my_bot/llm_client.py`**

Remove the function entirely. `Session.history()` is now the seam the original spec reserved it for.

- [ ] **Step 2: Update `handle_update` in `src/oh_my_bot/worker.py`**

Replace the body of the `with lock:` block so it: constructs a `Session` for the chat, tries `handle_command` first (sending its reply and returning if it matched), otherwise appends the user message, calls `run_llm_call` with `session.history()`, persists the assistant reply with `session.add_assistant(reply)`, and sends it. `handle_update` gains a `store` parameter.

- [ ] **Step 3: Update `main()` in `src/oh_my_bot/main.py`**

Construct `store = Store(config.db_path)` and call `store.init_schema()` before the loop, then pass `store` through to `handle_update`. Add the allowlist check before submitting an update:

```python
                user_id = (update.get("message") or {}).get("from", {}).get("id")
                chat_type = (update.get("message") or {}).get("chat", {}).get("type")
                if user_id not in config.allowed_user_ids or chat_type != "private":
                    logger.info("Dropping update from user %s (chat type %s)", user_id, chat_type)
                    continue
```

- [ ] **Step 4: Verify the allowlist rejects and admits correctly**

Write to `$SCRATCH/verify_allowlist.py` a small script that builds fake update dicts (an allowlisted private chat, a non-allowlisted user, an allowlisted user in a `"group"` chat, and an update with no `message` key) and asserts the filter expression admits only the first. Copy the exact condition from `main.py` so the test tracks the real code.

Expected: `OK` with only the first update admitted.

- [ ] **Step 5: Delete the scratch file, then verify end-to-end (manual, human required)**

Start your LLM server and run `uv run oh-my-bot`. From Telegram:
- Send "my name is Pavel", then "what is my name?" — the second reply must show it remembered. This is the whole point of the phase.
- Send `/status` — confirm it reports the message count and workspace.
- Send `/new`, then "what is my name?" — it must have forgotten.
- Message from a non-allowlisted account (or temporarily remove your id from `ALLOWED_USER_IDS` and restart) — confirm no reply and a "Dropping update" log line.

- [ ] **Step 6: Commit**

```bash
git add src/oh_my_bot/worker.py src/oh_my_bot/main.py src/oh_my_bot/llm_client.py
git commit -m "feat: give the bot conversation memory and enforce the user allowlist"
```

---
## Phase 2 — The loop

The bot becomes an agent. This phase introduces every security-relevant control at once, on purpose: the workspace guard, env scrubbing, and the approval gate all land before or alongside the tool that needs them.

### Task 5: `llm_client.py` — tool-calling connector

**Files:**
- Modify: `src/oh_my_bot/llm_client.py`, `src/oh_my_bot/worker.py`

**Interfaces:**
- Produces: `ToolCall` (`id`, `name`, `arguments: dict`, `to_wire()`), `AssistantMessage` (`content`, `tool_calls: list`, `usage: dict`), `parse_text_tool_calls(content) -> list`, and `complete(messages, tools=None) -> AssistantMessage`. `worker.run_llm_call` now returns `(status, AssistantMessage | error_str)` instead of a bare string. Consumed by `agent.py` (Task 10) and `context.py` (Task 13, via `usage`).

- [ ] **Step 1: Rewrite `src/oh_my_bot/llm_client.py`**

```python
import json
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import requests

# Matches a ```tool ...``` (or plain ```json ...```) fence, used when the backend or model
# ignores the native tools API and writes the call into the message body instead.
_FENCE_RE = re.compile(r"```(?:tool|json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass
class ToolCall:
    # One tool invocation requested by the model, normalized across native and text-fallback forms.
    id: str
    name: str
    arguments: dict = field(default_factory=dict)

    def to_wire(self) -> dict:
        # Renders this call back into OpenAI chat-completions shape for storage in the history.
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments)},
        }


@dataclass
class AssistantMessage:
    # One assistant turn: free text, tool calls, or both, plus the backend's token accounting.
    content: Optional[str] = None
    tool_calls: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)


def _normalize_tool_call(raw: dict) -> Optional[ToolCall]:
    # Converts one raw native tool_call entry into a ToolCall, tolerating backends that omit
    # the id or send arguments as an object instead of a JSON string.
    function = raw.get("function") or {}
    name = function.get("name")
    if not name:
        return None
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            return None
    return ToolCall(id=raw.get("id") or uuid.uuid4().hex, name=name, arguments=arguments or {})


def parse_text_tool_calls(content: str) -> list:
    # Extracts tool calls the model wrote into its message body as a fenced JSON block.
    if not content:
        return []
    calls = []
    for match in _FENCE_RE.finditer(content):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = payload.get("name") or payload.get("tool")
        if not name:
            continue
        arguments = payload.get("arguments") or payload.get("parameters") or {}
        if isinstance(arguments, dict):
            calls.append(ToolCall(id=uuid.uuid4().hex, name=name, arguments=arguments))
    return calls


class LLMConnector(ABC):
    @abstractmethod
    def complete(self, messages: list, tools: Optional[list] = None) -> AssistantMessage:
        # Sends chat messages plus tool schemas to a backend and returns its assistant turn.
        raise NotImplementedError


class OpenAICompatConnector(LLMConnector):
    def __init__(self, base_url: str, model: str):
        # Stores the backend's base URL and model name for later chat-completion calls.
        self.base_url = base_url
        self.model = model

    def complete(self, messages: list, tools: Optional[list] = None) -> AssistantMessage:
        # POSTs messages (and any tool schemas) to {base_url}/chat/completions and returns the
        # assistant turn, falling back to parsing a fenced block when no native tool_calls come back.
        payload = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
        resp = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        message = data["choices"][0]["message"]
        content = message.get("content")
        raw_calls = message.get("tool_calls") or []
        tool_calls = [c for c in (_normalize_tool_call(r) for r in raw_calls) if c]
        if not tool_calls:
            tool_calls = parse_text_tool_calls(content)
        return AssistantMessage(content=content, tool_calls=tool_calls, usage=data.get("usage") or {})
```

- [ ] **Step 2: Update `run_llm_call` in `src/oh_my_bot/worker.py`**

Change it to take a `tools` argument, pass it to `connector.complete`, and return the tuple `("ok", AssistantMessage)` / `("error", message)` / `("timeout", message)` rather than a user-facing string. `agent.py` (Task 10) turns those into user-facing text — the worker's job is only to make the call killable. Keep the existing drain-before-join ordering; the comment explaining why is still correct and still load-bearing.

Update `handle_update`'s existing call site to use `.content` so Phase 1 keeps working.

- [ ] **Step 3: Verify parsing of both call paths against a stub server**

Write to `$SCRATCH/verify_llm_client.py` a stub `HTTPServer` (same pattern as the original plan's Task 3) that returns, on successive requests:
1. A native `tool_calls` response with `arguments` as a JSON **string**.
2. A native response with `arguments` as an **object** and **no `id`** (the Ollama shape).
3. A response with no `tool_calls` and a ` ```tool ` fence in the content.
4. A plain text response with no tool calls at all.
5. A response whose fence contains malformed JSON.

Assert: cases 1-3 each yield exactly one `ToolCall` with `arguments == {"command": "ls"}` and a non-empty `id`; case 4 yields `tool_calls == []` and the content intact; case 5 yields `tool_calls == []` rather than raising. Also assert `usage` is carried through when present.

Expected: one `OK` line per case. Case 2 is the one that matters most in practice — it is what Ollama actually sends.

- [ ] **Step 4: Delete the scratch file and commit**

```bash
git add src/oh_my_bot/llm_client.py src/oh_my_bot/worker.py
git commit -m "feat: add tool-calling connector with text-fallback parsing"
```

---

### Task 6: `tools/` registry and workspace-scoped file tools

**Files:**
- Create: `src/oh_my_bot/tools/__init__.py`, `src/oh_my_bot/tools/files.py`

**Interfaces:**
- Produces: `ToolError` (exception), `ToolContext` (`session`, `config`), `TOOL_SCHEMAS` (list of OpenAI function schemas), `dispatch(tool_call, ctx) -> (output: str, ok: bool)`, and `read_file` / `write_file`. Consumed by `agent.py` (Task 10). `exec` is registered in Task 7 and `skill` in Task 16.

- [ ] **Step 1: Write `src/oh_my_bot/tools/files.py`**

```python
from pathlib import Path

MAX_READ_BYTES = 16384


class ToolError(Exception):
    # Raised when a tool cannot run; the message is returned to the model as the tool result.
    pass


def resolve_in_workspace(workspace: Path, path: str) -> Path:
    # Resolves a model-supplied path against the session workspace and refuses anything outside it.
    # This is the security boundary for the file tools: they are never confirmed by the user, so
    # they must be provably unable to touch anything beyond the workspace. resolve() also collapses
    # symlinks, so a symlink planted inside the workspace cannot be used to escape it.
    root = Path(workspace).resolve()
    # A leading "/" in `path` makes `root / path` absolute and discards `root`; the containment
    # check below is what catches that, so absolute paths are rejected rather than silently honored.
    target = (root / path).resolve()
    if target != root and not target.is_relative_to(root):
        raise ToolError(f"Path is outside the workspace and was refused: {path}")
    return target


def read_file(ctx, path: str) -> str:
    # Reads a UTF-8 text file from the session workspace, truncating very large files.
    target = resolve_in_workspace(ctx.session.workspace(), path)
    if not target.is_file():
        raise ToolError(f"No such file: {path}")
    data = target.read_text(errors="replace")
    if len(data) > MAX_READ_BYTES:
        return data[:MAX_READ_BYTES] + f"\n[truncated, {len(data) - MAX_READ_BYTES} more characters]"
    return data


def write_file(ctx, path: str, content: str) -> str:
    # Writes a UTF-8 text file inside the session workspace, creating parent directories as needed.
    target = resolve_in_workspace(ctx.session.workspace(), path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Wrote {len(content)} characters to {target}"
```

- [ ] **Step 2: Write `src/oh_my_bot/tools/__init__.py`**

A `ToolContext` dataclass (`session`, `config`, plus `approvals` and `telegram` added in Task 8), a `TOOL_SCHEMAS` list holding the OpenAI function schema for each tool, a `_HANDLERS` name→callable map, and:

```python
def dispatch(tool_call, ctx):
    # Runs one tool call and returns (output, ok); ok=False marks a failure for the breaker.
    handler = _HANDLERS.get(tool_call.name)
    if handler is None:
        return f"Unknown tool: {tool_call.name}", False
    try:
        return handler(ctx, **tool_call.arguments), True
    except ToolError as exc:
        return str(exc), False
    except TypeError as exc:
        return f"Bad arguments for {tool_call.name}: {exc}", False
    except Exception as exc:
        logger.exception("Tool %s crashed", tool_call.name)
        return f"{tool_call.name} failed: {exc}", False
```

The `TypeError` branch matters more than it looks: a 1.7B model routinely invents argument names, and this turns that into a tool result the model can learn from rather than a crashed turn.

- [ ] **Step 3: Verify the workspace guard exhaustively**

Write to `$SCRATCH/verify_files.py` a script that builds a `ToolContext` over a temp workspace and asserts every one of these is refused with `ToolError`:

```
"../escape.txt"
"../../escape.txt"
"/etc/passwd"
"/tmp/escape.txt"
"subdir/../../escape.txt"
"./../escape.txt"
```

plus a symlink case: create `workspace/link` pointing at `/tmp`, then assert `write_file("link/escape.txt")` is refused. Then assert the allowed cases work: `write_file("a.txt")`, `write_file("sub/dir/b.txt")` (creates parents), `read_file("a.txt")` round-trips, `read_file("missing.txt")` raises `ToolError`, and a >16 KiB file comes back truncated with the marker.

Finally assert nothing escaped for real: `assert not Path("/tmp/escape.txt").exists()`.

Expected: one `OK` line per case and a final `OK: nothing escaped the workspace`. **Do not proceed to Task 7 until every one of these passes** — Task 7 adds a tool that can run anything on this host, and this guard is what keeps the unconfirmed tools from being an easier path around it.

- [ ] **Step 4: Delete the scratch files and commit**

```bash
git add src/oh_my_bot/tools/
git commit -m "feat: add tool registry and workspace-scoped file tools"
```

---

### Task 7: `tools/exec.py` — host command execution

**Files:**
- Create: `src/oh_my_bot/tools/exec.py`
- Modify: `src/oh_my_bot/tools/__init__.py` (register `exec`)

**Interfaces:**
- Produces: `run_command(command, cwd, timeout, max_bytes) -> str` and the `exec` tool handler. The approval gate is added in Task 8; until then `exec` runs unconfirmed, so **do not run the bot against Telegram between Tasks 7 and 8** — verify with the scratch script only.

- [ ] **Step 1: Write `src/oh_my_bot/tools/exec.py`**

```python
import logging
import os
import signal
import subprocess

from dotenv import dotenv_values

logger = logging.getLogger(__name__)

# Every key defined in .env — scrubbed from the environment of any command the model runs.
# Without this, one `cat .env` (or `env`) prints the bot token straight into a chat.
SECRET_KEYS = frozenset(dotenv_values() or {})


def clean_env() -> dict:
    # Returns the current environment minus every secret loaded from .env.
    return {k: v for k, v in os.environ.items() if k not in SECRET_KEYS}


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


def _truncate(text: str, max_bytes: int) -> str:
    # Caps captured output, marking the cut so the model knows the result is incomplete.
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    kept = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return kept + f"\n[output truncated, {len(encoded) - max_bytes} more bytes]"


def exec_tool(ctx, command: str) -> str:
    # The `exec` tool: runs a shell command in the session workspace after approval has been granted.
    return run_command(
        command,
        cwd=ctx.session.workspace(),
        timeout=ctx.config.exec_timeout_seconds,
        max_bytes=ctx.config.exec_max_output_bytes,
    )
```

- [ ] **Step 2: Register `exec` in `src/oh_my_bot/tools/__init__.py`**

Add its schema — one required string parameter `command`, described as "A shell command to run. The working directory does not persist between calls; cd every time you need to." — and add `"exec": exec_tool` to `_HANDLERS`.

- [ ] **Step 3: Verify execution, killing, truncation, and env scrubbing**

Write to `$SCRATCH/verify_exec.py` a script asserting:
- `run_command("echo hi", ...)` returns `hi`.
- `run_command("echo out; echo err >&2", ...)` contains both — stderr is merged.
- `run_command("exit 3", ...)` includes `[exit code 3]`.
- `run_command("pwd", cwd=workspace)` returns the workspace path.
- A timeout: `run_command("sleep 30", timeout=1)` returns in under 3 seconds with `[killed after 1s timeout]`. Measure with `time.monotonic()` and assert the elapsed time — this is what proves the kill actually happened.
- Orphan cleanup: run `"sleep 30 & sleep 30"` with `timeout=1`, then assert `pgrep -f "sleep 30"` finds nothing. This is what `start_new_session` + `killpg` buys you over `process.kill()`.
- Truncation: `run_command("head -c 100000 /dev/zero | tr '\\0' 'a'", max_bytes=100)` returns a marked, truncated result.
- **Env scrubbing:** `run_command("env", ...)` must not contain the literal bot token, and must not contain the string `TELEGRAM_BOT_TOKEN`. Assert both.

Expected: one `OK` line per case. The env-scrubbing and orphan assertions are the two that would otherwise fail silently in production.

- [ ] **Step 4: Delete the scratch file and commit**

```bash
git add src/oh_my_bot/tools/
git commit -m "feat: add exec tool with scrubbed env, process-group kill, and output caps"
```

---

### Task 8: Approvals — inline keyboards and the confirmation gate

**Files:**
- Modify: `src/oh_my_bot/telegram_client.py`, `src/oh_my_bot/tools/__init__.py`
- Create: `src/oh_my_bot/approvals.py`

**Interfaces:**
- Produces: `send_message(..., reply_markup=None)`, `answer_callback_query(token, query_id, text)`, and `ApprovalRegistry` with `request(ctx, command) -> (allowed: bool, reason: str)` and `resolve(update) -> bool`. `main.py` (Task 11) routes `callback_query` updates into `resolve`; `dispatch` gates `exec` through `request`.

- [ ] **Step 1: Extend `src/oh_my_bot/telegram_client.py`**

Add an optional `reply_markup` parameter to `send_message` / `_send_single_message` (included in the JSON body only when set, and only on the final chunk), and:

```python
def answer_callback_query(token: str, query_id: str, text: str = "") -> None:
    # Acknowledges a tapped inline button so Telegram stops showing the spinner on it.
    url = f"https://api.telegram.org/bot{token}/answerCallbackQuery"
    try:
        resp = requests.post(url, json={"callback_query_id": query_id, "text": text}, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Failed to answer callback query: %s", _redact(token, str(exc)))
```

- [ ] **Step 2: Write `src/oh_my_bot/approvals.py`**

```python
import logging
import shlex
import threading
import uuid

from .telegram_client import answer_callback_query, send_message

logger = logging.getLogger(__name__)


def command_pattern(command: str) -> str:
    # Derives the always-allow pattern for a command: its program name.
    # "Always allow ls" therefore permits any later `ls ...`, but not `rm`.
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    return parts[0] if parts else command.strip()


class ApprovalRegistry:
    def __init__(self, token: str, timeout_seconds: int):
        # Tracks in-flight approval requests keyed by a short id that fits Telegram's 64-byte
        # callback_data limit (the command itself is far too long to put in the button).
        self.token = token
        self.timeout_seconds = timeout_seconds
        self._pending = {}
        self._lock = threading.Lock()

    def request(self, ctx, command: str):
        # Asks the user to confirm one command, blocking this chat's actor until they answer
        # or the request times out. Returns (allowed, reason).
        session = ctx.session
        if session.auto_approve():
            return True, "auto-approve on"
        if command_pattern(command) in session.store.load_approval_patterns(session.session_id):
            return True, "pattern previously allowed"

        request_id = uuid.uuid4().hex[:12]
        event = threading.Event()
        with self._lock:
            self._pending[request_id] = {"event": event, "verdict": None, "command": command, "ctx": ctx}

        keyboard = {
            "inline_keyboard": [[
                {"text": "Allow", "callback_data": f"a:{request_id}"},
                {"text": "Deny", "callback_data": f"d:{request_id}"},
                {"text": f"Always allow {command_pattern(command)}", "callback_data": f"p:{request_id}"},
            ]]
        }
        send_message(self.token, session.chat_id, f"Run this command?\n\n{command}", reply_markup=keyboard)

        if not event.wait(self.timeout_seconds):
            with self._lock:
                self._pending.pop(request_id, None)
            send_message(self.token, session.chat_id, "Approval timed out; treating it as a denial.")
            return False, "approval timed out"

        with self._lock:
            entry = self._pending.pop(request_id, None)
        verdict = entry["verdict"] if entry else "deny"
        if verdict == "pattern":
            session.store.add_approval_pattern(session.session_id, command_pattern(command))
            return True, "allowed, pattern remembered"
        return verdict == "allow", "allowed" if verdict == "allow" else "denied by the user"

    def resolve(self, update: dict) -> bool:
        # Handles a tapped button: records the verdict and wakes the blocked actor. Returns
        # False if the request is unknown (a stale button from before a restart).
        query = update["callback_query"]
        data = query.get("data") or ""
        action, _, request_id = data.partition(":")
        with self._lock:
            entry = self._pending.get(request_id)
        if entry is None:
            answer_callback_query(self.token, query["id"], "That request has expired.")
            return False
        entry["verdict"] = {"a": "allow", "d": "deny", "p": "pattern"}.get(action, "deny")
        answer_callback_query(self.token, query["id"], entry["verdict"].capitalize())
        entry["event"].set()
        return True
```

- [ ] **Step 3: Gate `exec` in `src/oh_my_bot/tools/__init__.py`**

In `dispatch`, before invoking the handler, if `tool_call.name == "exec"`, call `ctx.approvals.request(ctx, tool_call.arguments.get("command", ""))`. On denial return `("The user denied this command.", False)` — the `False` charges it to the tool-failure breaker, so a model that keeps re-asking is stopped, while a model that adapts gets a second route. Add `approvals` to `ToolContext`.

- [ ] **Step 4: Verify the approval flow without Telegram**

Write to `$SCRATCH/verify_approvals.py` a script that monkeypatches `approvals.send_message` and `approvals.answer_callback_query` to record calls instead of hitting the network, then asserts:
- `command_pattern("ls -la /tmp")` is `"ls"`; `command_pattern("")` does not raise; `command_pattern("echo \"unclosed")` falls back to a split rather than raising on the `shlex` error.
- Every `callback_data` produced is ≤ 64 bytes, including for a very long command. (Telegram silently rejects longer values — this is the bug you would otherwise find in production.)
- Allow: call `request` on a thread, resolve with `a:<id>`, assert `(True, ...)`.
- Deny: resolve with `d:<id>`, assert `(False, "denied by the user")`.
- Pattern: resolve with `p:<id>`, assert allowed **and** that a second `request` for `ls -la` returns immediately without sending a message.
- Auto-approve: `set_auto_approve(True)` then assert `request` returns immediately with no message sent.
- Timeout: build a registry with `timeout_seconds=1`, never resolve, and assert it returns `(False, "approval timed out")` in about a second and that the pending entry is cleaned up (`registry._pending == {}`).
- Stale button: `resolve` an unknown id returns `False` and does not raise.

Expected: one `OK` line per case.

- [ ] **Step 5: Delete the scratch file and commit**

```bash
git add src/oh_my_bot/approvals.py src/oh_my_bot/telegram_client.py src/oh_my_bot/tools/
git commit -m "feat: gate exec behind per-command Telegram approval"
```

---

### Task 9: `actors.py` — one thread per chat

**Files:**
- Create: `src/oh_my_bot/actors.py`

**Interfaces:**
- Produces: `ActorPool(handler)` with `submit(chat_id, item)`. Replaces `ThreadPoolExecutor` and `ChatLocks` in `main.py` (Task 11).

- [ ] **Step 1: Write `src/oh_my_bot/actors.py`**

A dict of `chat_id -> (Queue, Thread)` guarded by a registry lock (the same pattern `ChatLocks` already uses), created on first message. Each actor thread loops forever on `queue.get()` and calls `handler(item)`, catching and logging every exception so one bad turn never kills the actor. Threads are daemons and live for the process lifetime.

FIFO ordering falls out of the queue: a message arriving mid-turn waits, exactly as the per-chat lock did before — but now the wait costs no shared resource, which is what makes blocking on an approval safe.

- [ ] **Step 2: Delete `ChatLocks` from `src/oh_my_bot/worker.py`**

The actor's queue subsumes it. `worker.py` is now just `run_llm_call` and its child-process target.

- [ ] **Step 3: Verify ordering, isolation, and crash resilience**

Write to `$SCRATCH/verify_actors.py` a script asserting:
- Ten items submitted to one chat are handled in submission order.
- Two chats make progress concurrently: chat A's handler blocks on an `Event` for 2s; assert chat B's items complete while A is still blocked. This is the property the thread pool could not give you once approvals block.
- A handler that raises on one item does not stop that actor from processing the next.
- Submitting to 50 distinct chat ids creates 50 live actors without error.

Expected: one `OK` line per case.

- [ ] **Step 4: Delete the scratch file and commit**

```bash
git add src/oh_my_bot/actors.py src/oh_my_bot/worker.py
git commit -m "feat: replace the thread pool with one actor thread per chat"
```

---

### Task 10: `agent.py` — the loop and its three breakers

**Files:**
- Create: `src/oh_my_bot/agent.py`

**Interfaces:**
- Consumes: everything from Tasks 5-9.
- Produces: `run_turn(text, session, connector, config, approvals, token) -> None`, which drives one user message to completion and sends every reply itself.

The spec's build order put breakers in a later phase; they ship here instead. An unbounded loop that can call `exec` on this host is not a state to leave the code in between commits.

- [ ] **Step 1: Write `src/oh_my_bot/agent.py`**

```python
import logging
import time

from .session import handle_command
from .telegram_client import send_message
from .tools import TOOL_SCHEMAS, ToolContext, dispatch
from .worker import run_llm_call

logger = logging.getLogger(__name__)

PROGRESS_RESULT_CHARS = 500


def run_turn(text, session, connector, config, approvals, token) -> None:
    # Drives one user message to a final answer: model call, tool calls, repeat, under three
    # independent circuit breakers. Sends every message to the chat itself.
    command_reply = handle_command(text, session)
    if command_reply is not None:
        send_message(token, session.chat_id, command_reply)
        return

    session.add_user(text)
    ctx = ToolContext(session=session, config=config, approvals=approvals)
    iterations = 0
    llm_failures = 0
    tool_failures = 0

    while True:
        if iterations >= config.max_loop_iterations:
            send_message(token, session.chat_id, f"I hit my step limit after {iterations} steps. Send another message to continue from here.")
            return

        status, payload = run_llm_call(connector, session.history(), TOOL_SCHEMAS, config.llm_timeout_seconds)
        if status != "ok":
            llm_failures += 1
            logger.error("LLM call failed (%s/%s): %s", llm_failures, config.max_llm_retries, payload)
            if llm_failures >= config.max_llm_retries:
                send_message(token, session.chat_id, "I couldn't reach the AI service. Please try again shortly.")
                return
            # A transport failure is not a reasoning step, so it does not burn a loop iteration.
            time.sleep(min(2 ** llm_failures, 10))
            continue
        llm_failures = 0
        message = payload

        if not message.tool_calls:
            session.add_assistant(message.content)
            send_message(token, session.chat_id, message.content)
            return

        session.add_assistant(message.content, tool_calls=[c.to_wire() for c in message.tool_calls])
        for tool_call in message.tool_calls:
            output, ok = dispatch(tool_call, ctx)
            _send_progress(token, session.chat_id, tool_call, output)
            session.add_tool_result(tool_call.id, output)
            if ok:
                tool_failures = 0
            else:
                tool_failures += 1
                if tool_failures >= config.max_consecutive_tool_failures:
                    send_message(token, session.chat_id, f"A tool kept failing ({tool_failures} times in a row); stopping here.")
                    return
        iterations += 1


def _send_progress(token, chat_id, tool_call, output) -> None:
    # Posts one progress message per tool call: what was run, and a trimmed view of what came back.
    if tool_call.name == "exec":
        header = f"$ {tool_call.arguments.get('command', '')}"
    else:
        header = f"{tool_call.name}({', '.join(f'{k}=...' for k in tool_call.arguments)})"
    body = output if len(output) <= PROGRESS_RESULT_CHARS else output[:PROGRESS_RESULT_CHARS] + "\n[...]"
    send_message(token, chat_id, f"{header}\n\n{body}")
```

Note the ordering inside the tool loop: the assistant turn carrying `tool_calls` is persisted **before** any tool runs. If the process dies mid-tool, the history still contains a tool call with no result — which is malformed for a follow-up request, and is why a restart abandons in-flight turns rather than resuming them.

- [ ] **Step 2: Verify every loop path with a fake connector**

Write to `$SCRATCH/verify_agent.py` a script with a scripted fake connector (returns a queued list of `AssistantMessage`s) and monkeypatched `send_message`, asserting:
- **Happy path:** message with no tool calls → one send, history ends with an assistant message.
- **One tool round:** tool call → result → final answer. Assert the history is exactly `user, assistant(tool_calls), tool, assistant`, that the `tool` message's `tool_call_id` matches the call's id, and that a progress message was sent before the final answer.
- **Iteration breaker:** a connector that always returns a tool call. Assert exactly `max_loop_iterations` rounds ran and the "step limit" message was sent.
- **LLM retry:** a connector failing twice then succeeding. Assert the turn still completes and that the failures did **not** consume loop iterations.
- **LLM breaker:** a connector that always fails. Assert the "couldn't reach" message after `max_llm_retries` attempts.
- **Tool breaker:** a tool call to an unknown tool, repeatedly. Assert it stops after `max_consecutive_tool_failures`.
- **Failure counter resets:** fail, succeed, fail, succeed with `max_consecutive_tool_failures=2` — assert the turn is *not* killed, proving the counter is consecutive rather than cumulative.
- **Denial:** an approvals stub that always denies. Assert the tool result is `"The user denied this command."` and that repeated denials trip the tool breaker.
- **Commands short-circuit:** `run_turn("/status", ...)` sends the status and never calls the connector.

Expected: one `OK` line per case.

- [ ] **Step 3: Delete the scratch file and commit**

```bash
git add src/oh_my_bot/agent.py
git commit -m "feat: add the agentic loop with iteration, transport, and tool breakers"
```

---

### Task 11: `main.py` — the router

**Files:**
- Modify: `src/oh_my_bot/main.py`, `src/oh_my_bot/worker.py`

**Interfaces:**
- Consumes: everything above. Produces the runnable agent. `handle_update` is deleted from `worker.py`; `agent.run_turn` replaces it.

- [ ] **Step 1: Rewrite `main()` in `src/oh_my_bot/main.py`**

It now:
1. Loads config, builds the `Store` and calls `init_schema()`, builds the connector, the `ApprovalRegistry`, and an `ActorPool` whose handler builds a `Session` and calls `agent.run_turn`.
2. Long-polls with the existing capped backoff.
3. Routes each update:
   - `callback_query` → `approvals.resolve(update)`. **Route this before the message checks** — a callback query has no `update["message"]`, and the allowlist check must read `callback_query["from"]["id"]` for it.
   - `message` with text → allowlist and `chat.type == "private"` check, then `actors.submit(chat_id, ...)`.
   - anything else → ignored.
4. Advances `offset` immediately after each poll batch, as before.

- [ ] **Step 2: Delete `handle_update` and `ChatLocks` remnants from `src/oh_my_bot/worker.py`**

`worker.py` should now contain only `run_llm_call` and `_llm_call_target`.

- [ ] **Step 3: Verify routing with fake updates**

Write to `$SCRATCH/verify_router.py` a script that extracts the routing decision into the same shape `main.py` uses and asserts: an allowlisted private message is submitted; a non-allowlisted message is dropped; a group message from an allowlisted user is dropped; a callback query from an allowlisted user reaches `approvals.resolve`; a callback query from a **non**-allowlisted user is dropped (this is the one that would otherwise let anyone approve your commands by guessing a request id); an update with neither key is ignored without raising.

Expected: one `OK` line per case.

- [ ] **Step 4: End-to-end verification (manual, human required)**

Start the LLM server and run `uv run oh-my-bot`. From Telegram:
- Ask "what files are in the current directory?" — confirm an approval keyboard appears, tap **Allow**, confirm a `$ ls` progress message and then a final answer.
- Ask something requiring two steps ("count the lines in every .txt file here" after creating a couple) — confirm multiple approvals and progress messages.
- Tap **Deny** on one — confirm the model is told and either adapts or stops cleanly.
- Tap **Always allow ls**, then ask something that runs `ls` again — confirm no second prompt.
- Send `/auto`, confirm commands stop prompting; send `/new`, confirm prompting returns.
- Trigger the step limit (ask for something open-ended) — confirm the "step limit" message and that a follow-up message continues from where it stopped.
- Leave an approval unanswered past `APPROVAL_TIMEOUT_SECONDS` (set it to 30 temporarily) — confirm the timeout message and that the chat accepts new messages afterward.
- With an approval pending in your chat, message from a second allowlisted account — confirm it gets a reply while the first is still blocked.
- Ask it to `cat .env` — confirm no bot token appears in the output.

- [ ] **Step 5: Commit**

```bash
git add src/oh_my_bot/main.py src/oh_my_bot/worker.py
git commit -m "feat: route updates to per-chat agents and pending approvals"
```

---
## Phase 3 — Traces

### Task 12: Record every model exchange

**Files:**
- Modify: `src/oh_my_bot/worker.py`, `src/oh_my_bot/agent.py`

**Interfaces:**
- Consumes: `Store.append_trace` (Task 2). Produces: a queryable record of every request/response pair.

Debugging a 1.7B model means constantly asking "what did it actually see, and what did it actually say?" Reconstructing that from Telegram messages is guesswork; this makes it a SQL query.

- [ ] **Step 1: Capture the raw exchange**

`run_llm_call` already crosses a process boundary, so the child cannot write to the store (its connection is thread-local to the parent). Instead, have `_llm_call_target` return the raw response JSON alongside the parsed `AssistantMessage`, and have `agent.run_turn` — which runs in the actor thread and owns the connection — call `session.store.append_trace(session.session_id, request, response)` after each call.

Record the messages sent and the raw response, including on the failure paths: a trace of a call that failed is often the more useful one.

- [ ] **Step 2: Verify traces are written and readable**

Run a scripted turn (reuse the fake connector from Task 10), then:

```bash
uv run python -c "
import json, sqlite3
conn = sqlite3.connect('oh-my-bot.db')
rows = conn.execute('SELECT session_id, request, response FROM traces ORDER BY id DESC LIMIT 5').fetchall()
assert rows, 'no traces recorded'
for sid, req, resp in rows:
    print(sid[:8], len(json.loads(req)), 'messages sent')
print('OK: traces recorded')
"
```
Expected: one line per trace and `OK: traces recorded`.

- [ ] **Step 3: Add a convenience query to the README**

A short "Debugging" section showing how to dump the last exchange for a session with `sqlite3`. You will use this constantly.

- [ ] **Step 4: Commit**

```bash
git add src/oh_my_bot/worker.py src/oh_my_bot/agent.py README.md
git commit -m "feat: trace every LLM request and response to SQLite"
```

---

## Phase 4 — Compaction

### Task 13: `context.py` — token estimation with usage correction

**Files:**
- Create: `src/oh_my_bot/context.py`

**Interfaces:**
- Produces: `estimate_tokens(messages, ratio) -> int`, `message_chars(messages) -> int`, `update_ratio(store, model, messages, usage) -> float`. Consumed by Task 14 and by `agent.py`.

- [ ] **Step 1: Write the estimator and the calibration**

```python
def message_chars(messages) -> int:
    # Counts the characters that will actually be serialized into the request body,
    # including tool-call JSON, not just the visible content.
    return sum(len(json.dumps(m, default=str)) for m in messages)


def estimate_tokens(messages, ratio: float) -> int:
    # Estimates the prompt size in tokens using the learned characters-per-token ratio.
    return int(message_chars(messages) / max(ratio, 0.5))


def update_ratio(store, model: str, messages, usage: dict) -> float:
    # Corrects the characters-per-token ratio from the backend's own reported prompt_tokens,
    # so the estimate converges on whatever model is actually loaded. Returns the new ratio.
    prompt_tokens = (usage or {}).get("prompt_tokens")
    if not prompt_tokens:
        return store.get_token_ratio(model)
    observed = message_chars(messages) / prompt_tokens
    current = store.get_token_ratio(model)
    updated = current * 0.7 + observed * 0.3
    store.set_token_ratio(model, updated)
    return updated
```

Call `update_ratio` from `agent.run_turn` after every successful model call, passing the messages that were sent and `message.usage`.

- [ ] **Step 2: Verify convergence**

Write to `$SCRATCH/verify_context.py` a script that feeds `update_ratio` twenty synthetic responses whose `prompt_tokens` imply a true ratio of 3.0, starting from the 4.0 default, and asserts the stored ratio ends within 5% of 3.0. Also assert that a response with no `usage` leaves the ratio untouched (Ollama omits it in some configurations), and that `estimate_tokens` never divides by zero when the ratio is degenerate.

Expected: three `OK` lines.

- [ ] **Step 3: Delete the scratch file and commit**

```bash
git add src/oh_my_bot/context.py src/oh_my_bot/agent.py
git commit -m "feat: estimate prompt tokens and calibrate from reported usage"
```

---

### Task 14: Tiered compaction

**Files:**
- Modify: `src/oh_my_bot/context.py`, `src/oh_my_bot/agent.py`, `src/oh_my_bot/session.py`

**Interfaces:**
- Produces: `compact(messages, budget, ratio, summarize) -> (messages, note)`. Called from `agent.run_turn` before **every** model call, including mid-loop.

- [ ] **Step 1: Implement the three tiers in `src/oh_my_bot/context.py`**

Tiers run in order, each stopping as soon as the estimate is under budget:

1. **Squeeze tool outputs.** Walk the history oldest-first, replacing the `content` of `tool` messages with `[N characters elided]`, keeping the most recent 2 tool results intact. No model call, no lost structure.
2. **Trim.** Drop the oldest turns. Pin the system message and the first user message.
3. **Summarize.** Ask the model to summarize the dropped span, then splice the summary in as a single message right after the pinned pair.

**The invariant that makes trimming safe:** an assistant message carrying `tool_calls` and the `tool` messages answering it must be dropped together. A `tool` message whose originating call is gone — or an assistant call whose results are gone — is a malformed conversation, and strict backends reject the whole request with a 400. Implement trimming by only ever cutting at the boundary *before* a `user` message, never mid-block:

```python
def _safe_cut_points(messages, pinned: int) -> list:
    # Returns indices where the history can be cut without orphaning a tool call from its
    # results: only immediately before a user message, past the pinned prefix.
    return [i for i in range(pinned, len(messages)) if messages[i]["role"] == "user"]
```

- [ ] **Step 2: Wire it into `agent.run_turn`**

Before each `run_llm_call`, compute `budget = config.llm_context_tokens * config.compact_threshold_pct // 100`; if `estimate_tokens(history, ratio) > budget`, compact, persist the compacted history with `store.replace_messages`, and post a short note to the chat ("Compacted the conversation to stay within the context window."). Add a `/compact` command to `handle_command` that forces a pass.

Because this runs before every call rather than once per user message, a single tool-heavy turn cannot overflow on its own.

- [ ] **Step 3: Verify each tier in isolation and in sequence**

Write to `$SCRATCH/verify_compact.py` a script asserting:
- **Tier 1 alone:** a history whose bulk is old tool output drops under budget with tier 1 only; the two most recent tool results are still intact; no summarizer was called.
- **Tier 2:** a history of many short user/assistant turns trims to under budget; the system message and first user message survive.
- **Tier 3:** a history that cannot fit after trimming invokes the summarizer exactly once and splices its output in.
- **The invariant, exhaustively:** build a history containing several `assistant(tool_calls) → tool → tool` blocks, compact it at many different budgets (loop the budget from tiny to large), and after **each** run assert that every remaining `tool` message has a preceding assistant message whose `tool_calls` contains its `tool_call_id`, and that no remaining assistant `tool_calls` entry lacks its result. This is the assertion that catches the 400-on-malformed-history bug before Telegram does.
- **Idempotence:** compacting an already-compact history returns it unchanged.

Expected: one `OK` line per case.

- [ ] **Step 4: Verify end-to-end (manual, human required)**

Temporarily set `LLM_CONTEXT_TOKENS=1500` and run a long tool-heavy conversation. Confirm the compaction notice appears, the bot stays coherent afterwards, and no backend 400s appear in the logs. Restore the real value — and set it to match your server's actual configured window, not the model's advertised maximum. Ollama defaults `num_ctx` to 4096 regardless of the model.

- [ ] **Step 5: Delete the scratch file and commit**

```bash
git add src/oh_my_bot/context.py src/oh_my_bot/agent.py src/oh_my_bot/session.py
git commit -m "feat: compact context in tiers before it overflows"
```

---

## Phase 5 — Skills

Skills come last because their value depends on a loop you already trust, and because progressive disclosure is the choice most dependent on model quality — you want everything else working before you find out how well a 1.7B model picks a skill.

### Task 15: `skills.py` — discovery and loading

**Files:**
- Create: `src/oh_my_bot/skills.py`, `skills/check-disk/SKILL.md` (an example)

**Interfaces:**
- Produces: `Skill` (`name`, `description`, `dir`, `body_path`), `parse_frontmatter(text) -> (dict, body)`, `load_skills(skills_dir) -> dict`, `skill_index(skills) -> str`, `read_skill_body(skill) -> str`.

- [ ] **Step 1: Write the loader**

`parse_frontmatter` handles a leading `---` block of `key: value` lines by hand — no PyYAML. Unknown keys are parsed and ignored, leaving room to honor `allowed-tools` later without a format change. A `SKILL.md` missing `name` or `description` is skipped with a warning rather than crashing startup; the directory name is the fallback for `name`.

`load_skills` scans `SKILLS_DIR/*/SKILL.md`, reading **only** the frontmatter at startup. Bodies are read on demand — that is the whole point of progressive disclosure, and reading them eagerly here would quietly undo it.

`skill_index` renders one `- name: description` line per skill for the system prompt.

- [ ] **Step 2: Write an example skill at `skills/check-disk/SKILL.md`**

```markdown
---
name: check-disk
description: Investigate disk usage and find what is taking up space.
---

To investigate disk usage:

1. Run `df -h` to see which filesystem is full.
2. Run `du -sh */ | sort -rh | head -20` from the relevant directory to find the largest subdirectories.
3. Descend into the largest one and repeat until you find the cause.

Report the top few offenders with their sizes. Do not delete anything.
```

- [ ] **Step 3: Verify the loader**

Assert against a temp skills dir: a well-formed skill loads with the right name and description; a `SKILL.md` with no frontmatter is skipped without raising; one missing `description` is skipped; unknown frontmatter keys are ignored rather than fatal; `skill_index` includes every loaded skill; `read_skill_body` returns the body **without** the frontmatter block; and a missing `SKILLS_DIR` yields an empty dict rather than an error.

Expected: one `OK` line per case.

- [ ] **Step 4: Commit**

```bash
git add src/oh_my_bot/skills.py skills/
git commit -m "feat: add SKILL.md discovery and frontmatter parsing"
```

---

### Task 16: The `skill` tool and progressive disclosure

**Files:**
- Create: `src/oh_my_bot/tools/skill.py`
- Modify: `src/oh_my_bot/tools/__init__.py`, `src/oh_my_bot/session.py`, `src/oh_my_bot/main.py`

**Interfaces:**
- Produces: the `skill` tool, the skill index in the system prompt, and the `/skills` and `/skill <name>` commands.

- [ ] **Step 1: Put the index in the system prompt**

`Session` takes the loaded skills dict; `system_prompt()` appends the index under a heading explaining that these are available and that calling `skill(name)` loads the full instructions. Only names and descriptions go in — never bodies.

- [ ] **Step 2: Write `src/oh_my_bot/tools/skill.py`**

The handler takes a `name`, returns the body, and appends a line naming the skill's directory so the model can invoke sibling scripts through `exec`:

```python
def skill_tool(ctx, name: str) -> str:
    # Loads a skill's full instructions on demand, telling the model where its scripts live.
    skill = ctx.session.skills.get(name)
    if skill is None:
        available = ", ".join(sorted(ctx.session.skills)) or "none"
        raise ToolError(f"No skill named {name!r}. Available: {available}")
    return f"{read_skill_body(skill)}\n\nThis skill's files are in: {skill.dir}"
```

An unknown name raises `ToolError`, so a wrong guess becomes a tool result listing the real options — which is the cheapest correction available to a small model.

Running a skill's script goes through `exec`, so it hits the normal approval gate. That is deliberate: a bundled script is still arbitrary code.

- [ ] **Step 3: Add `/skills` and `/skill <name>`**

`/skills` lists the index. `/skill <name>` loads the body directly into the session history as a user message, bypassing the model's choice entirely — the override for when it picks wrong, and the way to tell whether its own choosing works at all.

- [ ] **Step 4: Verify**

Assert: the system prompt contains every skill's description and **no** skill body; `skill_tool` returns the body plus the directory line; an unknown name raises `ToolError` listing the available names; `/skills` lists them; `/skill check-disk` appends the body to the history; `/skill nope` returns a clear error without touching the history.

- [ ] **Step 5: Verify end-to-end (manual, human required)**

Ask the bot "my disk is full, can you look into it?" and see whether it calls `skill("check-disk")` on its own. Then ask the same thing after `/skill check-disk`. If the model never picks the skill unaided, that is a finding about progressive disclosure at this model size, not a bug — record it in the README and use the explicit command.

- [ ] **Step 6: Commit**

```bash
git add src/oh_my_bot/tools/ src/oh_my_bot/session.py src/oh_my_bot/main.py
git commit -m "feat: load skills on demand via the skill tool and /skill"
```

---

### Task 17: Update the README and close out the spec

**Files:**
- Modify: `README.md`, `docs/superpowers/specs/2026-08-28-agentic-harness-design.md`

- [ ] **Step 1: Rewrite the README's architecture and limitations sections**

Update the file table for the new modules, document every new environment variable, and add sections for: the user allowlist and how to find your Telegram user id; the approval flow and its escape hatches; writing a skill; the `sqlite3` debugging queries; and how to set `LLM_CONTEXT_TOKENS` correctly for your backend.

Replace "No conversation memory" in Known limitations with the real remaining ones: DMs only; an in-flight turn is abandoned on restart; `exec` runs unsandboxed on the host by design, with the user allowlist and per-command approval as the only boundary; progressive disclosure depends on model quality.

- [ ] **Step 2: Record the resolved open questions in the spec**

Replace the spec's "Open questions" section with the decisions table from this plan, plus anything you decided differently during implementation. A spec that still asks questions the code has answered is worse than no spec.

- [ ] **Step 3: Full manual regression**

Walk the spec's Testing section end to end. The security-relevant ones are not optional: allowlist rejection, workspace escape, env scrubbing, and approval expiry.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/
git commit -m "docs: document the agentic harness and record resolved decisions"
```

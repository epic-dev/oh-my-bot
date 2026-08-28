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
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    def init_schema(self) -> None:
        # Creates every table and index if they don't already exist.
        conn = self.connect()
        conn.executescript(SCHEMA)
        conn.commit()

    def get_or_create_session(self, chat_id: int) -> str:
        # Returns the chat's active session id, creating a first session if it has none.
        row = self.connect().execute(
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
            "INSERT INTO sessions (session_id, chat_id, active, auto_approve, created_at)"
            " VALUES (?, ?, 1, 0, ?)",
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
            (
                session_id,
                role,
                content,
                json.dumps(tool_calls) if tool_calls else None,
                tool_call_id,
                time.time(),
            ),
        )
        conn.commit()

    def load_messages(self, session_id: str) -> list:
        # Loads a session's full history in order, as chat-completion message dicts.
        rows = self.connect().execute(
            "SELECT role, content, tool_calls, tool_call_id FROM messages"
            " WHERE session_id = ? ORDER BY id",
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
        now = time.time()
        rows = [
            (
                session_id,
                m["role"],
                m.get("content"),
                json.dumps(m["tool_calls"]) if m.get("tool_calls") else None,
                m.get("tool_call_id"),
                now,
            )
            for m in messages
        ]
        with conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.executemany(
                "INSERT INTO messages (session_id, role, content, tool_calls, tool_call_id, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )

    def get_auto_approve(self, session_id: str) -> bool:
        # Reports whether the session has confirmation suspended via /auto.
        row = self.connect().execute(
            "SELECT auto_approve FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return bool(row and row["auto_approve"])

    def set_auto_approve(self, session_id: str, on: bool) -> None:
        # Turns per-command confirmation off (or back on) for the rest of the session.
        conn = self.connect()
        conn.execute(
            "UPDATE sessions SET auto_approve = ? WHERE session_id = ?",
            (1 if on else 0, session_id),
        )
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
        row = self.connect().execute(
            "SELECT ratio FROM token_ratios WHERE model = ?", (model,)
        ).fetchone()
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

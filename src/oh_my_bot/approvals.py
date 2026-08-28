import logging
import shlex
import threading
import uuid

from .telegram_client import answer_callback_query, send_message

logger = logging.getLogger(__name__)

# Telegram caps callback_data at 64 bytes, so the button carries a short request id and the
# command itself stays in the registry.
_REQUEST_ID_CHARS = 12
_PATTERN_LABEL_CHARS = 24

_VERDICTS = {"a": "allow", "d": "deny", "p": "pattern"}


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
        # callback_data limit.
        self.token = token
        self.timeout_seconds = timeout_seconds
        self._pending = {}
        self._lock = threading.Lock()

    def request(self, ctx, command: str):
        # Asks the user to confirm one command, blocking this chat's actor until they answer or
        # the request times out. Returns (allowed, reason). Blocking is safe because each chat has
        # its own actor thread: a chat waiting on a tap delays only itself.
        session = ctx.session
        if session.auto_approve():
            return True, "auto-approve is on"
        pattern = command_pattern(command)
        if pattern in session.store.load_approval_patterns(session.session_id):
            return True, f"{pattern} was previously always-allowed"

        request_id = uuid.uuid4().hex[:_REQUEST_ID_CHARS]
        event = threading.Event()
        with self._lock:
            self._pending[request_id] = {"event": event, "verdict": None}

        label = pattern if len(pattern) <= _PATTERN_LABEL_CHARS else pattern[:_PATTERN_LABEL_CHARS] + "…"
        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "Allow", "callback_data": f"a:{request_id}"},
                    {"text": "Deny", "callback_data": f"d:{request_id}"},
                ],
                [{"text": f"Always allow {label}", "callback_data": f"p:{request_id}"}],
            ]
        }
        send_message(self.token, session.chat_id, f"Run this command?\n\n{command}", reply_markup=keyboard)

        if not event.wait(self.timeout_seconds):
            with self._lock:
                self._pending.pop(request_id, None)
            send_message(self.token, session.chat_id, "Approval timed out; treating it as a denial.")
            return False, "the approval request timed out"

        with self._lock:
            entry = self._pending.pop(request_id, None)
        verdict = entry["verdict"] if entry else "deny"
        if verdict == "pattern":
            session.store.add_approval_pattern(session.session_id, pattern)
            return True, f"allowed, and {pattern} will not be asked about again"
        if verdict == "allow":
            return True, "allowed by the user"
        return False, "the user denied it"

    def resolve(self, update: dict) -> bool:
        # Handles a tapped button: records the verdict and wakes the blocked actor. Returns False
        # if the request is unknown — a stale button from before a restart, or one already
        # answered — so the caller can tell the difference.
        query = update["callback_query"]
        action, _, request_id = (query.get("data") or "").partition(":")
        with self._lock:
            entry = self._pending.get(request_id)
        if entry is None:
            answer_callback_query(self.token, query["id"], "That request has expired.")
            return False
        verdict = _VERDICTS.get(action, "deny")
        entry["verdict"] = verdict
        answer_callback_query(self.token, query["id"], verdict.capitalize())
        entry["event"].set()
        return True

import logging
import shutil
import time
from pathlib import Path

from .skills import read_skill_body, skill_index

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant operating on the user's computer via a Telegram bot.

You have tools available. Use them when the task requires acting on the system;
answer directly when it does not. After a tool returns, use its result to
continue. When you have the final answer, reply with plain text and no tool call.

Working directory: {workspace}

read_file and write_file only accept paths INSIDE that directory. Always pass a
plain relative path such as notes.txt or logs/output.txt. Never start a path
with / or ../ — those are refused. To reach anything outside the working
directory, use the exec tool instead.
"""

SKILLS_PROMPT = """

Skills available. Each is a set of instructions for a kind of task.
Call the `skill` tool with a name to load its full instructions before starting that task:

{index}
"""


class Session:
    def __init__(self, chat_id: int, store, config, skills=None):
        # Binds a chat to its active session id, the store that persists it, and the skills
        # discovered at startup.
        self.chat_id = chat_id
        self.store = store
        self.config = config
        self.skills = skills or {}
        self.session_id = store.get_or_create_session(chat_id)

    def _workspace_path(self, session_id: str) -> Path:
        # Builds the workspace path for one session id without creating it.
        return Path(self.config.workspace_root).resolve() / str(self.chat_id) / session_id

    def workspace(self) -> Path:
        # Returns this session's workspace directory, creating it on first use.
        path = self._workspace_path(self.session_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def system_prompt(self) -> str:
        # Builds the system message for this session: the standing instructions plus, when any
        # skills exist, their names and descriptions only. Bodies are loaded on demand by the
        # `skill` tool — putting them here would defeat progressive disclosure.
        prompt = SYSTEM_PROMPT.format(workspace=self.workspace())
        if self.skills:
            prompt += SKILLS_PROMPT.format(index=skill_index(self.skills))
        return prompt

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

    def replace_history(self, messages) -> None:
        # Persists a rewritten history (used by compaction). The leading system message is the
        # one system_prompt() generates fresh each turn, so it is stripped before storing —
        # storing it would duplicate it on every subsequent history() call.
        body = list(messages)
        if body and body[0].get("role") == "system":
            body = body[1:]
        self.store.replace_messages(self.session_id, body)

    def reset(self) -> None:
        # Starts a fresh session: new context, new empty workspace, auto-approve off, patterns cleared.
        old_workspace = self._workspace_path(self.session_id)
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


def handle_command(text: str, session: Session):
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
    if command == "/skills":
        if not session.skills:
            return "No skills are installed."
        return "Available skills:\n" + skill_index(session.skills)
    if command == "/skill":
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            return "Usage: /skill <name>. Use /skills to list them."
        name = parts[1].strip()
        skill = session.skills.get(name)
        if skill is None:
            available = ", ".join(sorted(session.skills)) or "none"
            return f"No skill named {name!r}. Available: {available}"
        # Loaded straight into the history, bypassing the model's own choice. This is both the
        # override for when it picks wrong and the way to tell whether its choosing works at all.
        session.store.append_message(
            session.session_id, "user",
            f"Follow these instructions:\n\n{read_skill_body(skill)}\n\n"
            f"This skill's files are in: {skill.dir}",
        )
        return f"Loaded the {name} skill into this conversation."
    if command == "/status":
        messages = session.store.load_messages(session.session_id)
        return (
            f"Session: {session.session_id[:8]}\n"
            f"Messages: {len(messages)}\n"
            f"Auto-approve: {'on' if session.auto_approve() else 'off'}\n"
            f"Workspace: {session.workspace()}"
        )
    return f"Unknown command: {command}"

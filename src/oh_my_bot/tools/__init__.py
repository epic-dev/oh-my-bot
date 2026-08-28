import logging
from dataclasses import replace

from .base import ToolContext, ToolError
from .exec import exec_tool
from .files import read_file, write_file
from .skill import skill_tool

logger = logging.getLogger(__name__)

__all__ = ["TOOL_SCHEMAS", "ToolContext", "ToolError", "dispatch", "tool_schemas"]

# OpenAI function schemas advertised to the model. exec is registered in Task 7 and skill in
# Task 16; both append to these two structures rather than replacing them.
SKILL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "skill",
        "description": (
            "Load the full instructions for one of the available skills, by name. Call this "
            "before attempting a task a skill covers, then follow the instructions it returns."
        ),
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "The skill's name."}},
            "required": ["name"],
        },
    },
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "exec",
            "description": (
                "Run a shell command on the user's computer. The working directory does not "
                "persist between calls: cd every time you need to. stderr is merged into the "
                "output. Long output is truncated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run."}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a UTF-8 text file from the working directory. Paths are relative to the "
                "working directory and cannot escape it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the working directory."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write a UTF-8 text file in the working directory, creating parent directories as "
                "needed. Paths are relative to the working directory and cannot escape it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the working directory."},
                    "content": {"type": "string", "description": "The full text to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
]

_HANDLERS = {
    "skill": skill_tool,
    "exec": exec_tool,
    "read_file": read_file,
    "write_file": write_file,
}


# Tools that cannot run until the user confirms them. Everything else is safe unconfirmed only
# because it is code-scoped to the session workspace (see tools/files.py).
APPROVAL_REQUIRED = frozenset({"exec"})


def _approve(tool_call, ctx):
    # Obtains permission to run one gated tool call, returning (allowed, reason). Fails closed:
    # a context with no approval registry refuses rather than running unconfirmed.
    if tool_call.name not in APPROVAL_REQUIRED:
        return True, ""
    if ctx.approvals is None:
        logger.error("No approval registry on the context; refusing to run %s", tool_call.name)
        return False, "no approval channel is available"
    return ctx.approvals.request(ctx, tool_call.arguments.get("command", ""))


def dispatch(tool_call, ctx):
    # Runs one tool call and returns (output, ok); ok=False marks a failure for the breaker.
    # Every failure mode becomes a tool result the model can read and react to, because a crashed
    # turn teaches it nothing and a small model invents argument names constantly.
    handler = _HANDLERS.get(tool_call.name)
    if handler is None:
        # Small models routinely call a tool named after the skill they want, rather than calling
        # `skill` with that name. Observed with Qwen3-1.7B on the first real test. Treating it as
        # the skill call it obviously meant costs nothing and cannot do anything unsafe: loading a
        # skill body is read-only.
        if tool_call.name in (getattr(ctx.session, "skills", None) or {}):
            tool_call = replace(tool_call, name="skill", arguments={"name": tool_call.name})
            handler = _HANDLERS["skill"]
        else:
            available = ", ".join(sorted(_HANDLERS))
            return f"Unknown tool: {tool_call.name}. Available tools: {available}", False
    allowed, reason = _approve(tool_call, ctx)
    if not allowed:
        # Counted as a failure so a model that keeps re-asking trips the breaker, while a model
        # that adapts still gets another route.
        return f"This command was not run: {reason}.", False
    try:
        return handler(ctx, **tool_call.arguments), True
    except ToolError as exc:
        return str(exc), False
    except TypeError as exc:
        return f"Bad arguments for {tool_call.name}: {exc}", False
    except Exception as exc:
        logger.exception("Tool %s crashed", tool_call.name)
        return f"{tool_call.name} failed: {exc}", False


def tool_schemas(skills=None) -> list:
    # Returns the schemas to advertise for this session. The `skill` tool is offered only when
    # there is at least one skill to load: advertising a tool whose every call must fail wastes
    # context and invites a small model to call it anyway.
    if skills:
        return TOOL_SCHEMAS + [SKILL_SCHEMA]
    return TOOL_SCHEMAS

import logging

from .base import ToolContext, ToolError
from .files import read_file, write_file

logger = logging.getLogger(__name__)

__all__ = ["TOOL_SCHEMAS", "ToolContext", "ToolError", "dispatch"]

# OpenAI function schemas advertised to the model. exec is registered in Task 7 and skill in
# Task 16; both append to these two structures rather than replacing them.
TOOL_SCHEMAS = [
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
    "read_file": read_file,
    "write_file": write_file,
}


def dispatch(tool_call, ctx):
    # Runs one tool call and returns (output, ok); ok=False marks a failure for the breaker.
    # Every failure mode becomes a tool result the model can read and react to, because a crashed
    # turn teaches it nothing and a small model invents argument names constantly.
    handler = _HANDLERS.get(tool_call.name)
    if handler is None:
        available = ", ".join(sorted(_HANDLERS))
        return f"Unknown tool: {tool_call.name}. Available tools: {available}", False
    try:
        return handler(ctx, **tool_call.arguments), True
    except ToolError as exc:
        return str(exc), False
    except TypeError as exc:
        return f"Bad arguments for {tool_call.name}: {exc}", False
    except Exception as exc:
        logger.exception("Tool %s crashed", tool_call.name)
        return f"{tool_call.name} failed: {exc}", False

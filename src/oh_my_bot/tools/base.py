from dataclasses import dataclass
from typing import Any, Optional


class ToolError(Exception):
    # Raised when a tool cannot run. The message is returned to the model as the tool result, so
    # it should read as an explanation the model can act on, not as an internal error.
    pass


@dataclass
class ToolContext:
    # Everything a tool handler needs: the session (for its workspace and history), the config
    # (for limits), the approval registry, and the typing indicator to suspend while a tool
    # call is waiting on the user rather than working.
    session: Any
    config: Any
    approvals: Optional[Any] = None
    typing: Optional[Any] = None

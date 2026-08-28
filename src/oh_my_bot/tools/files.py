from pathlib import Path

from .base import ToolError

MAX_READ_CHARS = 16384


def resolve_in_workspace(workspace, path: str) -> Path:
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
    if target.is_dir():
        raise ToolError(f"That is a directory, not a file: {path}")
    if not target.is_file():
        raise ToolError(f"No such file: {path}")
    data = target.read_text(errors="replace")
    if len(data) > MAX_READ_CHARS:
        return data[:MAX_READ_CHARS] + f"\n[truncated, {len(data) - MAX_READ_CHARS} more characters]"
    return data


def write_file(ctx, path: str, content: str) -> str:
    # Writes a UTF-8 text file inside the session workspace, creating parent directories as needed.
    target = resolve_in_workspace(ctx.session.workspace(), path)
    if target.is_dir():
        raise ToolError(f"That is a directory, not a file: {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Wrote {len(content)} characters to {target}"

from ..skills import read_skill_body
from .base import ToolError


def skill_tool(ctx, name: str) -> str:
    # Loads a skill's full instructions on demand, telling the model where its scripts live so it
    # can run them through exec (which means they still pass the normal approval gate — a bundled
    # script is arbitrary code like any other).
    skills = getattr(ctx.session, "skills", None) or {}
    skill = skills.get(name)
    if skill is None:
        available = ", ".join(sorted(skills)) or "none"
        # A wrong guess becomes a tool result listing the real options, which is the cheapest
        # correction available to a small model.
        raise ToolError(f"No skill named {name!r}. Available skills: {available}")
    return f"{read_skill_body(skill)}\n\nThis skill's files are in: {skill.dir}"

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# How much of a SKILL.md is read at discovery time. Only the frontmatter is wanted at startup —
# bodies are read on demand, which is the whole point of progressive disclosure. Reading them
# eagerly here would quietly undo it.
_FRONTMATTER_READ_BYTES = 8192

_DELIMITERS = ("---", "...")


@dataclass
class Skill:
    # One discovered skill: its identity and where its files live. The body is deliberately absent.
    name: str
    description: str
    dir: Path
    body_path: Path


def parse_frontmatter(text: str):
    # Splits a leading `---` block of `key: value` lines from the body, by hand rather than adding
    # PyYAML for six lines of parsing. Unknown keys are parsed and ignored, leaving room to honour
    # something like allowed-tools later without a format change. Text with no frontmatter, or an
    # unterminated block, comes back as ({}, original text).
    lines = text.split("\n")
    if not lines or lines[0].strip() != _DELIMITERS[0]:
        return {}, text
    for index in range(1, len(lines)):
        if lines[index].strip() in _DELIMITERS:
            meta = {}
            for line in lines[1:index]:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                key, separator, value = line.partition(":")
                if not separator:
                    continue
                meta[key.strip().lower()] = value.strip().strip('"').strip("'")
            return meta, "\n".join(lines[index + 1:]).lstrip("\n")
    return {}, text


def _read_head(path: Path) -> str:
    # Reads just enough of a file to contain its frontmatter.
    with path.open("r", errors="replace") as handle:
        return handle.read(_FRONTMATTER_READ_BYTES)


def load_skills(skills_dir) -> dict:
    # Discovers every <skills_dir>/<name>/SKILL.md, reading only frontmatter. A malformed or
    # incomplete skill is skipped with a warning rather than crashing startup: one bad file should
    # not stop the bot.
    root = Path(skills_dir)
    skills = {}
    if not root.is_dir():
        logger.info("No skills directory at %s", root)
        return skills
    for path in sorted(root.glob("*/SKILL.md")):
        try:
            meta, _ = parse_frontmatter(_read_head(path))
        except OSError:
            logger.exception("Could not read skill at %s", path)
            continue
        name = (meta.get("name") or path.parent.name).strip()
        description = (meta.get("description") or "").strip()
        if not description:
            logger.warning("Skipping %s: no description in its frontmatter", path)
            continue
        if name in skills:
            logger.warning("Skipping %s: the name %r is already taken", path, name)
            continue
        skills[name] = Skill(name=name, description=description, dir=path.parent, body_path=path)
    logger.info("Loaded %s skill(s): %s", len(skills), ", ".join(sorted(skills)) or "none")
    return skills


def skill_index(skills: dict) -> str:
    # Renders the name and description of each skill for the system prompt. Never the body —
    # that is what the `skill` tool is for.
    return "\n".join(f"- {name}: {skills[name].description}" for name in sorted(skills))


def read_skill_body(skill: Skill) -> str:
    # Reads a skill's instructions on demand, without its frontmatter.
    _, body = parse_frontmatter(skill.body_path.read_text(errors="replace"))
    return body.strip()

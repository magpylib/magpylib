import importlib.metadata
import os
import re
from pathlib import Path

import pytest

import magpylib as m

SKILL_DIR = Path(m.__file__).parent / ".agents" / "skills" / "magpylib"


@pytest.mark.skipif(
    not os.environ.get("CI"),
    reason=(
        "Installed package metadata only matches the source version in a clean "
        "install (e.g. CI). In a local editable checkout the installed version "
        "can lag behind the source tree, so this check is gated to CI."
    ),
)
def test_version() -> None:
    assert importlib.metadata.version("magpylib") == m.__version__


def _frontmatter(text: str) -> dict[str, str]:
    """Read the SKILL.md YAML frontmatter without depending on a YAML parser."""
    assert text.startswith("---\n"), "SKILL.md must open with YAML frontmatter"
    block = text.split("---\n", 2)[1]
    fields: dict[str, list[str]] = {}
    key = None
    for line in block.splitlines():
        if match := re.match(r"^([a-z][\w-]*):\s*(.*)$", line):
            key = match.group(1)
            value = match.group(2)
            fields[key] = [] if value in {">-", ">", "|", "|-"} else [value]
        elif key is not None and line.startswith("  "):
            fields[key].append(line.strip())
    return {k: " ".join(v) for k, v in fields.items()}


def test_agent_skill_is_packaged() -> None:
    """The Agent Skill ships inside the package, next to the code it describes."""
    assert (SKILL_DIR / "SKILL.md").is_file()


def test_agent_skill_frontmatter() -> None:
    """Frontmatter follows the Agent Skills spec (https://agentskills.io)."""
    fields = _frontmatter((SKILL_DIR / "SKILL.md").read_text(encoding="utf-8"))

    name = fields.get("name", "")
    assert name == SKILL_DIR.name, "name must match the parent directory name"
    assert re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?", name)
    assert "--" not in name

    description = fields.get("description", "")
    assert description, "description is required"
    assert len(description) <= 1024, "description exceeds the 1024 character limit"


def test_agent_skill_links_resolve() -> None:
    """Relative links out of SKILL.md point at files that are shipped too."""
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    targets = re.findall(r"\]\((?!https?://|#)([^)]+)\)", text)
    assert targets, "expected SKILL.md to reference its files in references/"
    missing = [t for t in targets if not (SKILL_DIR / t).is_file()]
    assert not missing, f"broken relative links in SKILL.md: {missing}"

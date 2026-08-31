import importlib.metadata
import os
import re
import tarfile
import zipfile
from pathlib import Path

import pytest

import magpylib as m

SKILL_DIR = Path(m.__file__).parent / ".agents" / "skills" / "magpylib"
DIST_DIR = Path(__file__).resolve().parents[1] / "dist"
SKILL_IN_DIST = "magpylib/.agents/skills/magpylib"


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


def test_agent_skill_is_present() -> None:
    """The Agent Skill sits inside the package, next to the code it describes.

    This resolves through ``magpylib.__file__``, which is the source tree under
    an editable install. It shows the skill is in the right place, not that it
    ships -- ``test_agent_skill_ships_in_distributions`` is the check for that.
    """
    assert (SKILL_DIR / "SKILL.md").is_file()


def _distribution_members() -> dict[str, list[str]]:
    """Map each built artifact in ``dist/`` to the paths it contains."""
    members: dict[str, list[str]] = {}
    for wheel in sorted(DIST_DIR.glob("*.whl")):
        with zipfile.ZipFile(wheel) as archive:
            members[wheel.name] = archive.namelist()
    for sdist in sorted(DIST_DIR.glob("*.tar.gz")):
        with tarfile.open(sdist) as archive:
            members[sdist.name] = archive.getnames()
    return members


def test_agent_skill_ships_in_distributions() -> None:
    """The built wheel and sdist really carry the skill and its references.

    Package data is easy to lose to a build-backend or ignore-file change, and
    a source-tree assertion cannot see that happen. Build first with
    ``uvx nox -s build``; without artifacts there is nothing to inspect.
    """
    artifacts = _distribution_members()
    if not artifacts:
        pytest.skip("no artifacts in dist/ -- run `uvx nox -s build` first")

    references = sorted(p.name for p in (SKILL_DIR / "references").glob("*.md"))
    assert references, "expected reference files beside SKILL.md"
    expected = ["SKILL.md", *(f"references/{name}" for name in references)]

    for artifact, paths in artifacts.items():
        missing = [
            rel
            for rel in expected
            if not any(path.endswith(f"{SKILL_IN_DIST}/{rel}") for path in paths)
        ]
        assert not missing, f"{artifact} is missing {missing}"


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

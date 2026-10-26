import functools
import importlib.metadata
import os
import re
from pathlib import Path

import plotly.io as pio
import pytest

import magpylib as m

SKILL_DIR = Path(m.__file__).parent / ".agents" / "skills" / "magpylib"

# Objects the skill's examples use without defining them.
SKILL_EXAMPLE_NAMES = """
import numpy as np
from scipy.spatial.transform import Rotation as R
import magpylib as magpy

cube = magpy.magnet.Cuboid(
    dimension=(0.005, 0.005, 0.005), polarization=(0, 0, 1.2), position=(0, 0, 0.01)
)
source = magpy.magnet.Cuboid(
    dimension=(0.01, 0.01, 0.01), polarization=(0, 0, 1), position=(0, 0, -0.03)
)
magnet_a = magnet1 = source.copy(position=(0.02, 0, 0))
magnet_b = magnet2 = source.copy(position=(-0.02, 0, 0))
sources = [magnet_a, magnet_b]
coll = magpy.Collection(magnet_a.copy(), magnet_b.copy())
targets = [magpy.misc.Dipole(moment=(0, 0, 1), position=(0, 0, 0.05))]
sensor = magpy.Sensor(position=(0, 0, 0.03))
sensors = [sensor]
obs = (0, 0, 0.05)
observers = grid = [(0, 0, 0.05), (0.01, 0, 0.05)]
heights = np.linspace(0.02, 0.05, 5)
B = []
obj = cube.copy()
displacement = (0, 0, 0.01)
rotation = R.from_rotvec((0, 0, 10), degrees=True)
"""


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


def test_agent_skill_is_well_formed() -> None:
    """Frontmatter follows the Agent Skills spec and relative links resolve."""
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    head = text.split("---\n")[1]
    assert re.search(r"^name: magpylib$", head, re.MULTILINE)
    description = re.search(r"^description: >-\n((?:  .*\n)+)", head, re.MULTILINE)
    assert 0 < len(" ".join(description[1].split())) <= 1024
    links = re.findall(r"\]\((?!https?://|#)([^)]+)\)", text)
    assert [link for link in links if not (SKILL_DIR / link).is_file()] == []


def _skill_examples():
    examples = []
    for path in sorted(SKILL_DIR.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for block in re.finditer(r"```python\n(.*?)```", text, re.DOTALL):
            line = text[: block.start()].count("\n") + 1
            name = f"{path.relative_to(SKILL_DIR).as_posix()}:{line}"
            examples.append(pytest.param(block[1], id=name))
    return examples


@pytest.mark.parametrize("code", _skill_examples())
def test_agent_skill_examples_run(code, monkeypatch) -> None:
    """Every example in the skill runs on this version of Magpylib."""
    monkeypatch.setattr(m, "show", functools.partial(m.show, return_fig=True))
    monkeypatch.setattr(m.defaults.display, "backend", "plotly")
    monkeypatch.setattr(pio.renderers, "default", "json")
    namespace = {}
    exec(SKILL_EXAMPLE_NAMES, namespace)  # noqa: S102  # pylint: disable=exec-used
    exec(code, namespace)  # noqa: S102  # pylint: disable=exec-used

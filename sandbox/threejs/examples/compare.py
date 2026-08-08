"""Render one scene with Plotly and with the prototype, side by side.

Run with `python sandbox/threejs/compare.py`; writes `compare.html` and opens
it. The scene deliberately exercises all three `mesh3d` colour mechanisms and
all three `scatter3d` modes, since those are where the prototype has been
wrong before.
"""

from __future__ import annotations

# the backend lives one level up, beside the JS it serves
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# printing is what this script is for
# ruff: noqa: T201
import webbrowser
from pathlib import Path

from magpylib_threejs import register

import magpylib as magpy

HERE = Path(__file__).parent

_PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Plotly vs three.js</title>
<style>
  html, body {{ margin: 0; height: 100%; font: 13px sans-serif; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; height: 100%; }}
  .pane {{ display: flex; flex-direction: column; border-left: 1px solid #ddd; }}
  .pane:first-child {{ border-left: none; }}
  h2 {{ margin: 0; padding: 6px 10px; font-size: 13px; background: #f4f4f4;
       border-bottom: 1px solid #ddd; font-weight: 600; }}
  iframe {{ flex: 1; border: 0; width: 100%; }}
</style>
<div class="grid">
  <div class="pane"><h2>Plotly (reference)</h2><iframe src="{left}"></iframe></div>
  <div class="pane"><h2>three.js (prototype)</h2><iframe src="{right}"></iframe></div>
</div>
"""


def build_scene():
    """Objects covering every colour mechanism and scatter mode."""
    magnet = magpy.magnet.Cuboid(dimension=(1, 2, 3), polarization=(0, 0, 1))
    moving = magpy.magnet.Cylinder(
        dimension=(1, 1), polarization=(0, 0, 1), position=(3, 0, 0)
    )
    moving.position = [(3, 0, z) for z in range(4)]  # -> markers+lines
    return [
        magnet,  # intensity + colorscale
        moving,
        magpy.magnet.Sphere(diameter=1.5, polarization=(1, 0, 0), position=(-3, 0, 0)),
        magpy.current.Circle(diameter=5, current=1),  # scatter3d lines
        magpy.misc.Dipole(moment=(0, 0, 1), position=(0, 3, 0)),  # flat colour
        magpy.Sensor(  # facecolor
            position=(0, -3, 0), pixel=[(x, 0, 0) for x in (-0.3, 0, 0.3)]
        ),
    ]


def main():
    register()
    objects = build_scene()

    # same scene, same units, so only the rendering differs
    common = {"units_length": "m", "markers": [(4, 4, 4)], "return_fig": True}

    fig = magpy.show(*objects, backend="plotly", **common)
    (HERE / "compare_plotly.html").write_text(
        fig.to_html(full_html=True, include_plotlyjs="cdn")
    )

    (HERE / "compare_threejs.html").write_text(
        magpy.show(*objects, backend="threejs", **common)
    )

    page = HERE / "compare.html"
    page.write_text(
        _PAGE.format(left="compare_plotly.html", right="compare_threejs.html")
    )
    print(f"wrote {page}")
    webbrowser.open(f"file://{page}")


if __name__ == "__main__":
    main()

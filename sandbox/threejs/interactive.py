"""An ordinary Magpylib script that happens to pick an interactive backend.

Everything here is normal usage: build objects, call `show`. The only line that
knows a prototype is involved is the import, which is what registers the
backend -- exactly as `pip install` would for one shipped in a package.

Click an object to select it, W/E/R to move, rotate and resize it, P to aim its
polarization. Values can be typed in the panel, and the whole session undoes.
The HUD counts round-trips to Python: dragging costs none.

There is nothing else: no page, no file, no settings. The backend pins what it
needs when it is imported.
"""

from __future__ import annotations

import numpy as np
import threejs_edit  # noqa: F401  -- the import registers the backend

import magpylib as magpy


def main():
    # three levels, so the tree has something to nest: two sub-assemblies
    # inside a rig, and a couple of loose objects beside them
    array = magpy.Collection(
        *[
            magpy.magnet.Cuboid(
                dimension=(0.8, 0.8, 1.6),
                polarization=(0, 0, 1 if n % 2 == 0 else -1),
                position=(n * 1.2 - 1.8, 0, 0),
                style_label=f"pole {n + 1}",
            )
            for n in range(4)
        ],
        style_label="halbach array",
    )
    yoke = magpy.Collection(
        magpy.magnet.Cylinder(
            dimension=(1.2, 0.4),
            polarization=(0, 0, 1),
            position=(0, 0, -1.6),
            style_label="backing disc",
        ),
        magpy.magnet.Sphere(
            diameter=0.9,
            polarization=(1, 0, 0),
            position=(0, 0, -2.6),
            style_label="trim magnet",
        ),
        style_label="yoke",
    )
    rig = magpy.Collection(array, yoke, style_label="rig")

    objects = [
        rig,
        magpy.Sensor(
            position=(0, -3, 0),
            pixel=[(x, 0, 0) for x in (-0.3, 0, 0.3)],
            style_label="probe",
        ),
        magpy.misc.Dipole(moment=(0, 0, 1), position=(0, 3, 0), style_label="stray"),
        # a path, so there is something to play back
        magpy.magnet.Sphere(
            diameter=0.7,
            polarization=(0, 1, 0),
            position=np.linspace((-4, 2, 2), (4, 2, 2), 40),
            style_label="sweeper",
        ),
    ]

    magpy.show(
        *objects, backend="threejs-edit", style_magnetization_color_mode="tricycle"
    )


if __name__ == "__main__":
    main()

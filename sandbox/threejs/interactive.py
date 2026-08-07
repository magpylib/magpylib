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

import threejs_edit  # noqa: F401  -- the import registers the backend

import magpylib as magpy


def main():
    objects = [
        magpy.magnet.Cuboid(dimension=(1, 2, 3), polarization=(0, 0, 1)),
        magpy.magnet.Cylinder(
            dimension=(1, 1), polarization=(0, 0, 1), position=(3, 0, 0)
        ),
        magpy.magnet.Sphere(diameter=1.5, polarization=(1, 0, 0), position=(-3, 0, 0)),
        magpy.misc.Dipole(moment=(0, 0, 1), position=(0, 3, 0)),
        magpy.Sensor(position=(0, -3, 0), pixel=[(x, 0, 0) for x in (-0.3, 0, 0.3)]),
    ]

    magpy.show(*objects, backend="threejs-edit")


if __name__ == "__main__":
    main()

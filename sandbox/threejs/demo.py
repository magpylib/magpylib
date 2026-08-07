"""Drive the prototype three.js backend and report what the API does.

Run with `python sandbox/threejs/demo.py`; writes `cuboid.html` next to it.
Every warning Magpylib raises is printed rather than swallowed -- they are
the interesting output.
"""

from __future__ import annotations

# printing is what this script is for
# ruff: noqa: T201
import warnings
import webbrowser
from pathlib import Path

from magpylib_threejs import register

import magpylib as magpy

HERE = Path(__file__).parent


def report(label, func):
    """Run `func`, printing every warning and exception it produces."""
    print(f"\n--- {label} ---")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = func()
        except Exception as exc:  # noqa: BLE001
            print(f"  RAISED {type(exc).__name__}: {exc}")
            result = None
        for warning in caught:
            print(f"  warns: {warning.message}")
    return result


def main():
    register()

    cube = magpy.magnet.Cuboid(dimension=(1, 2, 3), polarization=(0, 0, 1))

    html = report(
        "one cuboid",
        lambda: magpy.show(cube, backend="threejs", return_fig=True),
    )
    if html:
        (HERE / "cuboid.html").write_text(html)
        print(f"  wrote cuboid.html ({len(html)} bytes)")

    # a second, differently coloured object -- checks per-object identity
    other = magpy.magnet.Cuboid(
        dimension=(1, 1, 1), polarization=(1, 0, 0), position=(3, 0, 0)
    )
    html = report(
        "two cuboids",
        lambda: magpy.show(cube, other, backend="threejs", return_fig=True),
    )
    if html:
        (HERE / "two_cuboids.html").write_text(html)
        print(f"  wrote two_cuboids.html ({len(html)} bytes)")

    # things the backend has NOT declared support for
    with_path = magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))
    with_path.position = [(0, 0, 0), (2, 0, 0), (4, 0, 0)]
    report(
        "object with a path (scatter3d: markers+lines)",
        lambda: magpy.show(with_path, backend="threejs", return_fig=True),
    )
    report(
        "animation=True (undeclared capability)",
        lambda: magpy.show(
            with_path, backend="threejs", animation=True, return_fig=True
        ),
    )
    # a sensor is autosized but still mesh3d-only, so it draws without a warning
    report(
        "a sensor (autosized, but mesh3d)",
        lambda: magpy.show(
            cube, magpy.Sensor(position=(0, 0, 4)), backend="threejs", return_fig=True
        ),
    )
    html = report(
        "a current (scatter3d: lines)",
        lambda: magpy.show(
            magpy.current.Circle(diameter=2, current=1),
            backend="threejs",
            return_fig=True,
        ),
    )
    if html:
        (HERE / "current.html").write_text(html)
        print(f"  wrote current.html ({len(html)} bytes)")

    # all three scatter3d modes at once: 'lines' from a current, 'markers+lines'
    # from a path, and 'markers' from the markers= kwarg
    moving = magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))
    moving.position = [(0, 0, z) for z in range(4)]
    html = report(
        "every object type at once",
        lambda: magpy.show(
            cube,
            moving,
            magpy.current.Circle(diameter=4, current=1),
            magpy.current.Polyline(vertices=[(0, 0, 0), (1, 1, 1)], current=1),
            magpy.misc.Dipole(moment=(0, 0, 1), position=(0, 3, 0)),
            magpy.Sensor(position=(0, -3, 0), pixel=[(x, 0, 0) for x in (0, 0.3)]),
            markers=[(4, 4, 4)],
            backend="threejs",
            return_fig=True,
        ),
    )
    if html:
        (HERE / "everything.html").write_text(html)
        print(f"  wrote everything.html ({len(html)} bytes)")
    report(
        "subplots (undeclared capability)",
        lambda: magpy.show(
            {"objects": [cube], "row": 1, "col": 1},
            {"objects": [other], "row": 1, "col": 2},
            backend="threejs",
            return_fig=True,
        ),
    )
    report(
        "an option the backend does not accept",
        lambda: magpy.show(
            cube, backend="threejs", threejs_wireframe=True, return_fig=True
        ),
    )
    # magpylib discards the returned figure unless return_fig is set; what the
    # backend still owns is whether to *display*, which is why it reads the flag.
    # The browser is stubbed out so running the demo stays side-effect free.
    opened = []
    webbrowser.open = opened.append
    result = report(
        "return_fig=False (magpylib discards the figure; backend displays)",
        lambda: magpy.show(cube, backend="threejs"),
    )
    print(
        f"  magpy.show returned: {type(result).__name__}; opened {len(opened)} tab(s)"
    )


if __name__ == "__main__":
    main()

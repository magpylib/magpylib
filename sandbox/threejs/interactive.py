"""A **script**: builds a scene, renders it with the editor layer, opens it.

The editor front-end itself is `editor.js`; the host-side policy it needs --
which objects can be resized by scaling, how to read a polarization -- is
below. Run it, do not import it.

Pick-and-drag on top of the prototype backend, to test the editing loop.

The question this answers: can a drag run without a Python round-trip? A
round-trip per frame would cap interaction at whatever `show()` costs, which is
seconds on a large scene -- unusable. The answer is yes, on two conditions,
and both are things the probe already found:

1. Geometry must be **scene-independent**, or moving one object invalidates
   every other object's vertices and the JS scene cannot be kept. That means
   `units_length` pinned and `sizemode="absolute"` (finding 1).
2. The edit must be a **transform**. Moving and rotating are matrix updates the
   GPU already does. Changing a dimension is not -- that needs new geometry,
   so it needs Python, but it is not a per-frame operation.

So the loop is: drag entirely in JS at frame rate, and tell Python **once**, on
drop. The HUD counts round-trips to make the point checkable rather than
asserted.

Run with `python sandbox/threejs/interactive.py`.
"""

from __future__ import annotations

# printing is what this script is for
# ruff: noqa: T201
import json
import webbrowser
from pathlib import Path

from magpylib_threejs import register, render_page

import magpylib as magpy

HERE = Path(__file__).parent

#: The editor front-end, kept beside this file as real JavaScript.
_INTERACTION_JS = (HERE / "editor.js").read_text()

# `import` statements must sit at the top of a module script, so the addon
# import is prepended to the page rather than appended with the rest.
_IMPORT = """
import { TransformControls } from 'three/addons/controls/TransformControls.js';
"""


#: Classes whose mesh is *exactly* the base mesh scaled, so a resize can be
#: previewed with `mesh.scale` and magpylib told only the final value. Every
#: entry below was verified by rendering twice and comparing vertices.
#:
#: `constraint` records which scale axes are independent, which follows from
#: how many numbers the parameter has: a Cylinder's `dimension` is
#: (diameter, height), so x and y are locked together and z is free.
#:
#: Excluded, and why:
#:  * `CylinderSegment` -- its angles do not scale with its radii.
#:  * `Sensor` -- a composite. Its pixels sit at real coordinates while the
#:    cross is styled, so `style.size` scales only part of the mesh; scaling
#:    the whole thing would displace the pixels.
#:  * meshes (`Tetrahedron`, `TriangularMesh`) -- vertices are the parameter,
#:    so there is no single scale to drag.
SCALE_COVARIANT = {
    "Cuboid": ("dimension", "free"),
    "Sphere": ("diameter", "uniform"),
    "Cylinder": ("dimension", "xy"),
    # A Dipole has no physical size at all -- its arrow is styled geometry, and
    # `style.size` is one scalar, so the resize is uniform by construction and
    # there is no aspect ratio to preserve. Only holds with sizemode="absolute";
    # under "scaled" the value is a multiplier on the scene-derived autosize.
    "Dipole": ("style.size", "uniform"),
}


def _resolve(obj, path):
    """Read a dotted attribute path, falling back to the library default.

    An unset style property reads as `None` on the object and only takes its
    value from `magpy.defaults.display.style.<family>` when the figure is
    drawn. `merged()` does not help: it resolves set-vs-inherited *within* the
    object's own tree, so `Dipole().style.merged().size` is still `None` while
    the effective size is 1. A host reading style has to consult the default
    itself.
    """
    value = obj
    for part in path.split("."):
        value = getattr(value, part)
    if value is None and path.startswith("style."):
        node = magpy.defaults.display.style
        for part in (type(obj).__name__.lower(), *path.split(".")[1:]):
            node = getattr(node, part)
        value = node
    return value


def shape_of(obj):
    """The scale-covariant shape parameter of `obj`, or None."""
    entry = SCALE_COVARIANT.get(type(obj).__name__)
    if entry is None:
        return None
    attr, constraint = entry
    value = _resolve(obj, attr)
    return {
        "kind": type(obj).__name__,
        "attr": attr,
        "value": value.tolist() if hasattr(value, "tolist") else float(value),
        "constraint": constraint,
    }


def polarization_of(obj):
    """The polarization vector of `obj`, or None if it has none.

    Reported in the object's **local** frame, which is how magpylib stores it.
    The rendered `intensity` is the vertex projected on the *world* vector, so
    the two differ by the object's orientation -- reading the attribute as if
    it were world-space is wrong for anything rotated (measured: 0.44 off for a
    50 degree rotation). That is one reason this edit is not previewed in the
    browser; see `_MAGNETIZATION_NOTE`.
    """
    pol = getattr(obj, "polarization", None)
    if pol is None:
        return None
    return {
        "attr": "polarization",
        "value": [float(component) for component in pol],
    }


def main():
    # The host owns the objects, so it can resolve object_id back to the
    # object and read the transform Magpylib does not put in the payload.
    # `object_id` is documented as valid for exactly this: "an interactive
    # viewer holding the same objects". Nothing has to change in Magpylib.
    registry: dict[int, object] = {}

    def show(scene):
        anchors = {oid: obj.position for oid, obj in registry.items()}
        return render_page(scene, extra_js=_INTERACTION_JS, anchors=anchors)

    register("threejs-edit", show)

    # Pin both scene-dependent scalings, or dragging one object would change
    # the geometry of the others and the JS scene could not be kept.
    magpy.defaults.display.style.sensor.sizemode = "absolute"
    magpy.defaults.display.style.dipole.sizemode = "absolute"

    objects = [
        magpy.magnet.Cuboid(dimension=(1, 2, 3), polarization=(0, 0, 1)),
        magpy.magnet.Cylinder(
            dimension=(1, 1), polarization=(0, 0, 1), position=(3, 0, 0)
        ),
        magpy.magnet.Sphere(diameter=1.5, polarization=(1, 0, 0), position=(-3, 0, 0)),
        magpy.misc.Dipole(moment=(0, 0, 1), position=(0, 3, 0)),
        magpy.Sensor(position=(0, -3, 0), pixel=[(x, 0, 0) for x in (-0.3, 0, 0.3)]),
    ]

    # units_length="m" matches magpylib's own unit, so positions need no
    # conversion. Any other unit and the host must scale the anchors to match.
    registry.update({id(obj): obj for obj in objects})

    html = magpy.show(
        *objects, backend="threejs-edit", units_length="m", return_fig=True
    )
    shapes = {
        str(oid): s for oid, obj in registry.items() if (s := shape_of(obj)) is not None
    }
    polarizations = {
        str(oid): p
        for oid, obj in registry.items()
        if (p := polarization_of(obj)) is not None
    }
    html = html.replace(
        # the addon import has to precede the module body
        "import { OrbitControls }",
        _IMPORT.strip() + "\nimport { OrbitControls }",
    ).replace(
        "const DATA =",
        f"const UNIT = 'm';\nconst SHAPES = {json.dumps(shapes)};\n"
        f"const POLARIZATION = {json.dumps(polarizations)};\nconst DATA =",
    )

    page = HERE / "interactive.html"
    page.write_text(html)
    print(f"wrote {page}")
    webbrowser.open(f"file://{page}")


if __name__ == "__main__":
    main()

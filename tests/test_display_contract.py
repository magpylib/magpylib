"""The `Scene` a display backend receives, pinned for `API_VERSION`.

A fixed scene -- every object class, a nested collection, a path, a user
model of each kind, markers, a 3D and a 2D panel, two animation frames -- is
captured under each combination of the two declarations that change what a
backend receives, and reduced to its structure: the envelope's fields, and
for every kind of trace its keys and the type of each value, arrays by rank.
Geometry is left out, so a finer mesh passes; a key added, renamed, removed
or retyped does not.

A failure here is an `API_VERSION` decision, not a test to update in passing:
an addition only needs the golden file regenerated,

    MAGPYLIB_UPDATE_CONTRACT=1 pytest tests/test_display_contract.py

while a removal, a rename or a change of meaning also needs the version
bumped and a new golden file named after it.
"""

import dataclasses
import json
import numbers
import os
from pathlib import Path

import numpy as np
import pytest

import magpylib as magpy
from magpylib._src.display.api import DisplayBackend
from magpylib.graphics.backend import API_VERSION

GOLDEN = Path(__file__).parent / "testdata" / f"scene_contract_v{API_VERSION}.json"


def describe(value):
    """The type of a payload value, without its size: `ndarray[f8,1d]`."""
    if value is None:
        return "None"
    if isinstance(value, bool | np.bool_):
        return "bool"
    if isinstance(value, numbers.Integral):
        return "int"
    if isinstance(value, numbers.Real):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, np.ndarray):
        return f"ndarray[{value.dtype.kind},{value.ndim}d]"
    if isinstance(value, list | tuple):
        inner = sorted({describe(v) for v in value})
        return f"{type(value).__name__}[{'|'.join(inner)}]"
    if isinstance(value, dict):
        return (
            "dict{"
            + ",".join(f"{k}:{describe(v)}" for k, v in sorted(value.items()))
            + "}"
        )
    return type(value).__name__


def signatures(traces):
    """Each distinct trace layout once: its type, and its other keys' value
    types."""
    found = {
        json.dumps(
            {k: (v if k == "type" else describe(v)) for k, v in sorted(t.items())},
            sort_keys=True,
        )
        for t in traces
    }
    return [json.loads(s) for s in sorted(found)]


def structure(scene):
    """What the contract promises about `scene`, its geometry left out."""
    # what lines a trace up from frame to frame, its legend entry included:
    # a backend follows a trace through an animation by its position
    layouts = [
        [
            (t["type"], t.get("object_id"), t.get("name"), t.get("showlegend"))
            for t in f.traces
        ]
        for f in scene.frames
    ]
    return {
        "scene": {
            f.name: describe(getattr(scene, f.name))
            for f in dataclasses.fields(scene)
            if f.name not in ("panels", "frames", "animation")
        },
        "animation": {
            f.name: describe(getattr(scene.animation, f.name))
            for f in dataclasses.fields(scene.animation)
        },
        "panels": [
            {
                "cell": [p.row, p.col],
                "kind": p.kind,
                "ranges": describe(p.ranges),
                "labels": sorted(p.labels),  # which axes, not their wording
                "objects": [type(o).__name__ for o in p.objects],
            }
            for p in scene.panels
        ],
        "frames": {
            "count": len(scene.frames),
            "label": describe(scene.frames[0].label),
            "same traces in every frame": all(lay == layouts[0] for lay in layouts),
        },
        "traces": signatures(t for f in scene.frames for t in f.traces),
        "native_traces": signatures(t for f in scene.frames for t in f.native_traces),
    }


def capture(gradient, merge):
    """The scene a backend declaring `gradient` and `merge` receives."""
    scenes = []
    name = "contract_probe"
    magpy.register_backend(
        name,
        scenes.append,
        api_version=API_VERSION,
        supports_animation=True,
        supports_subplots=True,
        supports_colorgradient=gradient,
        supports_native_traces=True,
        merge_traces=merge,
    )
    try:
        triangle = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        tetra = [*triangle, (0, 0, 1)]
        magnets = [
            magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1)),
            magpy.magnet.Cylinder(polarization=(0, 0, 1), dimension=(1, 1)),
            magpy.magnet.CylinderSegment(
                polarization=(0, 0, 1), dimension=(0.5, 1, 1, 0, 90)
            ),
            magpy.magnet.Sphere(polarization=(0, 0, 1), diameter=1),
            magpy.magnet.Tetrahedron(polarization=(0, 0, 1), vertices=tetra),
            magpy.magnet.TriangularMesh(
                polarization=(0, 0, 1),
                vertices=tetra,
                faces=[(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)],
            ),
        ]
        others = [
            magpy.current.Circle(current=1, diameter=1),
            magpy.current.Polyline(current=1, vertices=triangle),
            magpy.current.TriangleStrip(current=1, vertices=[*triangle, (1, 1, 0)]),
            magpy.current.TriangleSheet(
                current_densities=[(1, 0, 0)], vertices=triangle, faces=[(0, 1, 2)]
            ),
            magpy.misc.Dipole(moment=(0, 0, 1)),
            magpy.misc.Triangle(polarization=(0, 0, 1), vertices=triangle),
            # a field, so that the sensor showing its pixel field can compute
            magpy.misc.CustomSource(
                field_func=lambda field, observers: np.zeros_like(observers, float)  # noqa: ARG005
            ),
        ]
        for i, obj in enumerate([*magnets, *others]):
            obj.position = (2 * i, 0, 0)
        moving = magnets[0]
        moving.move([(0, 0, 0), (0, 0, 1)], start=0)
        moving.style.model3d.add_trace(
            backend="generic",
            constructor="Mesh3d",
            kwargs={
                "x": [0, 1, 0],
                "y": [0, 0, 1],
                "z": [0, 0, 0],
                "i": [0],
                "j": [1],
                "k": [2],
            },
        )
        moving.style.model3d.add_trace(
            backend=name, constructor="Thing", kwargs={"x": [0], "y": [0], "z": [0]}
        )
        probe = magpy.Sensor(
            position=(0, 2, 2),
            pixel=[(0, 0, 0), (0.1, 0, 0)],
            style_pixel_field_source="Bz",
        )
        bare = magpy.Sensor(position=(2, 2, 2))
        group = magpy.Collection(
            magpy.Collection(*magnets[1:3]), magnets[3], style_label="group"
        )
        shown = [moving, group, *magnets[4:], *others, probe, bare]
        magpy.show(
            {"objects": shown, "col": 1},
            {"objects": [moving, probe], "col": 2, "output": "Bz"},
            backend=name,
            animation=True,
            markers=[(0, 0, 3)],
        )
    finally:
        DisplayBackend.backends.pop(name, None)
    (scene,) = scenes
    return scene


VARIANTS = {
    "gradient, merged": (True, True),
    "gradient, unmerged": (True, False),
    "sliced, merged": (False, True),
    "sliced, unmerged": (False, False),
}


def test_the_scene_a_backend_receives_is_pinned():
    """Against `tests/testdata/scene_contract_v<API_VERSION>.json`; see the
    module docstring for what a failure means."""
    found = {variant: structure(capture(*flags)) for variant, flags in VARIANTS.items()}
    found = json.loads(json.dumps(found))  # tuples as lists, as the file has them
    if os.environ.get("MAGPYLIB_UPDATE_CONTRACT"):
        GOLDEN.write_text(json.dumps(found, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"regenerated {GOLDEN.name}")
    expected = json.loads(GOLDEN.read_text())
    for variant in VARIANTS:
        assert found[variant] == expected[variant], variant

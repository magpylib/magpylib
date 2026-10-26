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
from magpylib.graphics.backend import ALLOWED_LINESTYLES, ALLOWED_SYMBOLS, API_VERSION

#: What `line_dash` takes: the long names, one per style
LINE_DASHES = ALLOWED_LINESTYLES[:6]

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


def test_the_scene_names_symbols_and_dashes_in_the_dialect():
    """Magpylib's names, which every backend translates: CustomSource used to
    send plotly's 'diamond'."""
    traces = [t for f in capture(True, False).frames for t in f.traces]
    symbols = {t["marker_symbol"] for t in traces if t.get("marker_symbol")}
    dashes = {t["line_dash"] for t in traces if t.get("line_dash")}
    assert symbols <= set(ALLOWED_SYMBOLS)
    assert dashes <= set(LINE_DASHES)


@pytest.mark.parametrize(
    ("style", "long_name"),
    list(
        zip(
            ALLOWED_LINESTYLES,
            [*LINE_DASHES, *LINE_DASHES[:2], "dashdot", *["dotted"] * 3],
            strict=True,
        )
    ),
)
def test_a_line_style_reaches_the_backend_under_its_long_name(style, long_name):
    """The style accepts twelve spellings, a dash tuple among them; a backend
    used to receive whichever the user chose, and Matplotlib and Pyvista
    failed on some of their own."""
    scenes = []
    magpy.register_backend("dash_probe", scenes.append)
    coil = magpy.current.Circle(current=1, diameter=1, style_line_style=style)
    coil.move([(0, 0, 0), (0, 0, 1)], start=0)
    coil.style.path.line.style = style
    magpy.show(coil, backend="dash_probe")
    dashes = {t["line_dash"] for t in scenes[0].frames[0].traces if t.get("line_dash")}
    assert long_name in dashes
    assert dashes <= set(LINE_DASHES)


@pytest.mark.parametrize("backend", ["matplotlib", "plotly", "pyvista"])
def test_every_backend_draws_the_loose_line_styles(backend):
    """Matplotlib rejected 'loosely dotted' and 'loosely dashdotted' as names,
    and the dash tuple broke it and Pyvista inside an array conversion."""
    coil = magpy.current.Circle(
        current=1, diameter=1, style_line_style="loosely dotted"
    )
    coil.move([(0, 0, 0), (0, 0, 1)], start=0)
    coil.style.path.line.style = (0, (1, 1))
    cube = magpy.magnet.Cuboid(
        polarization=(0, 0, 1),
        dimension=(1, 1, 1),
        style_path_line_style="loosely dashdotted",
    )
    cube.move([(2, 0, 0), (2, 0, 1)], start=0)
    extra = {"fig_off_screen": True} if backend == "pyvista" else {}
    assert magpy.show(coil, cube, backend=backend, return_fig=True, **extra) is not None


@pytest.mark.parametrize("symbol", ["arrow", "cone"])
def test_pixels_reading_nothing_keep_the_frames_in_line(symbol):
    """The first frame's pixels read nothing, the second's do. Pixels all
    reading nothing, null pixels hidden, used to raise on every backend, and
    arrows sent the pixels' 'cube' as a marker symbol, which Matplotlib
    rejected."""
    scenes = []
    magpy.register_backend("pixels_probe", scenes.append, supports_animation=True)
    sensor = magpy.Sensor(
        position=[(0, 0, 0), (1, 0, 0)],
        pixel=[(0, 0, 1), (0, 0, 2)],  # first on the dipole's axis: no Bxy
        style_pixel_field_source="Bxy",
        style_pixel_field_symbol=symbol,
        style_pixel_field_shownull=False,
    )
    dipole = magpy.misc.Dipole(moment=(0, 0, 1))
    magpy.show(dipole, sensor, backend="pixels_probe", animation=True)
    first, second = scenes[0].frames
    assert [t["type"] for t in first.traces] == [t["type"] for t in second.traces]
    symbols = {
        s
        for t in (*first.traces, *second.traces)
        if t.get("marker_symbol") is not None
        for s in np.atleast_1d(t["marker_symbol"])
    }
    assert symbols <= set(ALLOWED_SYMBOLS)


def drawn_scene(*objects, **kwargs):
    """The scene a backend with color gradients receives for `objects`."""
    scenes = []
    magpy.register_backend("trace_probe", scenes.append, supports_colorgradient=True)
    try:
        magpy.show(*objects, backend="trace_probe", units_length="m", **kwargs)
    finally:
        DisplayBackend.backends.pop("trace_probe", None)
    return scenes[0]


TETRA = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=float)
TETRA_FACES = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]


def test_drawing_writes_nothing_to_the_objects():
    """Drawing a TriangleStrip set `faces` and `mesh` on it, and a
    TriangleSheet `mesh`."""
    strip = magpy.current.TriangleStrip(
        current=1,
        vertices=[(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)],
        style_mesh_grid_show=True,
    )
    sheet = magpy.current.TriangleSheet(
        vertices=TETRA[:3], faces=[(0, 1, 2)], current_densities=[(1, 0, 0)]
    )
    before = [set(vars(obj)) for obj in (strip, sheet) if obj.style]  # style is lazy
    drawn_scene(strip, sheet)
    assert [set(vars(obj)) for obj in (strip, sheet)] == before


def test_each_copy_along_a_path_is_drawn_at_its_own_step():
    """A TriangularMesh drew its last vertices at every path step; every copy
    took the last step's magnetization; and a Triangle's orientation markers
    left the copies after them uncolored."""
    mesh = magpy.magnet.TriangularMesh(
        vertices=[TETRA, 3 * TETRA],
        faces=TETRA_FACES,
        polarization=(0, 0, 1),
        style_path_frames=[0],
    )
    (trace,) = [t for t in drawn_scene(mesh).frames[0].traces if t["type"] == "mesh3d"]
    assert np.max(trace["x"]) == 1

    cube = magpy.magnet.Cuboid(
        dimension=(1, 1, 1),
        polarization=[(0, 0, 1), (1, 0, 0)],
        position=[(0, 0, 0), (3, 0, 0)],
        style_path_frames=[0, 1],
    )
    (trace,) = [t for t in drawn_scene(cube).frames[0].traces if t["type"] == "mesh3d"]
    x, z, intensity = (np.asarray(trace[k]) for k in ("x", "z", "intensity"))
    assert np.corrcoef(z[x < 2], intensity[x < 2])[0, 1] > 0.99
    assert np.corrcoef(x[x > 2], intensity[x > 2])[0, 1] > 0.99

    triangle = magpy.misc.Triangle(
        vertices=[TETRA[:3], 2 * TETRA[:3]],
        polarization=(0, 0, 1),
        position=[(0, 0, 0), (5, 0, 0)],
        style_path_frames=[0, 1],
    )
    traces = drawn_scene(triangle).frames[0].traces
    colored = [t for t in traces if t.get("intensity") is not None]
    x = np.concatenate([t["x"] for t in colored])
    assert x.min() < 2
    assert x.max() > 4


def test_disconnected_mesh_parts_keep_their_colors():
    """Each part is drawn in its own color, the magnetization as one arrow;
    6.0 colored the parts by magnetization again, so they looked alike."""
    mesh = magpy.magnet.TriangularMesh(
        vertices=np.vstack([TETRA, TETRA + np.array((3, 0, 0))]),
        faces=TETRA_FACES + [tuple(i + 4 for i in f) for f in TETRA_FACES],
        polarization=(0, 0, 1),
        check_disconnected="ignore",
        style_mesh_disconnected_show=True,
    )
    traces = drawn_scene(mesh).frames[0].traces
    parts = [t for t in traces if t["type"] == "mesh3d"]
    assert [t["color"] for t in parts] == ["red", "blue"]
    assert all(t.get("intensity") is None for t in parts)
    assert sum(t.get("mode") == "lines" for t in traces) == 1

"""Tests for the public display data model and the generic->Scene adapter."""

import numpy as np
import pytest

import magpylib as magpy
from magpylib._src.display.api import (
    AnimationSettings,
    DisplayBackend,
    Frame,
    Mesh,
    Panel,
    Scatter2D,
    Scatter3D,
    Scene,
)
from magpylib._src.display.backend_registry import RegisteredBackend
from magpylib._src.display.scene_builder import scene_from_generic


@pytest.fixture
def scene_backend():
    """Register a backend that converts the payload and hands back the Scene."""
    name = "scene_probe"
    captured = {}

    def show_func(data, **kwargs):
        captured["raw"] = data
        captured["scene"] = scene_from_generic(
            data,
            subplot_specs=kwargs.get("subplot_specs"),
            canvas=kwargs.get("canvas"),
            canvas_update=kwargs.get("canvas_update", True),
            fig_kwargs=kwargs.get("fig_kwargs"),
            show_kwargs=kwargs.get("show_kwargs"),
        )
        return data

    RegisteredBackend(
        name=name,
        show_func=show_func,
        supports_animation=True,
        supports_subplots=True,
        supports_colorgradient=True,
        supports_animation_output=True,
    )
    try:
        yield name, captured
    finally:
        RegisteredBackend.backends.pop(name, None)


def make_source():
    return magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))


# --- Scene helpers ---------------------------------------------------------


def test_scene_grid_helpers():
    """Grid size and panel lookup derive from the panels themselves."""
    scene = Scene(
        panels=(Panel(row=1, col=1), Panel(row=2, col=1, kind="chart2d")),
        frames=(Frame(label="0"),),
    )
    assert (scene.n_rows, scene.n_cols) == (2, 1)
    assert scene.has_subplots is True
    assert scene.is_animation is False
    assert scene.panel(2, 1).kind == "chart2d"
    assert scene.panel(9, 9) is None


def test_empty_scene_reports_a_single_cell():
    """An empty grid must not report zero rows/cols."""
    scene = Scene()
    assert (scene.n_rows, scene.n_cols) == (1, 1)
    assert scene.has_subplots is False


def test_display_backend_defaults_are_conservative():
    """Capabilities default off so a new one never changes existing backends."""
    assert DisplayBackend.supports_animation is False
    assert DisplayBackend.supports_subplots is False
    assert DisplayBackend.supports_colorgradient is False
    assert DisplayBackend.supports_animation_output is False
    # a preference, not a capability -- and the plotly-friendly default
    assert DisplayBackend.merge_traces is True
    with pytest.raises(NotImplementedError):
        DisplayBackend().show(Scene())


# --- adapter ---------------------------------------------------------------


def test_mesh_geometry_round_trips(scene_backend):
    """vertices/faces must carry exactly the x/y/z and i/j/k data."""
    name, cap = scene_backend
    magpy.show(make_source(), backend=name)

    raw = next(t for t in cap["raw"]["frames"][0]["data"] if t.get("type") == "mesh3d")
    mesh = next(t for t in cap["scene"].frames[0].traces if isinstance(t, Mesh))

    assert mesh.vertices.shape == (len(raw["x"]), 3)
    assert mesh.faces.shape == (len(raw["i"]), 3)
    np.testing.assert_array_equal(mesh.vertices[:, 0], raw["x"])
    np.testing.assert_array_equal(mesh.vertices[:, 2], raw["z"])
    np.testing.assert_array_equal(mesh.faces[:, 1], raw["j"])


def test_trace_common_fields_are_translated(scene_backend):
    """The plotly-flavoured names are renamed, not dropped."""
    name, cap = scene_backend
    magpy.show(make_source(), backend=name)

    raw = next(t for t in cap["raw"]["frames"][0]["data"] if t.get("type") == "mesh3d")
    mesh = next(t for t in cap["scene"].frames[0].traces if isinstance(t, Mesh))

    assert mesh.label == raw["name"]
    assert mesh.group == raw["legendgroup"]
    assert mesh.show_in_legend == bool(raw["showlegend"])
    assert mesh.opacity == raw["opacity"]
    assert (mesh.colormap is None) == (raw.get("colorscale") is None)


def test_scatter3d_draw_is_a_set_not_a_class(scene_backend):
    """One trace may draw lines, markers and text at once."""
    name, cap = scene_backend
    src = make_source()
    src.position = [(0, 0, 0), (0, 0, 1)]
    src.style.path.numbering = True
    magpy.show(src, backend=name)

    scatters = [t for t in cap["scene"].frames[0].traces if isinstance(t, Scatter3D)]
    assert scatters, "expected a path trace"
    combined = next(t for t in scatters if len(t.draw) > 1)
    assert {"lines", "markers", "text"} == combined.draw
    assert combined.points.shape[1] == 3


def test_scatter2d_has_no_opacity(scene_backend):
    """2D traces never carry an opacity, so the model must not invent one."""
    name, cap = scene_backend
    src, sensor = make_source(), magpy.Sensor(position=(3, 0, 0))
    magpy.show(
        {"objects": [src, sensor], "row": 1, "col": 1},
        {"objects": [src, sensor], "row": 1, "col": 2, "output": "Bx"},
        backend=name,
    )

    flat = [t for f in cap["scene"].frames for t in f.traces]
    curve = next(t for t in flat if isinstance(t, Scatter2D))
    assert not hasattr(curve, "opacity")
    assert curve.group_title is not None
    assert curve.x.shape == curve.y.shape


def test_panels_carry_kind_and_drop_3d_metadata_for_charts(scene_backend):
    """chart2d panels get no ranges/labels: magpylib's are 3D and meaningless."""
    name, cap = scene_backend
    src, sensor = make_source(), magpy.Sensor(position=(3, 0, 0))
    magpy.show(
        {"objects": [src, sensor], "row": 1, "col": 1},
        {"objects": [src, sensor], "row": 1, "col": 2, "output": "Bx"},
        backend=name,
    )
    scene = cap["scene"]

    p3d = scene.panel(1, 1)
    p2d = scene.panel(1, 2)
    assert p3d.kind == "scene3d"
    assert p3d.ranges.shape == (3, 2)
    assert set(p3d.labels) == {"x", "y", "z"}

    assert p2d.kind == "chart2d"
    assert p2d.ranges is None
    assert p2d.labels == {}


def test_frame_title_is_per_frame(scene_backend):
    """Title lives on the frame and varies across an animation."""
    name, cap = scene_backend
    src = make_source()
    src.position = [(0, 0, 0), (0, 0, 1), (0, 0, 2)]
    magpy.show(src, backend=name, animation=True)

    scene = cap["scene"]
    assert scene.is_animation
    titles = [f.title for f in scene.frames]
    assert all(t is not None for t in titles)
    assert len(set(titles)) == len(titles), "each frame gets its own title"


def test_animation_settings_replace_input_kwargs(scene_backend):
    """The `input_kwargs` bag becomes a typed object."""
    name, cap = scene_backend
    magpy.show(make_source(), backend=name)

    anim = cap["scene"].animation
    raw = cap["raw"]["input_kwargs"]
    assert isinstance(anim, AnimationSettings)
    assert anim.fps == raw["animation_fps"]
    assert anim.max_frames == raw["animation_maxframes"]
    assert anim.output == raw["animation_output"]


def test_native_traces_keep_placement_information(scene_backend):
    """coordsargs and extra are what a backend needs to place the model."""
    name, cap = scene_backend
    src = make_source()
    src.style.model3d.add_trace(
        backend=name,
        constructor="Line",
        kwargs={"x": [0, 1], "y": [0, 1], "z": [0, 1]},
    )
    magpy.show(src, backend=name)

    native = cap["scene"].frames[0].native_traces
    assert len(native) == 1
    model = native[0]
    assert model.constructor == "Line"
    assert set(model.coordsargs) == {"x", "y", "z"}
    assert "color" in model.extra
    assert not hasattr(model, "backend")


def test_unknown_trace_type_is_rejected():
    """A trace the model cannot express must fail loudly, not silently drop."""
    data = {"frames": [{"name": "0", "data": [{"type": "surface"}], "layout": {}}]}
    with pytest.raises(ValueError, match="surface"):
        scene_from_generic(data)


def test_marker_size_may_be_per_point(scene_backend):
    """A multi-pixel sensor's field curve carries one marker size per point.

    Regression guard: an array-valued `size` makes dataclass `==` ambiguous,
    which is why the adapter tests key presence rather than comparing styles.
    """
    name, cap = scene_backend
    src = make_source()
    sensor = magpy.Sensor(position=(3, 0, 0), pixel=[(0, 0, 0), (0, 0, 0.5)])
    magpy.show(
        {"objects": [src, sensor], "row": 1, "col": 1},
        {"objects": [src, sensor], "row": 1, "col": 2, "output": "Bx"},
        backend=name,
    )

    flat = [t for f in cap["scene"].frames for t in f.traces]
    sizes = [
        t.marker.size for t in flat if isinstance(t, Scatter2D) and t.marker is not None
    ]
    assert sizes, "expected marker styling on the field curve"
    assert any(isinstance(s, np.ndarray) for s in sizes), (
        f"expected an array-valued marker size, got {sizes}"
    )

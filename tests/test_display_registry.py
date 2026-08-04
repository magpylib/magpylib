"""Tests for the display-backend registry."""

import warnings

import pytest

import magpylib as magpy
from magpylib._src.display.backend_registry import RegisteredBackend
from magpylib._src.exceptions import MagpylibBadUserInput


def make_source():
    """A minimal displayable source."""
    return magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))


@pytest.fixture
def noop_backend():
    """Register a backend returning the generic data untouched.

    Yields ``(name, calls)``, where `calls` collects the kwargs each `show`
    dispatch handed to the backend.
    """
    name = "noop"
    calls = []

    def show_func(data, **kwargs):
        calls.append(kwargs)
        return data

    RegisteredBackend(
        name=name,
        show_func=show_func,
        supports_animation=False,
        supports_subplots=False,
        supports_colorgradient=False,
        supports_animation_output=False,
    )
    try:
        yield name, calls
    finally:
        RegisteredBackend.backends.pop(name, None)


def test_registration_and_show_dispatch(noop_backend):
    """A registered backend receives the generic frame structure."""
    name, calls = noop_backend
    assert name in RegisteredBackend.backends

    data = magpy.show(make_source(), backend=name)

    assert set(data) == {"frames", "ranges", "labels", "input_kwargs"}
    assert len(calls) == 1
    frame = data["frames"][0]
    assert {"name", "data", "extra_backend_traces", "layout"} <= set(frame)
    assert frame["data"], "expected at least one trace"


def test_registered_backend_is_selectable_everywhere(noop_backend):
    """The registry is the single source of truth for every backend field."""
    name, _ = noop_backend

    default_backend = magpy.defaults.display.backend
    try:
        magpy.defaults.display.backend = name
        assert magpy.defaults.display.backend == name
    finally:
        magpy.defaults.display.backend = default_backend

    obj = make_source()
    obj.style.model3d.add_trace(backend=name, constructor="Mesh3d")
    assert obj.style.model3d.data[0].backend == name


def test_unsupported_feature_warns_and_falls_back(noop_backend):
    """An unsupported feature warns and is replaced by the fallback value."""
    name, _ = noop_backend
    src = make_source()
    src.move([(0, 0, 0), (0, 0, 1)], start=0)

    with pytest.warns(UserWarning, match="does not support 'animation'"):
        magpy.show(src, backend=name, animation=True)


def test_unsupported_animation_output_warns(noop_backend):
    """A second fallback path, to pin the loop rather than one branch of it."""
    name, _ = noop_backend
    src = make_source()
    src.move([(0, 0, 0), (0, 0, 1)], start=0)

    with pytest.warns(UserWarning, match="does not support") as record:
        magpy.show(src, backend=name, animation=True, animation_output="gif")

    messages = [str(w.message) for w in record]
    assert any("does not support 'animation_output'" in m for m in messages)


def test_subplot_grid_collapses_for_backend_without_subplots(noop_backend):
    """A homogeneous grid warns and is flattened onto a single plot.

    The grid is not visible in the dispatch kwargs -- `row`/`col` are consumed
    before `show` reaches the backend -- so this is detected from the resolved
    `max_rows`/`max_cols` instead.
    """
    name, calls = noop_backend
    a, b = make_source(), make_source()
    b.position = (3, 0, 0)

    with pytest.warns(UserWarning, match="does not support 'subplots'"):
        data = magpy.show(
            {"objects": [a], "row": 1, "col": 1},
            {"objects": [b], "row": 2, "col": 1},
            backend=name,
        )

    assert calls[-1]["max_rows"] is None
    assert calls[-1]["max_cols"] is None
    placements = {
        (trace.get("row"), trace.get("col"))
        for frame in data["frames"]
        for trace in frame["data"]
    }
    assert placements == {(1, 1)}


def test_mixed_grid_is_not_silently_flattened(noop_backend):
    """A grid mixing 3D and 2D panels has no single-plot equivalent."""
    name, _ = noop_backend
    src = make_source()
    sensor = magpy.Sensor(position=[(0, 0, i) for i in range(3)])

    with pytest.warns(UserWarning, match="mixes 3D and 2D panels"):
        magpy.show(
            {"objects": [src, sensor], "row": 1, "col": 1, "output": "model3d"},
            {"objects": [src, sensor], "row": 2, "col": 1, "output": "Bx"},
            backend=name,
        )


def test_no_subplot_warning_without_a_grid(noop_backend):
    """The common single-plot case must stay silent."""
    name, _ = noop_backend
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        magpy.show(make_source(), backend=name)


def test_builtin_backends_keep_their_subplot_support():
    """Built-ins all declare subplot support, so nothing regresses for them."""
    for builtin in magpy.SUPPORTED_PLOTTING_BACKENDS:
        assert RegisteredBackend.backends[builtin].supports["subplots"] is True


def test_unknown_backend_lists_registered_names(noop_backend):
    """The error message reports what is registered, not a frozen tuple."""
    name, _ = noop_backend
    with pytest.raises(MagpylibBadUserInput, match="Input backend must be one of"):
        magpy.show(make_source(), backend="nope")

    # the runtime-registered name is offered as a valid alternative
    with pytest.raises(MagpylibBadUserInput, match=rf"'{name}'"):
        magpy.show(make_source(), backend="nope")


def test_builtin_backends_are_registered():
    """The built-ins go through the same registry as third parties."""
    for name in magpy.SUPPORTED_PLOTTING_BACKENDS:
        assert name in RegisteredBackend.backends


def test_supported_plotting_backends_still_lists_builtins_only(noop_backend):
    """The public constant keeps its meaning: built-ins, not the registry."""
    name, _ = noop_backend
    assert name in RegisteredBackend.backends
    assert name not in magpy.SUPPORTED_PLOTTING_BACKENDS
    assert magpy.SUPPORTED_PLOTTING_BACKENDS == ("matplotlib", "plotly", "pyvista")

"""Tests for the display-backend registry."""

import warnings

import pytest

import magpylib as magpy
from magpylib._src.display.api import DisplayBackend
from magpylib._src.display.backend_registry import RegisteredBackend
from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib.graphics import backend as public_backend


def make_source():
    """A minimal displayable source."""
    return magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))


@pytest.fixture
def noop_backend():
    """Register a backend returning the generic data untouched.

    Yields ``(name, calls)``, where `calls` collects the Scene each `show`
    dispatch handed to the backend.
    """
    name = "noop"
    calls = []

    def show_func(scene):
        calls.append(scene)
        return scene

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
        DisplayBackend.backends.pop(name, None)


def test_registration_and_show_dispatch(noop_backend):
    """A registered backend receives the generic frame structure."""
    name, calls = noop_backend
    assert name in DisplayBackend.backends

    scene = magpy.show(make_source(), backend=name)

    assert len(calls) == 1
    assert scene.frames, "expected at least one frame"
    frame = scene.frames[0]
    assert frame.traces, "expected at least one trace"
    # traces stay plain dicts in magpylib's dialect, so a new key or a new
    # trace type needs no API change
    assert isinstance(frame.traces[0], dict)
    assert frame.traces[0]["type"] in {"mesh3d", "scatter3d", "scatter"}
    assert scene.panels
    assert scene.panels[0].kind == "scene3d"


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
    name, _ = noop_backend
    a, b = make_source(), make_source()
    b.position = (3, 0, 0)

    with pytest.warns(UserWarning, match="does not support 'subplots'"):
        scene = magpy.show(
            {"objects": [a], "row": 1, "col": 1},
            {"objects": [b], "row": 2, "col": 1},
            backend=name,
        )

    assert scene.has_subplots is False
    assert (scene.n_rows, scene.n_cols) == (1, 1)
    placements = {
        (trace.get("row"), trace.get("col"))
        for frame in scene.frames
        for trace in frame.traces
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
        assert DisplayBackend.backends[builtin].supports["subplots"] is True


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
        assert name in DisplayBackend.backends


def test_supported_plotting_backends_still_lists_builtins_only(noop_backend):
    """The public constant keeps its meaning: built-ins, not the registry."""
    name, _ = noop_backend
    assert name in DisplayBackend.backends
    assert name not in magpy.SUPPORTED_PLOTTING_BACKENDS
    assert magpy.SUPPORTED_PLOTTING_BACKENDS == ("matplotlib", "plotly", "pyvista")


def test_backends_are_display_backend_subclasses():
    """The built-ins go through the same class a third party would subclass."""
    for name in magpy.SUPPORTED_PLOTTING_BACKENDS:
        assert isinstance(DisplayBackend.backends[name], DisplayBackend)


def test_declaring_a_subclass_registers_it():
    """Subclassing with a `name` is the declarative way to register."""

    class Declarative(DisplayBackend):
        name = "declarative"
        supports_animation = True
        supports_subplots = True
        supports_colorgradient = True

        def show(self, scene):
            return scene

    try:
        assert "declarative" in DisplayBackend.backends
        scene = magpy.show(make_source(), backend="declarative")
        assert scene.frames
    finally:
        DisplayBackend.backends.pop("declarative", None)


def test_api_version_mismatch_warns():
    """A backend written against an older payload must not fail silently."""

    class Stale(DisplayBackend):
        name = "stale"
        api_version = 0
        supports_animation = True
        supports_subplots = True
        supports_colorgradient = True

        def show(self, scene):
            return scene

    try:
        with pytest.warns(UserWarning, match="api_version"):
            magpy.show(make_source(), backend="stale")
    finally:
        DisplayBackend.backends.pop("stale", None)


def test_undeclared_trace_type_warns():
    """`handles_traces` is what makes adding a trace type safe."""

    class Picky(DisplayBackend):
        name = "picky"
        handles_traces = frozenset({"scatter3d"})
        supports_animation = True
        supports_subplots = True
        supports_colorgradient = True

        def show(self, scene):
            return scene

    try:
        with pytest.warns(UserWarning, match=r"does not declare support.*mesh3d"):
            magpy.show(make_source(), backend="picky")
    finally:
        DisplayBackend.backends.pop("picky", None)


def test_handles_traces_none_means_assume_all():
    """The default must stay silent -- no warning for backends that don't declare."""

    class Quiet(DisplayBackend):
        name = "quiet"
        supports_animation = True
        supports_subplots = True
        supports_colorgradient = True

        def show(self, scene):
            return scene

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            magpy.show(make_source(), backend="quiet")
    finally:
        DisplayBackend.backends.pop("quiet", None)


def test_register_backend_is_public_api():
    """Adding a backend must not require reaching into `_src`."""
    assert "register_backend" in magpy.__all__
    assert callable(magpy.register_backend)


def test_public_register_backend_defaults_capabilities_off():
    """Capabilities default False so a later magpylib cannot change behaviour."""
    try:
        magpy.register_backend("minimal", lambda scene: scene)
        backend = DisplayBackend.backends["minimal"]
        assert backend.supports == {
            "animation": False,
            "subplots": False,
            "colorgradient": False,
            "animation_output": False,
            "native_traces": True,
        }
        assert magpy.show(make_source(), backend="minimal").frames
    finally:
        DisplayBackend.backends.pop("minimal", None)


def test_backend_api_is_importable_from_the_public_module():
    """A backend author imports from magpylib.graphics.backend, not _src."""
    assert public_backend.DisplayBackend is DisplayBackend
    assert public_backend.ENTRY_POINT_GROUP == "magpylib.backends"
    for name in ("Scene", "Panel", "Frame", "AnimationSettings", "API_VERSION"):
        assert hasattr(public_backend, name)


def test_entry_point_discovery_is_lazy_and_runs_once(monkeypatch):
    """Entry points are resolved on first lookup, never at import."""
    calls = []

    def fake_entry_points(*, group):
        calls.append(group)
        return []

    # patch where the name is bound, not where it is defined
    monkeypatch.setattr(
        "magpylib._src.display.api.entry_points", fake_entry_points, raising=True
    )
    monkeypatch.setattr(DisplayBackend, "_discovered", False, raising=False)

    DisplayBackend.discover()
    DisplayBackend.discover()

    assert calls == ["magpylib.backends"], "discovery must run exactly once"


def test_a_broken_entry_point_warns_and_does_not_break_show(monkeypatch):
    """One bad plugin must not take down every figure."""

    class BadEntry:
        name = "broken"
        value = "nonexistent.module:Backend"

        def load(self):
            msg = "no such module"
            raise ImportError(msg)

    monkeypatch.setattr(
        "magpylib._src.display.api.entry_points",
        lambda *, group: [BadEntry()],  # noqa: ARG005
        raising=True,
    )
    monkeypatch.setattr(DisplayBackend, "_discovered", False, raising=False)

    with pytest.warns(UserWarning, match="Could not load display backend"):
        DisplayBackend.discover()

    # unrelated backends keep working
    assert magpy.show(make_source(), backend="matplotlib", return_fig=True)

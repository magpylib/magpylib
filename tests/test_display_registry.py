"""Tests for the display-backend registry."""

import dataclasses
import subprocess
import sys
import types
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

import magpylib as magpy
from magpylib._src.display.api import DisplayBackend
from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib.graphics import backend as public_backend
from magpylib.graphics.backend import TRACE_META_KEYS, drawing_properties


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

    magpy.register_backend(name, show_func)
    try:
        yield name, calls
    finally:
        DisplayBackend.backends.pop(name, None)


def test_registration_and_show_dispatch(noop_backend):
    """A registered backend receives the generic frame structure."""
    name, calls = noop_backend
    assert name in DisplayBackend.backends

    scene = magpy.show(make_source(), backend=name, return_fig=True)

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
            return_fig=True,
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
    """The error message reports what is registered, not a frozen tuple, with
    each backend's description and url, which nothing showed before."""
    name, _ = noop_backend
    with pytest.raises(MagpylibBadUserInput, match="Input backend must be one of"):
        magpy.show(make_source(), backend="nope")

    # the runtime-registered name is offered as a valid alternative
    with pytest.raises(MagpylibBadUserInput, match=rf"'{name}'"):
        magpy.show(make_source(), backend="nope")

    magpy.register_backend(
        "described", lambda scene: scene, description="A demo", url="https://x.y"
    )
    with pytest.raises(
        MagpylibBadUserInput, match=r"'described' \(A demo, https://x.y\)"
    ):
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
        scene = magpy.show(make_source(), backend="declarative", return_fig=True)
        assert scene.frames
    finally:
        DisplayBackend.backends.pop("declarative", None)


@pytest.mark.parametrize(
    ("declared", "emitted", "warning"),
    [
        (0, 1, r"api_version 0, older"),
        (None, 1, None),  # unset counts as 1: silent today
        (None, 2, r"does not declare api_version, so counts as 1, older"),
    ],
)
def test_api_version_mismatch_warns(declared, emitted, warning, monkeypatch):
    """A backend written against an older payload must not fail silently --
    including one that never said which: it used to inherit the current
    version, so the warning could not fire for it."""
    monkeypatch.setattr("magpylib._src.display.backend_registry.API_VERSION", emitted)
    try:
        magpy.register_backend("stale", lambda scene: scene, api_version=declared)
        if warning is None:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                magpy.show(make_source(), backend="stale")
        else:
            with pytest.warns(UserWarning, match=warning):
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
            "native_traces": False,
        }
        assert magpy.show(make_source(), backend="minimal", return_fig=True).frames
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


def test_discovery_through_a_subclass_flags_the_base(monkeypatch):
    """`discover()` must mark the base done, not just the calling subclass.

    Setting the flag on `cls` left DisplayBackend._discovered False, so the
    base would resolve entry points a second time later.
    """
    calls = []
    monkeypatch.setattr(
        "magpylib._src.display.api.entry_points",
        lambda *, group: calls.append(group) or [],
        raising=True,
    )
    monkeypatch.setattr(DisplayBackend, "_discovered", False, raising=False)

    class Sub(DisplayBackend):
        name = "sub_probe"

        def show(self, scene):
            return scene

    try:
        Sub.discover()
        DisplayBackend.discover()
        assert calls == ["magpylib.backends"], "base rediscovered after subclass call"
    finally:
        DisplayBackend.backends.pop("sub_probe", None)


def test_discovery_does_not_reinstantiate_a_registered_backend(monkeypatch):
    """Defining the class already registers it; discovery must not build it again."""
    built = []

    class Counting(DisplayBackend):
        name = "counting_probe"

        def __init__(self):
            built.append(1)
            super().__init__()

        def show(self, scene):
            return scene

    class Entry:
        name = "counting_probe"
        value = "pkg:Counting"

        def load(self):
            return Counting

    monkeypatch.setattr(
        "magpylib._src.display.api.entry_points",
        lambda *, group: [Entry()],  # noqa: ARG005
        raising=True,
    )
    monkeypatch.setattr(DisplayBackend, "_discovered", False, raising=False)

    try:
        before = len(built)
        DisplayBackend.discover()
        assert len(built) == before
    finally:
        DisplayBackend.backends.pop("counting_probe", None)


@pytest.mark.parametrize("name", ["matplotlib", "plotly", "pyvista"])
def test_builtins_declare_the_trace_types_they_draw(name):
    """`handles_traces` must be real on the built-ins, not just plumbing.

    Otherwise the warning can never fire for a shipped backend and a newly
    added trace type would go unnoticed.
    """
    backend = DisplayBackend.backends[name]
    assert backend.handles_traces == frozenset({"mesh3d", "scatter3d", "scatter"})


@pytest.mark.parametrize("name", ["matplotlib", "plotly", "pyvista"])
def test_builtins_do_not_warn_about_their_own_traces(name):
    """Declaring the set must not make ordinary figures noisy."""
    obj = make_source()
    sensor = magpy.Sensor(position=(3, 0, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        magpy.show(
            {"objects": [obj, sensor], "row": 1, "col": 1},
            {"objects": [obj, sensor], "row": 1, "col": 2, "output": "Bx"},
            backend=name,
            return_fig=True,
        )


@pytest.mark.parametrize(
    ("name", "accepted"),
    [
        ("matplotlib", {"antialiased", "return_animation", "fig_*", "show_*"}),
        ("plotly", {"renderer", "fig_*", "show_*"}),
        ("pyvista", {"jupyter_backend", "mp4_quality", "fig_*", "show_*"}),
    ],
)
def test_builtins_declare_the_options_they_accept(name, accepted):
    """Declared so a misspelled argument can be reported rather than ignored.
    The two starred namespaces are what each forwards to its library."""
    assert DisplayBackend.backends[name].accepts_options == frozenset(accepted)


@pytest.mark.parametrize("name", ["matplotlib", "plotly", "pyvista"])
def test_misspelled_keyword_argument_warns(name):
    """`show(retrun_fig=True)` used to do nothing at all, silently."""
    with pytest.warns(UserWarning, match=r"unexpected keyword argument.*retrun_fig"):
        magpy.show(make_source(), backend=name, return_fig=True, retrun_fig=True)


@pytest.mark.parametrize(
    ("name", "option"),
    [
        ("matplotlib", {"antialiased": False}),
        ("plotly", {"renderer": None}),
        ("pyvista", {"jupyter_backend": None}),
    ],
)
def test_declared_options_do_not_warn(name, option):
    """A legitimate backend-specific argument must stay silent."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        magpy.show(make_source(), backend=name, return_fig=True, **option)


def test_return_fig_false_discards_the_backend_figure():
    """`show()` documents that it returns None unless `return_fig` is set.

    All three built-in backends return None of their own accord, so the
    promise held by convention; a third-party backend returning its figure
    unconditionally leaked it into the caller.
    """
    try:
        magpy.register_backend("leaky", lambda scene: "THE FIGURE")  # noqa: ARG005
        assert magpy.show(make_source(), backend="leaky") is None
        assert (
            magpy.show(make_source(), backend="leaky", return_fig=True) == "THE FIGURE"
        )
    finally:
        DisplayBackend.backends.pop("leaky", None)


def test_undeclared_accepts_options_accepts_anything():
    """The default must not make existing third-party backends noisy."""
    try:
        magpy.register_backend("anything", lambda scene: scene)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            magpy.show(make_source(), backend="anything", whatever=1)
    finally:
        DisplayBackend.backends.pop("anything", None)


# --- Step 3: per-object identity ---------------------------------------------


def _traces_of(*objs, backend_name, **kwargs):
    captured = {}

    magpy.register_backend(
        backend_name,
        lambda scene: captured.setdefault(
            "t", [t for f in scene.frames for t in f.traces]
        ),
        supports_animation=True,
        supports_subplots=True,
        supports_colorgradient=True,
    )
    try:
        magpy.show(*objs, backend=backend_name, **kwargs)
        return captured["t"]
    finally:
        DisplayBackend.backends.pop(backend_name, None)


def test_object_id_identifies_unmerged_traces():
    """Separate objects never merge, so identity works without any flag."""
    a, b = make_source(), make_source()
    b.position = (3, 0, 0)

    traces = _traces_of(a, b, backend_name="id_plain")

    assert {t["object_id"] for t in traces} == {id(a), id(b)}


def test_merged_traces_carry_no_object_id():
    """A fused trace must report None, never the first object's id.

    Merges take unlisted keys from the first trace, so without reconciliation
    a Collection's fused mesh would claim to be one of its children.
    """
    a, b = make_source(), make_source()
    b.position = (3, 0, 0)
    collection = magpy.Collection(a, b, style_label="Ring")

    traces = _traces_of(collection, backend_name="id_merged")

    assert len(traces) == 1, "default styling should still merge the children"
    assert traces[0]["object_id"] is None


def test_merge_traces_false_unmerges_collection_children():
    """The flag's one real effect: Collection children become identifiable."""
    a, b = make_source(), make_source()
    b.position = (3, 0, 0)
    collection = magpy.Collection(a, b, style_label="Ring")

    class Unmerged(DisplayBackend):
        name = "id_unmerged"
        merge_traces = False
        supports_animation = True
        supports_subplots = True
        supports_colorgradient = True

        def show(self, scene):
            return [t for f in scene.frames for t in f.traces]

    try:
        traces = magpy.show(collection, backend="id_unmerged", return_fig=True)
        assert {t["object_id"] for t in traces} == {id(a), id(b)}
    finally:
        DisplayBackend.backends.pop("id_unmerged", None)


def test_object_id_survives_intra_object_merges():
    """The four within-object merges must keep the id, not blank it.

    A path-varying object has its path-frame copies and sub-traces merged
    before any backend sees them; that is one object, so the id stands.
    """
    obj = make_source()
    obj.position = [(0, 0, 0), (0, 0, 1), (0, 0, 2)]
    obj.style.path.show = True

    traces = _traces_of(obj, backend_name="id_path")

    assert traces
    assert {t["object_id"] for t in traces} == {id(obj)}


@pytest.mark.parametrize("name", ["matplotlib", "plotly", "pyvista"])
def test_metadata_keys_never_reach_the_plotting_library(name):
    """object_id, row and col must be stripped: plotly rejects unknown
    properties outright, and the backend guide's own example,
    `go.Mesh3d(**drawing_properties(trace))`, failed on 'row'."""
    assert "object_id" in TRACE_META_KEYS
    assert {"row", "col"} == public_backend.PLACEMENT_KEYS
    assert drawing_properties({"x": 1, "object_id": 7, "row": 1, "col": 2}) == {"x": 1}
    if name == "plotly":
        import plotly.graph_objects as go  # noqa: PLC0415

        scene = _scene_of(make_source(), backend_name="example_probe")
        (mesh,) = (t for t in scene.frames[0].traces if t["type"] == "mesh3d")
        go.Mesh3d(**drawing_properties(mesh))
    # a real render, subplots included, must not raise
    assert (
        magpy.show(
            {"objects": [make_source()], "col": 1},
            {"objects": [make_source()], "col": 2},
            backend=name,
            return_fig=True,
        )
        is not None
    )


def _scene_of(*objs, backend_name, **kwargs):
    """The `Scene` handed to a backend that declares every capability and
    keeps each object's traces apart, so every trace names its object."""
    magpy.register_backend(
        backend_name,
        lambda scene: scene,
        supports_animation=True,
        supports_subplots=True,
        supports_colorgradient=True,
        merge_traces=False,
    )
    try:
        return magpy.show(*objs, backend=backend_name, return_fig=True, **kwargs)
    finally:
        DisplayBackend.backends.pop(backend_name, None)


def test_a_panel_carries_its_objects_as_given():
    """Collections whole and in order: the hierarchy no trace describes.

    Every trace under a Collection reports the outermost one as its
    legendgroup, so a backend drawing a nested legend needs the objects --
    and they are also what an object_id resolves against.
    """
    inner = magpy.Collection(make_source(), make_source(), style_label="inner")
    outer = magpy.Collection(inner, make_source(), style_label="outer")
    loose = make_source()
    loose.position = (3, 0, 0)

    scene = _scene_of(outer, loose, backend_name="objects_given")

    (panel,) = scene.panels
    assert [id(o) for o in panel.objects] == [id(outer), id(loose)]
    assert panel.objects[0].children[0] is inner
    # every object drawn, however deep, is reachable from them
    reachable = {
        id(o) for top in panel.objects for o in (top, *getattr(top, "children_all", ()))
    }
    magnets = {id(o) for o in (*inner.children, outer.children[1], loose)}
    ids = {t["object_id"] for f in scene.frames for t in f.traces}
    assert ids == magnets
    assert ids <= reachable


def test_panel_objects_are_as_passed_not_deduplicated_across_nesting():
    """A child passed beside its collection is there twice, as it was given:
    once on its own, once inside -- the backend decides how to list it."""
    child = make_source()
    collection = magpy.Collection(child, make_source())

    scene = _scene_of(collection, child, [child], backend_name="objects_twice")

    (panel,) = scene.panels
    assert [id(o) for o in panel.objects] == [id(collection), id(child)]


def test_each_cell_carries_its_own_objects():
    a, b = make_source(), make_source()
    b.position = (3, 0, 0)

    scene = _scene_of(
        {"objects": [a], "row": 1, "col": 1},
        {"objects": [b, a], "row": 1, "col": 2},
        backend_name="objects_by_cell",
    )

    assert [id(o) for o in scene.panel(1, 1).objects] == [id(a)]
    assert [id(o) for o in scene.panel(1, 2).objects] == [id(b), id(a)]


def test_panel_kind_defaults_absent_cells_to_3d():
    """Every cell of a grid needs a kind, including empty ones."""
    scene = public_backend.Scene(
        panels=(
            public_backend.Panel(row=1, col=1, kind="scene3d"),
            public_backend.Panel(row=1, col=2, kind="chart2d"),
        )
    )
    assert scene.panel_kind(1, 1) == "scene3d"
    assert scene.panel_kind(1, 2) == "chart2d"
    assert scene.panel_kind(9, 9) == "scene3d"


@pytest.mark.parametrize(
    "spelling",
    [
        {"renderer": "png"},
        {"probe_renderer": "png"},
        {"probe": {"renderer": "png"}},
    ],
)
def test_backend_prefixed_options_reach_the_backend(spelling):
    """All three spellings of a backend argument must arrive.

    Backend-prefixed arguments that were neither fig_ nor show_ used to be
    extracted from kwargs and then dropped, so `plotly_renderer=...` silently
    did nothing -- and could not even be reported as a typo, because it never
    reached `Scene.options`.
    """
    seen = []
    try:
        magpy.register_backend("probe", lambda scene: seen.append(dict(scene.options)))
        magpy.show(make_source(), backend="probe", **spelling)
        assert seen[0] == {"renderer": "png"}
    finally:
        DisplayBackend.backends.pop("probe", None)


# --- scene-dependent geometry ------------------------------------------------


def test_units_length_default_scales_what_the_backend_receives():
    """`defaults.display.units.length` sets the unit of the payload coordinates.

    With the default 'auto' the unit -- and so every coordinate handed over --
    is inferred from the extent of the whole scene. Pinning it is what makes
    the geometry a backend receives depend only on the objects themselves,
    which is what a backend splicing single objects into an existing figure
    needs. A per-call `units_length` still wins over the default.
    """

    def extent(**kwargs):
        traces = _traces_of(make_source(), backend_name="units_probe", **kwargs)
        xyz = np.concatenate(
            [
                np.stack([t["x"], t["y"], t["z"]], axis=1)
                for t in traces
                if t["type"] == "mesh3d"
            ]
        )
        return round(float(np.ptp(xyz, axis=0).max()), 6)

    assert extent() == 1.0  # a 1 m cube, 'auto' resolves to 'm'

    magpy.defaults.display.units.length = "mm"
    assert extent() == 1000.0

    assert extent(units_length="km") == 0.001


def test_discovery_waits_until_the_import_has_finished(monkeypatch):
    """A lookup during `import magpylib` must not resolve entry points.

    The defaults tree validates its own default backend while this package is
    still executing, and that is a backend-name lookup. Resolving entry points
    there loads third-party code into a half-built magpylib: the module a
    backend is told to subclass from -- `magpylib.graphics.backend` -- imports
    the style and display stack, which comes back round to `default_settings`
    before that name is bound. Every backend shipped the documented way failed
    with a circular import, and none of them registered.
    """
    calls = []
    monkeypatch.setattr(
        "magpylib._src.display.api.entry_points",
        lambda *, group: calls.append(group) or [],
        raising=True,
    )
    monkeypatch.setattr(DisplayBackend, "_discovered", False, raising=False)
    monkeypatch.setattr(DisplayBackend, "_importing", True, raising=False)

    DisplayBackend.discover()

    assert calls == [], "no entry point may be loaded while magpylib imports"
    assert not DisplayBackend._discovered, "the scan must still be pending"

    monkeypatch.setattr(DisplayBackend, "_importing", False, raising=False)
    DisplayBackend.discover()

    assert calls == ["magpylib.backends"], "and run on the first lookup after"


def test_importing_magpylib_resolves_no_entry_points():
    """End to end, in an interpreter that has only just imported magpylib.

    In-process this cannot be observed -- by the time a test runs, something
    has looked a backend name up -- so it is asked of a fresh one.
    """
    code = (
        "import magpylib\n"
        "from magpylib._src.display.api import DisplayBackend\n"
        "print(DisplayBackend._discovered, DisplayBackend._importing)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "False"], out.stderr


def test_an_entry_point_that_is_not_a_backend_warns(monkeypatch):
    """Silence here is the worst outcome: the name advertised simply is not
    there, and the package looks installed and inert."""

    class NotABackend:
        name = "impostor"
        value = "somewhere:Thing"

        def load(self):
            return None

    monkeypatch.setattr(
        "magpylib._src.display.api.entry_points",
        lambda *, group: [NotABackend()],  # noqa: ARG005
        raising=True,
    )
    monkeypatch.setattr(DisplayBackend, "_discovered", False, raising=False)

    with pytest.warns(UserWarning, match="registered no backend"):
        DisplayBackend.discover()

    assert "impostor" not in DisplayBackend.backends


def test_an_entry_point_naming_a_module_is_not_reported_as_broken(monkeypatch):
    """Importing the module is what registers the backend, so nothing is wrong.

    The entry-point value may name the class or the module defining it. In the
    second case `load()` returns a module, and the backend is already in the
    registry by then -- reporting that as "not a DisplayBackend" would call a
    working install broken, and under `-W error` would break it.
    """
    module = types.ModuleType("fake_backend_package")

    class ModuleBackend(DisplayBackend):
        name = "from_a_module"

        def show(self, scene):
            return scene

    module.ModuleBackend = ModuleBackend

    class Entry:
        name = "from_a_module"
        value = "fake_backend_package"

        def load(self):
            return module

    monkeypatch.setattr(
        "magpylib._src.display.api.entry_points",
        lambda *, group: [Entry()],  # noqa: ARG005
        raising=True,
    )
    monkeypatch.setattr(DisplayBackend, "_discovered", False, raising=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DisplayBackend.discover()

    assert "from_a_module" in DisplayBackend.backends


def test_the_report_names_the_object_it_got(monkeypatch):
    """`type(loaded).__name__` is the metaclass for a class -- "type", which
    tells the author nothing about what magpylib actually found."""

    class Plain:
        pass

    class Entry:
        name = "plain"
        value = "somewhere:Plain"

        def load(self):
            return Plain

    monkeypatch.setattr(
        "magpylib._src.display.api.entry_points",
        lambda *, group: [Entry()],  # noqa: ARG005
        raising=True,
    )
    monkeypatch.setattr(DisplayBackend, "_discovered", False, raising=False)

    with pytest.warns(UserWarning, match="Plain"):
        DisplayBackend.discover()


# --- declarations are checked where they are written -------------------------


@pytest.mark.parametrize(
    ("attr", "value", "outcome"),
    [
        ("handles_traces", ["scatter3d"], frozenset({"scatter3d"})),  # coerced
        ("accepts_options", (n for n in ["x"]), frozenset({"x"})),  # read once
        ("accepts_options", {"*"}, frozenset({"*"})),
        ("api_version", np.int64(1), 1),
        ("api_version", None, None),  # unset: counts as 1
        ("handles_traces", "mesh3d", TypeError),  # a string is not a set of names
        ("accepts_options", 3, TypeError),
        ("accepts_options", {"fig*"}, ValueError),  # a namespace ends in `_*`
        ("supports_animation", "yes", TypeError),
        ("api_version", "1", TypeError),  # used to warn that 1 differs from 1
        ("api_version", True, TypeError),
        ("description", 3, TypeError),
        ("name", 3, TypeError),
        ("name", "auto", ValueError),  # show()'s inference, never a backend
    ],
)
def test_a_declaration_is_checked_at_the_class_statement(attr, value, outcome):
    """Reported where it is written, naming the attribute: a list in
    `handles_traces` used to fail at the first show(), from inside a set
    difference in the dispatcher."""
    body = {attr: value, "show": staticmethod(lambda scene: scene)}
    if attr != "name":
        body["name"] = "checked"
    try:
        if isinstance(outcome, type) and issubclass(outcome, Exception):
            with pytest.raises(outcome, match=attr):
                type("Checked", (DisplayBackend,), body)
            assert "checked" not in DisplayBackend.backends
        else:
            type("Checked", (DisplayBackend,), body)
            assert getattr(DisplayBackend.backends["checked"], attr) == outcome
    finally:
        DisplayBackend.backends.pop("checked", None)


@pytest.mark.parametrize(
    ("name", "keyword", "error"),
    [
        ("leak", {"_discovered": True}, TypeError),  # any attribute used to pass
        (None, {}, TypeError),  # used to fail on `.title()`
        ("", {}, ValueError),  # used to register nothing, silently
    ],
)
def test_register_backend_checks_its_arguments(name, keyword, error):
    with pytest.raises(error, match=r"name|unknown backend capability"):
        magpy.register_backend(name, lambda scene: scene, **keyword)
    assert "leak" not in DisplayBackend.backends


def test_a_class_from_elsewhere_replacing_a_backend_is_reported():
    """A notebook re-running its cell is the same module and stays quiet; a
    package, or this test module, taking over 'matplotlib' is worth a word.
    Subclassing a registered backend without a `name` of its own registers
    nothing: it used to replace what it subclassed."""
    builtin = DisplayBackend.backends["matplotlib"]

    class Tweaked(type(builtin)):
        pass

    assert DisplayBackend.backends["matplotlib"] is builtin

    with pytest.warns(UserWarning, match=r"replaces the 'matplotlib' display backend"):

        class Impostor(DisplayBackend):
            name = "matplotlib"

            def show(self, scene):
                return scene

    assert isinstance(DisplayBackend.backends["matplotlib"], Impostor)


# --- option routing ----------------------------------------------------------


@pytest.mark.parametrize(
    ("accepts", "given", "expected", "warning"),
    [
        # `stylesheet` is the backend's, `style_color` is magpylib's: the
        # underscore decides. `stylesheet` used to become the style property
        # 'heet'; `figure` and `showcase` vanished.
        (
            {"stylesheet", "figure", "showcase"},
            {"stylesheet": "dark", "figure": 1, "showcase": 2},
            {"stylesheet": "dark", "figure": 1, "showcase": 2},
            None,
        ),
        # names that never reach a backend are reported when declared
        ({"legend_position", "zoom"}, {}, {}, r"legend_position.*zoom.*reserves"),
        # fig_*/show_* are plain options, the dict spelling flattened one
        # level with values verbatim; `Scene` has no field for them
        (
            None,
            {"fig_width": 400, "show_block": False, "fig": {"layout": {"h": 3}}},
            {"fig_width": 400, "fig_layout": {"h": 3}, "show_block": False},
            None,
        ),
        # undeclared, `fig_width` is the typo it looks like on a backend with
        # no figure constructor; a bare `figsize` used to vanish on any backend
        (set(), {"fig_width": 400}, {"fig_width": 400}, r"unexpected.*fig_width"),
        ({"fig_*", "show_*"}, {"figsize": 3}, {"figsize": 3}, r"unexpected.*figsize"),
        # a trailing star accepts its namespace, and only that one
        ({"fig_*"}, {"fig_x": 1, "show_block": False}, None, r"unexpected.*show_block"),
        # `probe_fig_title` landed in two places and was reported as ignored
        ({"fig_*"}, {"probe_fig_title": "x"}, {"fig_title": "x"}, None),
        # magpylib's own arguments are read from the bare spelling only, and a
        # typo is reported as typed, not with its prefix stripped
        (
            set(),
            {"probe_return_fig": True},
            {"return_fig": True},
            r"unexpected.*probe_return_fig",
        ),
        # the ladder from #657: dict < underscore < backend-prefixed dict <
        # backend-prefixed underscore, each flattened before the merge so a
        # dict at one level does not wipe the keys of the one below
        (
            None,
            {
                "fig": {"a": 1, "b": 1, "c": 1, "d": 1, "e": 1},
                "fig_b": 2,
                "fig_c": 2,
                "fig_d": 2,
                "probe": {"fig": {"c": 3, "d": 3}},
                "probe_fig": {"d": 4},
                "probe_fig_e": 5,
            },
            {"fig_a": 1, "fig_b": 2, "fig_c": 3, "fig_d": 4, "fig_e": 5},
            None,
        ),
    ],
)
def test_every_option_is_routed_through_accepts_options(
    accepts, given, expected, warning
):
    seen = []
    try:
        magpy.register_backend("probe", seen.append, accepts_options=accepts)
        if warning is None:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                magpy.show(make_source(), backend="probe", **given)
        else:
            with pytest.warns(UserWarning, match=warning) as caught:
                magpy.show(make_source(), backend="probe", **given)
            # on the caller's line, not in magpylib's frames
            assert all(w.filename == __file__ for w in caught)
    finally:
        DisplayBackend.backends.pop("probe", None)
    if expected is not None:
        assert seen[0].options == expected
    assert seen[0].return_fig is False
    fields = {f.name for f in dataclasses.fields(public_backend.Scene)}
    assert not fields & {"fig_kwargs", "show_kwargs"}


@pytest.mark.parametrize(
    ("name", "extra", "warning"),
    [
        # with `return_fig` the show call never runs, so `show_*` went nowhere
        ("matplotlib", {"return_fig": True, "show_block": False}, r"show_block"),
        ("plotly", {"return_fig": True, "renderer": "png"}, r"renderer"),
        ("pyvista", {"return_fig": True, "show_jupyter_backend": "x"}, r"show_jup"),
        # nor does `fig_*` on a canvas the user gave
        ("matplotlib", {"canvas": plt.figure, "fig_facecolor": "k"}, r"fig_facecolor"),
    ],
)
def test_builtins_report_what_they_cannot_use(name, extra, warning):
    extra = {k: v() if callable(v) else v for k, v in extra.items()}
    with pytest.warns(UserWarning, match=warning + r".*ignored"):
        magpy.show(make_source(), backend=name, **extra)


# a `Plotter` whose constructor rejected an argument is half-built, and
# pyvista's `__del__` raises again on it at garbage collection
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.parametrize(
    ("name", "given", "read", "expected"),
    [
        ("matplotlib", {"fig_figsize": (6, 6)}, "get_size_inches", (6, 6)),
        ("plotly", {"fig_layout": {"width": 600}}, "layout.width", 600),
        ("pyvista", {"fig_window_size": (600, 600)}, "window_size", (600, 600)),
        # main raised "multiple values for keyword argument 'off_screen'"
        ("pyvista", {"fig_off_screen": True}, "off_screen", True),
        ("matplotlib", {"fig_nonsense": 1}, None, AttributeError),
        ("plotly", {"fig_layout_width": -5}, None, ValueError),
        ("pyvista", {"fig_window_size": "x"}, None, IndexError),
        ("pyvista", {"fig_shape": (2, 2)}, None, MagpylibBadUserInput),  # magpylib's
    ],
)
def test_fig_arguments_reach_the_library(name, given, read, expected):
    """The value as given -- a revision once dropped the user's figsize -- and
    the library's own error when it refuses, keeping its type, with a note
    naming the show() argument. What the backend sets itself, such as
    pyvista's subplot grid, is not the user's to override."""
    if read is not None:
        value = magpy.show(make_source(), backend=name, return_fig=True, **given)
        for attr in read.split("."):
            value = getattr(value, attr)
        value = value() if callable(value) else value
        assert tuple(np.atleast_1d(value)) == tuple(np.atleast_1d(expected))
        return
    with pytest.raises(expected) as info:
        magpy.show(make_source(), backend=name, return_fig=True, **given)
    (argument,) = given
    if expected is not MagpylibBadUserInput:
        assert any(argument in note for note in info.value.__notes__)


# --- the payload, frozen for v6 ------------------------------------------------


def _payload_of_a_mixed_scene():
    """Every trace of a scene with unlabelled objects, a collection, a
    TriangularMesh and a 2D panel, built from fresh objects each time."""
    cube = make_source()
    mesh = magpy.magnet.TriangularMesh(
        polarization=(0, 0, 1),
        vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
        faces=[(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)],
        position=(3, 0, 0),
    )
    sensor = magpy.Sensor(position=[(0, 0, 2), (1, 0, 2)])
    group = magpy.Collection(cube, sensor)
    scene = _scene_of(
        {"objects": [group, mesh], "col": 1},
        {"objects": [group, mesh], "col": 2, "output": "Bz"},
        backend_name="payload_probe",
    )
    return [t for f in scene.frames for t in f.traces]


def test_the_payload_is_backend_neutral_and_deterministic():
    """No plotly settings (`showscale`, `lighting_*`) and no process-local id in
    any string: legend groups used to be a repr such as 'Cuboid(id=4510333120)'
    and the 2D hover text listed the same reprs, so two runs of one scene gave
    two payloads. `flatshading` stays, a hint any renderer can honour, and
    plotly still gets its settings."""
    first, second = _payload_of_a_mixed_scene(), _payload_of_a_mixed_scene()

    plotly_only = {"showscale", "lighting_ambient", "lighting_facenormalsepsilon"}
    assert not any(plotly_only & set(t) for t in first)
    assert any(t.get("flatshading") for t in first)

    def strings(traces):
        return [{k: v for k, v in t.items() if isinstance(v, str)} for t in traces]

    assert strings(first) == strings(second)
    assert not any("id=" in v for t in strings(first) for v in t.values())

    fig = magpy.show(
        magpy.magnet.TriangularMesh.from_ConvexHull(
            polarization=(0, 0, 1), points=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        ),
        backend="plotly",
        return_fig=True,
    )
    (mesh,) = (t for t in fig.data if t.type == "mesh3d")
    assert mesh.showscale is False
    assert (mesh.lighting.ambient, mesh.lighting.facenormalsepsilon) == (0.7, 0)


def test_the_pixel_helpers_are_public():
    """What magpylib-studio imported from `_src` to draw sensor pixels as
    instances, sized and coloured as the built-ins do them."""
    probe = magpy.Sensor(
        position=[(0, 0, 1), (0, 0, 2), (0, 0, 3)],
        pixel=[(0, 0, 0), (0.1, 0, 0)],
        style_pixel_field_source="Bxy",
    )
    fields = public_backend.sensor_pixel_fields([make_source(), probe, magpy.Sensor()])
    assert list(fields) == [probe]
    assert fields[probe]["B"].shape == (3, 2, 3)

    norms = [[0, 1, 2, 4]]  # 0 reads nothing
    scale = public_backend.scale_pixel_norms
    np.testing.assert_allclose(scale(norms, "linear"), [[np.nan, 0, 1 / 3, 1]])
    np.testing.assert_allclose(
        scale(norms, "linear", minimum=0.1), [[np.nan, 0.1, 0.4, 1]]
    )
    np.testing.assert_allclose(scale(norms, "uniform"), [[np.nan, 1, 1, 1]])
    with pytest.raises(ValueError, match="scaling"):
        scale(norms, "cubic")

    colors = public_backend.colors_from_colorscale(
        [0, 1, np.nan], "Viridis", cmin=0, cmax=1
    )
    assert list(colors) == ["#440154", "#fde725", "#b2beb5"]

"""Tests for the display-backend registry."""

import subprocess
import sys
import types
import warnings

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
        scene = magpy.show(make_source(), backend="declarative", return_fig=True)
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
        ("matplotlib", {"antialiased", "return_animation"}),
        ("plotly", {"renderer"}),
        ("pyvista", {"jupyter_backend", "mp4_quality"}),
    ],
)
def test_builtins_declare_the_options_they_accept(name, accepted):
    """Declared so a misspelled argument can be reported rather than ignored."""
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
    """object_id must be stripped: plotly rejects unknown properties outright."""
    assert "object_id" in TRACE_META_KEYS
    assert drawing_properties({"x": 1, "object_id": 7}) == {"x": 1}
    # a real render must not raise
    assert magpy.show(make_source(), backend=name, return_fig=True) is not None


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

"""Characterization tests for display style resolution.

Pins the precedence chain implemented by ``magpylib._src.style.get_style``:

    show() kwargs  >  obj.style  >  family defaults  >  base defaults  >  hardcoded

where family/base defaults live on ``magpy.defaults.display.style`` and an
explicitly set ``None`` on the object defers to the next layer down. Any
refactor of the style internals must keep this suite green.
"""

import copy
import json

import pytest

import magpylib as magpy
from magpylib._src.defaults.defaults_classes import default_settings
from magpylib._src.style import CurrentStyle, MagnetStyle, SensorStyle, get_style


@pytest.fixture(autouse=True)
def _reset_defaults():
    """Isolate each test from magpy.defaults mutations."""
    magpy.defaults.reset()
    yield
    magpy.defaults.reset()


def make_cuboid(**kwargs):
    """Return a simple magnet object of the 'magnet' style family."""
    return magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1), **kwargs)


# ------------------------------------------------------------------
# unit level: get_style
# ------------------------------------------------------------------


def test_magpy_defaults_is_default_settings():
    """The display machinery resolves against the same instance exposed as
    magpy.defaults, so user mutations of magpy.defaults must affect show()."""
    assert magpy.defaults is default_settings


def test_hardcoded_defaults_resolution():
    """Untouched objects resolve to the hardcoded base + family defaults and
    to the family-specific style class."""
    style = get_style(make_cuboid(), default_settings)
    assert isinstance(style, MagnetStyle)
    # base defaults
    assert style.opacity == 1
    assert style.path.line.width == 1
    assert style.path.marker.size == 3
    assert style.color is None  # colorsequence is applied later, not here
    # magnet family defaults
    assert style.magnetization.show is True
    assert style.magnetization.arrow.offset == 1
    # color validation lowercases hex values ("#E71111" in DEFAULTS)
    assert style.magnetization.color.north == "#e71111"

    style = get_style(magpy.current.Circle(current=1, diameter=1), default_settings)
    assert isinstance(style, CurrentStyle)
    # current family defaults
    assert style.arrow.offset == 0.5
    assert style.line.width == 2


def test_family_default_mutation_applies():
    """A mutated family default wins over the hardcoded value, only for
    objects of that family, without touching the objects themselves."""
    magpy.defaults.display.style.magnet.magnetization.show = False
    cuboid = make_cuboid()
    assert get_style(cuboid, default_settings).magnetization.show is False
    # the object style itself remains unset
    assert cuboid.style.magnetization.show is None
    # other families are unaffected
    sensor_style = get_style(magpy.Sensor(), default_settings)
    assert isinstance(sensor_style, SensorStyle)
    assert sensor_style.size == 1


def test_base_default_mutation_applies():
    """A mutated base default applies to all object families."""
    magpy.defaults.display.style.base.color = "black"
    assert get_style(make_cuboid(), default_settings).color == "black"
    assert get_style(magpy.Sensor(), default_settings).color == "black"


def test_object_style_overrides_defaults():
    """Values set on obj.style win over family and base defaults."""
    cuboid = make_cuboid()
    cuboid.style.magnetization.show = False  # family default is True
    cuboid.style.opacity = 0.5  # base default is 1
    style = get_style(cuboid, default_settings)
    assert style.magnetization.show is False
    assert style.opacity == 0.5


def test_show_kwargs_override_object_style():
    """style_* kwargs (as passed through show()) win over obj.style."""
    cuboid = make_cuboid()
    cuboid.style.magnetization.show = False
    cuboid.style.color = "blue"
    style = get_style(
        cuboid, default_settings, style_magnetization_show=True, style_color="red"
    )
    assert style.magnetization.show is True
    assert style.color == "red"


def test_explicit_none_defers_to_defaults():
    """An explicit None on obj.style falls through to the defaults layer,
    whatever that layer currently holds."""
    cuboid = make_cuboid()
    cuboid.style.magnetization.show = None
    cuboid.style.opacity = None
    style = get_style(cuboid, default_settings)
    assert style.magnetization.show is True  # hardcoded family default
    assert style.opacity == 1  # hardcoded base default

    magpy.defaults.display.style.magnet.magnetization.show = False
    assert get_style(cuboid, default_settings).magnetization.show is False


def test_style_dict_equals_underscore_kwargs():
    """The style dict form and the magic underscore form resolve identically."""
    cuboid = make_cuboid()
    from_dict = get_style(
        cuboid, default_settings, style={"magnetization": {"show": False}}
    )
    from_kwargs = get_style(cuboid, default_settings, style_magnetization_show=False)
    assert from_dict.as_dict() == from_kwargs.as_dict()


def test_constructor_kwarg_equals_attribute_set():
    """Styles set via constructor kwarg, style dict, or attribute assignment
    resolve identically."""
    via_kwarg = make_cuboid(style_label="mag1")
    via_dict = make_cuboid(style={"label": "mag1"})
    via_attr = make_cuboid()
    via_attr.style.label = "mag1"
    dicts = [
        get_style(obj, default_settings).as_dict()
        for obj in (via_kwarg, via_dict, via_attr)
    ]
    assert dicts[0]["label"] == "mag1"
    assert dicts[0] == dicts[1] == dicts[2]


def test_foreign_family_key_silently_filtered():
    """A style key valid for another family (magnetization on a Sensor) is
    filtered out without error and without corrupting resolution."""
    style = get_style(magpy.Sensor(), default_settings, style_magnetization_show=False)
    assert isinstance(style, SensorStyle)
    assert "magnetization" not in style.as_dict()


def test_invalid_style_key_raises():
    """A style key unknown to every family raises a ValueError."""
    with pytest.raises(ValueError, match="invalid"):
        get_style(make_cuboid(), default_settings, style_bananas=5)


def test_get_style_does_not_mutate_inputs():
    """Resolution must be pure: neither obj.style nor magpy.defaults may be
    modified by a get_style call with overriding kwargs."""
    cuboid = make_cuboid()
    cuboid.style.magnetization.show = False
    obj_before = cuboid.style.as_dict()
    defaults_before = magpy.defaults.display.style.as_dict()
    get_style(
        cuboid,
        default_settings,
        style_color="red",
        style_opacity=0.1,
        style_magnetization_show=True,
    )
    assert cuboid.style.as_dict() == obj_before
    assert magpy.defaults.display.style.as_dict() == defaults_before


def test_resolution_cache_invalidation_and_isolation():
    """The cached defaults layer must refresh on magpy.defaults mutations and
    every resolved style must be an independent copy of the cache entry."""
    cuboid = make_cuboid()
    assert get_style(cuboid, default_settings).magnetization.show is True
    # mutation after a cache-priming resolution must take effect
    magpy.defaults.display.style.magnet.magnetization.show = False
    assert get_style(cuboid, default_settings).magnetization.show is False

    # resolved styles are detached copies, not shared cache objects
    first = get_style(cuboid, default_settings)
    first.magnetization.arrow.width = 99
    second = get_style(cuboid, default_settings)
    assert second.magnetization.arrow.width != 99
    # mutable leaves (model3d traces) are not shared with the object style
    second.model3d.add_trace({"constructor": "Mesh3d"})
    assert cuboid.style.model3d.data == ()


# ------------------------------------------------------------------
# model3d traces: a collection of nodes inside the style tree
# ------------------------------------------------------------------

TRACE = {
    "backend": "generic",
    "constructor": "Mesh3d",
    "kwargs": {
        "i": [0],
        "j": [1],
        "k": [2],
        "x": [0, 1, 0],
        "y": [0, 0, 1],
        "z": [0] * 3,
    },
}


def test_model3d_data_is_always_a_tuple():
    """`model3d.data` reads as a tuple whether unset, filled or emptied, so
    traces can only be replaced, never mutated in place - the pattern the
    compound example relies on to rebuild itself."""
    cuboid = make_cuboid()
    model3d = cuboid.style.model3d
    assert model3d.data == ()

    model3d.add_trace(TRACE)
    assert isinstance(model3d.data, tuple)
    assert len(model3d.data) == 1

    model3d.data = []  # how a rebuild drops its old traces
    assert model3d.data == ()
    with pytest.raises(AttributeError):
        model3d.data.clear()


def test_model3d_trace_edits_are_observed():
    """Edits inside a trace bubble up like any other style change, so tools
    holding an observer on obj.style see them."""
    cuboid = make_cuboid()
    events = []
    cuboid.style.observe(lambda path, value: events.append((path, value)))

    cuboid.style.model3d.add_trace(TRACE)
    cuboid.style.model3d.data[0].show = False
    cuboid.style.set("model3d.data.0.scale", 2)

    assert events[1:] == [("model3d.data.0.show", False), ("model3d.data.0.scale", 2)]
    assert cuboid.style.get("model3d.data.0.scale") == 2


def test_show_kwargs_do_not_take_over_a_trace():
    """A trace passed through show kwargs is resolved against a throwaway
    style, which must not take it out of the tree it belongs to."""
    cuboid = make_cuboid()
    cuboid.style.model3d.add_trace(TRACE)
    trace = cuboid.style.model3d.data[0]
    events = []
    cuboid.style.observe(lambda path, _: events.append(path))

    magpy.show(cuboid, style_model3d_data=[trace], backend="plotly", return_fig=True)

    assert trace._parent is cuboid.style.model3d
    trace.scale = 2
    assert events == ["model3d.data.0.scale"]


def test_trace_deepcopy_is_detached():
    """The documented way to derive a trace from another one - deepcopy, edit,
    add - must not reach back into the style tree the original belongs to."""
    cuboid = make_cuboid()
    cuboid.style.model3d.add_trace(TRACE)
    events = []
    cuboid.style.observe(lambda path, _: events.append(path))

    derived = copy.deepcopy(cuboid.style.model3d.data[0])
    derived.scale = 3
    derived.kwargs["x"] = [9]

    assert events == []
    assert cuboid.style.model3d.data[0].scale == 1
    assert cuboid.style.model3d.data[0].kwargs["x"] != [9]

    cuboid.style.model3d.add_trace(derived)
    assert len(cuboid.style.model3d.data) == 2


def test_show_kwargs_preserve_trace_values():
    """Detaching caller-supplied traces must hand the values over unchanged -
    a tuple that arrived as a tuple is still one when it is validated."""
    trace = {"backend": "matplotlib", "constructor": "plot", "args": ((0, 1),) * 3}
    fig = magpy.show(
        make_cuboid(), style_model3d_data=[trace], backend="matplotlib", return_fig=True
    )
    assert fig is not None


def test_style_schemas_are_json():
    """schema() is the GUI contract: every style class, and the defaults tree,
    must survive json.dumps - including the trace fields reached through
    `model3d.data`, one of which defaults to a callable."""
    for style_class in (MagnetStyle, SensorStyle, CurrentStyle):
        json.dumps(style_class.schema())
    json.dumps(magpy.defaults.schema())

    trace = MagnetStyle.schema()["properties"]["model3d"]["properties"]["data"]
    assert trace["type"] == ["array", "null"]
    assert "default" not in trace["items"]["properties"]["updatefunc"]


def test_default_traces_keep_their_owner_through_resolution():
    """Resolution hands out copies of a defaults-layer trace, so the layer
    keeps its own - and an edit to it invalidates the resolved-style cache."""
    base_model3d = magpy.defaults.display.style.base.model3d
    base_model3d.data = [TRACE]
    default_trace = base_model3d.data[0]
    cuboid = make_cuboid()

    resolved = get_style(cuboid, default_settings)
    assert len(resolved.model3d.data) == 1
    assert resolved.model3d.data[0] is not default_trace
    assert default_trace._parent is base_model3d

    default_trace.scale = 5  # after the cache was primed by the resolution above
    assert get_style(cuboid, default_settings).model3d.data[0].scale == 5


# ------------------------------------------------------------------
# end-to-end: resolved styles must reach the rendered figure
# ------------------------------------------------------------------

SHOW_KW = {"backend": "plotly", "return_fig": True, "style_magnetization_show": False}


def test_e2e_show_kwarg_beats_object_style():
    """show() style kwarg wins over obj.style in the rendered trace."""
    cuboid = make_cuboid()
    cuboid.style.color = "blue"
    fig = magpy.show(cuboid, **SHOW_KW, style_color="red")
    assert fig.data[0].color == "red"


def test_e2e_object_style_reaches_figure():
    """obj.style values survive into the rendered trace; unset color falls
    back to the first colorsequence entry."""
    cuboid = make_cuboid()
    cuboid.style.opacity = 0.3
    cuboid.style.label = "MyMag"
    fig = magpy.show(cuboid, **SHOW_KW)
    assert fig.data[0].opacity == 0.3
    assert fig.data[0].name.startswith("MyMag")
    first_cycle_color = magpy.defaults.display.colorsequence[0].lower()
    assert fig.data[0].color == first_cycle_color


def test_e2e_mutated_base_default_reaches_figure():
    """A mutated base default color reaches the trace instead of the
    colorsequence fallback."""
    magpy.defaults.display.style.base.color = "black"
    fig = magpy.show(make_cuboid(), **SHOW_KW)
    assert fig.data[0].color == "black"


def test_show_preserves_style_instance_and_observers():
    """show() must not swap out the object's style instance or drop change
    observers bound to it - GUIs and other tools hold references across
    renders. (Rendering temporarily replaces the style internally, but the
    original instance is restored as-is.)"""
    cuboid = make_cuboid()
    cuboid.style.magnetization.mode = "arrow"
    style_before = cuboid.style
    events = []
    cuboid.style.observe(lambda path, value: events.append((path, value)))

    magpy.show(cuboid, **SHOW_KW)

    assert cuboid.style is style_before  # same instance, not a restored copy
    assert cuboid.style.magnetization.mode == "arrow"  # user value intact
    # the observer bound before the render still fires afterwards
    cuboid.style.opacity = 0.5
    assert events == [("opacity", 0.5)]

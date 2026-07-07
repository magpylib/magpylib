"""Characterization tests for display style resolution.

Pins the precedence chain implemented by ``magpylib._src.style.get_style``:

    show() kwargs  >  obj.style  >  family defaults  >  base defaults  >  hardcoded

where family/base defaults live on ``magpy.defaults.display.style`` and an
explicitly set ``None`` on the object defers to the next layer down. Any
refactor of the style internals must keep this suite green.
"""

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

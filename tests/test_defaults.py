import sys

import pytest

import magpylib as magpy
from magpylib._src.defaults.defaults_classes import DefaultSettings
from magpylib._src.defaults.defaults_utility import (
    ALLOWED_LINESTYLES,
    ALLOWED_SYMBOLS,
    SUPPORTED_PLOTTING_BACKENDS,
    get_registered_backends,
)
from magpylib._src.display.api import DisplayBackend
from magpylib._src.display.backend_registry import RegisteredBackend
from magpylib._src.style import DisplayStyle

bad_inputs = {
    "display_autosizefactor": (0,),  # float>0
    "display_animation_maxfps": (0,),  # int>0
    "display_animation_fps": (0,),  # int>0
    "display_animation_time": (0,),  # int>0
    "display_animation_maxframes": (0,),  # int>0
    "display_animation_slider": ("notbool"),  # bool
    "display_animation_output": ("filename.badext", "badext"),  # bool
    "display_backend": ("plotty",),  # str typo
    "display_colorsequence": (["#2E91E5", "wrongcolor"], 123),  # iterable of colors
    "display_style_base_path_line_width": (-1,),  # float>=0
    "display_style_base_path_line_style": ("wrongstyle",),
    "display_style_base_path_line_color": ("wrongcolor",),  # color
    "display_style_base_path_marker_size": (-1,),  # float>=0
    "display_style_base_path_marker_symbol": ("wrongsymbol",),
    "display_style_base_path_marker_color": ("wrongcolor",),  # color
    "display_style_base_path_show": ("notbool", 1),  # bool
    "display_style_base_path_frames": (True, False, ["1"], "1"),  # int or iterable
    "display_style_base_path_numbering": ("notbool",),  # bool
    "display_style_base_description_show": ("notbool",),  # bool
    "display_style_base_description_text": (
        False,
    ),  # DOES NOT RAISE, transforms everything into str
    "display_style_base_opacity": (-1,),  # 0<=float<=1
    "display_style_base_model3d_showdefault": ("notbool",),
    "display_style_base_color": ("wrongcolor",),  # color
    "display_style_magnet_magnetization_show": ("notbool",),
    "display_style_magnet_magnetization_arrow_size": (-1,),  # float>=0
    "display_style_magnet_magnetization_color_north": ("wrongcolor",),
    "display_style_magnet_magnetization_color_middle": ("wrongcolor",),
    "display_style_magnet_magnetization_color_south": ("wrongcolor",),
    "display_style_magnet_magnetization_color_transition": (-0.2,),  # 0<=float<=1
    "display_style_magnet_magnetization_color_mode": (
        "wrongmode",
    ),  # bicolor, tricolor, tricycle
    "display_style_magnet_magnetization_mode": (
        "wrongmode",
    ),  # 'auto', 'arrow', 'color', 'arrow+color'
    "display_style_current_arrow_show": ("notbool",),
    "display_style_current_arrow_size": (-1,),  # float>=0
    "display_style_current_arrow_width": (-1,),  # float>=0
    "display_style_sensor_size": (-1,),  # float>=0
    "display_style_sensor_arrows_x_color": ("wrongcolor",),
    "display_style_sensor_arrows_x_show": ("notbool",),
    "display_style_sensor_arrows_y_color": ("wrongcolor",),
    "display_style_sensor_arrows_y_show": ("notbool",),
    "display_style_sensor_arrows_z_color": ("wrongcolor",),
    "display_style_sensor_arrows_z_show": ("notbool",),
    "display_style_sensor_pixel_size": (-1,),  # float>=0
    "display_style_sensor_pixel_color": ("notbool",),
    "display_style_sensor_pixel_symbol": ("wrongsymbol",),
    "display_style_dipole_size": (-1,),  # float>=0
    "display_style_dipole_pivot": ("wrongpivot",),  # middle, tail, tip
    "display_style_triangle_orientation_show": ("notbool",),
    "display_style_triangle_orientation_size": (-1,),
    "display_style_triangle_orientation_color": ("wrongcolor",),
    "display_style_triangle_orientation_offset": ("-1",),  # float, int
    "display_style_triangle_orientation_symbol": ("arrow0d"),  # "cone", "arrow3d"
    "display_style_triangularmesh_mesh_disconnected_colorsequence": (1,),
    "display_style_markers_marker_size": (-1,),  # float>=0
    "display_style_markers_marker_color": ("wrongcolor",),
    "display_style_markers_marker_symbol": ("wrongsymbol",),
}


def get_bad_test_data():
    """create parametrized bad style test data"""
    # all property validators raise ValueError
    return [
        (k, v, pytest.raises(ValueError, match=r".*"))
        for k, tup in bad_inputs.items()
        for v in tup
    ]


@pytest.mark.parametrize(
    ("key", "value", "expected_errortype"),
    get_bad_test_data(),
)
def test_defaults_bad_inputs(key, value, expected_errortype):
    """testing defaults setting on bad inputs"""
    c = DefaultSettings().reset()
    with expected_errortype:
        c.update(**{key: value})


# dict of good input.
# This is just for check. dict keys should not be tuples in general, but the test will iterate
# over the values for each key
good_inputs = {
    "display_autosizefactor": (1,),  # float>0
    "display_animation_maxfps": (10,),  # int>0
    "display_animation_fps": (10,),  # int>0
    "display_animation_time": (10,),  # int>0
    "display_animation_maxframes": (200,),  # int>0
    "display_animation_slider": (True, False),  # bool
    "display_animation_output": ("filename.mp4", "gif"),  # bool
    "display_backend": ["auto", *SUPPORTED_PLOTTING_BACKENDS],  # str typo
    "display_colorsequence": (
        ("#2e91e5", "#0d2a63"),
        ("blue", "red"),
    ),  # ]),  # iterable of colors
    "display_style_base_path_line_width": (0, 1),  # float>=0
    "display_style_base_path_line_style": ALLOWED_LINESTYLES,
    "display_style_base_path_line_color": ("blue", "#2E91E5"),  # color
    "display_style_base_path_marker_size": (0, 1),  # float>=0
    "display_style_base_path_marker_symbol": ALLOWED_SYMBOLS,
    "display_style_base_path_marker_color": ("blue", "#2E91E5"),  # color
    "display_style_base_path_show": (True, False),  # bool
    "display_style_base_path_frames": (-1, (1, 3)),  # int or iterable
    "display_style_base_path_numbering": (True, False),  # bool
    "display_style_base_description_show": (True, False),  # bool
    "display_style_base_description_text": ("a string",),  # string
    "display_style_base_opacity": (0, 0.5, 1),  # 0<=float<=1
    "display_style_base_model3d_showdefault": (True, False),
    "display_style_base_color": ("blue", "#2E91E5"),  # color
    "display_style_magnet_magnetization_show": (True, False),
    "display_style_magnet_magnetization_size": (0, 1),  # float>=0
    "display_style_magnet_magnetization_color_north": ("blue", "#2E91E5"),
    "display_style_magnet_magnetization_color_middle": ("blue", "#2E91E5"),
    "display_style_magnet_magnetization_color_south": ("blue", "#2E91E5"),
    "display_style_magnet_magnetization_color_transition": (0, 0.5, 1),  # 0<=float<=1
    "display_style_magnet_magnetization_color_mode": (
        "bicolor",
        "tricolor",
        "tricycle",
    ),
    "display_style_magnet_magnetization_mode": (
        "auto",
        "arrow",
        "color",
        "arrow+color",
        "color+arrow",
    ),
    "display_style_current_arrow_show": (True, False),
    "display_style_current_arrow_size": (0, 1),  # float>=0
    "display_style_current_arrow_width": (0, 1),  # float>=0
    "display_style_sensor_size": (0, 1),  # float>=0
    "display_style_sensor_arrows_x_color": ("magenta",),
    "display_style_sensor_arrows_x_show": (True, False),
    "display_style_sensor_arrows_y_color": ("yellow",),
    "display_style_sensor_arrows_y_show": (True, False),
    "display_style_sensor_arrows_z_color": ("cyan",),
    "display_style_sensor_arrows_z_show": (True, False),
    "display_style_sensor_pixel_size": (0, 1),  # float>=0
    "display_style_sensor_pixel_color": ("blue", "#2E91E5"),
    "display_style_sensor_pixel_symbol": ALLOWED_SYMBOLS,
    "display_style_dipole_size": (0, 1),  # float>=0
    "display_style_dipole_pivot": (
        "middle",
        "tail",
        "tip",
    ),  # pivot middle, tail, tip
    "display_style_triangle_orientation_show": (True, False),
    "display_style_triangle_orientation_size": (0, 1),  # float>=0
    "display_style_triangle_orientation_color": ("yellow",),
    "display_style_triangle_orientation_offset": (-1, 0.5, 2),  # float, int
    "display_style_triangle_orientation_symbol": ("cone", "arrow3d"),
    "display_style_markers_marker_size": (0, 1),  # float>=0
    "display_style_markers_marker_color": ("blue", "#2E91E5"),
    "display_style_markers_marker_symbol": ALLOWED_SYMBOLS,
}


def get_good_test_data():
    """create parametrized good style test data"""
    good_test_data = []
    for key, tup in good_inputs.items():
        for value in tup:
            expected = value
            if "color" in key and isinstance(value, str):
                expected = value.lower()  # hex color gets lowered
            good_test_data.append((key, value, expected))
    return good_test_data


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    get_good_test_data(),
)
def test_defaults_good_inputs(key, value, expected):
    """testing defaults setting on bad inputs"""
    c = DefaultSettings()
    c.update(**{key: value})
    v0 = c
    for v in key.split("_"):
        v0 = getattr(v0, v)
    assert v0 == expected, f"Input {key} should be {expected}; instead received {v0}."


@pytest.mark.parametrize(
    "style_class",
    [
        "base",
        "base_model3d",
        "base_path",
        "base_path_line",
        "base_path_marker",
        "current",
        "current_arrow",
        "dipole",
        "magnet",
        "magnet_magnetization",
        "magnet_magnetization_color",
        "markers",
        "markers_marker",
        "sensor",
        "sensor_pixel",
    ],
)
def test_bad_style_classes(style_class):
    """testing properties which take classes as properties"""
    c = DisplayStyle().reset()
    with pytest.raises(
        TypeError,
        match=(r"The"),
    ):
        c.update(**{style_class: "bad class"})


def test_bad_default_classes():
    """testing properties which take classes as properties"""
    with pytest.raises(
        TypeError,
        match=r"The display property of DefaultSettings must be",
    ):
        magpy.defaults.display = "wrong input"
    with pytest.raises(
        TypeError,
        match=r"The animation property of Display must be",
    ):
        magpy.defaults.display.animation = "wrong input"
    with pytest.raises(
        TypeError,
        match=r"The style property of Display must be",
    ):
        magpy.defaults.display.style = "wrong input"


@pytest.fixture
def dummy_backend():
    """Register a do-nothing display backend for the duration of a test."""
    name = "dummy"
    RegisteredBackend(
        name=name,
        show_func=lambda data, **_kwargs: data,
        supports_animation=False,
        supports_subplots=False,
        supports_colorgradient=False,
        supports_animation_output=False,
    )
    default_backend = magpy.defaults.display.backend
    try:
        yield name
    finally:
        magpy.defaults.display.backend = default_backend
        DisplayBackend.backends.pop(name, None)


def test_backend_fields_follow_the_registry(dummy_backend):
    """Backend fields accept backends registered after import time.

    The allowed set is resolved from `DisplayBackend.backends` on every
    use, not frozen when the property tree is declared. Should the registry
    ever move, `get_registered_backends` must follow it or this fails.
    """
    assert dummy_backend in get_registered_backends()

    magpy.defaults.display.backend = dummy_backend
    assert magpy.defaults.display.backend == dummy_backend

    obj = magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))
    obj.style.model3d.add_trace(backend=dummy_backend, constructor="Mesh3d")
    assert obj.style.model3d.data[0].backend == dummy_backend

    # the generated schema reports the backends registered at generation time
    enum = magpy.defaults.display.schema()["properties"]["backend"]["enum"]
    assert dummy_backend in enum


def test_registered_backends_falls_back_to_builtins(monkeypatch):
    """Without the registry imported, the built-in backends are reported.

    `DefaultSettings()` validates `display.backend` while `magpylib` is still
    importing, before the registry module exists -- importing it from the
    defaults would be circular.
    """
    monkeypatch.delitem(sys.modules, "magpylib._src.display.display")
    assert get_registered_backends() == SUPPORTED_PLOTTING_BACKENDS
    assert DefaultSettings().display.backend == "auto"


def test_backend_fields_still_reject_unknown_names():
    """Opening the choice set to the registry does not accept anything."""
    with pytest.raises(ValueError, match="one of"):
        magpy.defaults.display.backend = "plotty"

    obj = magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))
    with pytest.raises(ValueError, match="one of"):
        obj.style.model3d.add_trace(backend="plotty", constructor="Mesh3d")


def test_bad_deferred_style():
    """test error raise on deferred style attribution"""
    c = magpy.magnet.Cuboid(style_badstyle="ASDF")
    with pytest.raises(
        AttributeError,
        match=r".* has been initialized with some invalid style arguments.*",
    ):
        magpy.show(c)  # style property gets called, style kwargs applied

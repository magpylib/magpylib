import pytest

import magpylib as magpy
from magpylib._src.defaults.defaults_classes import DefaultSettings
from magpylib._src.defaults.defaults_utility import ALLOWED_LINESTYLES
from magpylib._src.defaults.defaults_utility import ALLOWED_SYMBOLS
from magpylib._src.defaults.defaults_utility import SUPPORTED_PLOTTING_BACKENDS
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
    # pylint: disable=possibly-used-before-assignment
    bad_test_data = []
    for k, tup in bad_inputs.items():
        for v in tup:
            if "description_text" not in k:
                if "color" in k and "transition" not in k and "mode" not in k:
                    # color attributes use a the color validator, which raises a ValueError
                    errortype = ValueError
                else:
                    # all other parameters raise AssertionError
                    errortype = AssertionError
            bad_test_data.append((k, v, pytest.raises(errortype)))
    return bad_test_data


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
    "display_style_triangle_orientation_size": (True, False),
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
    assert v0 == expected, f"{key} should be {expected}, but received {v0} instead"


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
    with pytest.raises(ValueError):
        c.update(**{style_class: "bad class"})


def test_bad_default_classes():
    """testing properties which take classes as properties"""
    with pytest.raises(ValueError):
        magpy.defaults.display = "wrong input"
    with pytest.raises(ValueError):
        magpy.defaults.display.animation = "wrong input"
    with pytest.raises(ValueError):
        magpy.defaults.display.style = "wrong input"


def test_bad_deferred_style():
    """test error raise on deferred style attribution"""
    c = magpy.magnet.Cuboid(style_badstyle="ASDF")
    with pytest.raises(
        AttributeError,
        match=r".* has been initialized with some invalid style arguments.*",
    ):
        magpy.show(c)  # style property gets called, style kwargs applied

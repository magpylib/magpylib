import pytest
import magpylib as magpy
from magpylib._lib.config import DefaultConfig
from magpylib._lib.style import DisplayStyle
from magpylib._lib.default_utils import (
    LINESTYLES_MATPLOTLIB_TO_PLOTLY,
    SYMBOLS_MATPLOTLIB_TO_PLOTLY,
)

bad_inputs = {
    "checkinputs": (-1,),  # bool
    "edgesize": (0,),  # float>0
    "itercylinder": (0.1,),  # int>0
    "display_autosizefactor": (0,),  # float>0
    "display_animation_maxfps": (0,),  # int>0
    "display_animation_maxframes": (0,),  # int>0
    "display_animation_slider":  ('notbool'),  # bool
    "display_backend": ("plotty",),  # str typo
    "display_colorsequence": (["#2E91E5", "wrongcolor"],),  # iterable of colors
    "display_style_base_path_line_width": (-1,),  # float>=0
    "display_style_base_path_line_style": ("wrongstyle",),
    "display_style_base_path_line_color": ("wrongcolor",),  # color
    "display_style_base_path_marker_size": (-1,),  # float>=0
    "display_style_base_path_marker_symbol": ("wrongsymbol",),
    "display_style_base_path_marker_color": ("wrongcolor",),  # color
    "display_style_base_description_show": ("notbool",),  # bool
    "display_style_base_description_text": (
        False,
    ),  # DOES NOT RAISE, transforms everything into str
    "display_style_base_opacity": (-1,),  # 0<=float<=1
    "display_style_base_mesh3d_show": ("notbool",),
    "display_style_base_mesh3d_data": (dict(x=1, y=1),),  # dict with x,y,z,i,j,k
    "display_style_base_color": ("wrongcolor",),  # color
    "display_style_magnet_magnetization_show": ("notbool",),
    "display_style_magnet_magnetization_size": (-1,),  # float>=0
    "display_style_magnet_magnetization_color_north": ("wrongcolor",),
    "display_style_magnet_magnetization_color_middle": ("wrongcolor",),
    "display_style_magnet_magnetization_color_south": ("wrongcolor",),
    "display_style_magnet_magnetization_color_transition": (-0.2,),  # 0<=float<=1
    "display_style_magnet_magnetization_color_mode": (
        "wrongmode",
    ),  # bicolor, tricolor, tricycle
    "display_style_current_arrow_show": ("notbool",),
    "display_style_current_arrow_size": (-1,),  # float>=0
    "display_style_current_arrow_width": (-1,),  # float>=0
    "display_style_sensor_size": (-1,),  # float>=0
    "display_style_sensor_pixel_size": (-1,),  # float>=0
    "display_style_sensor_pixel_color": ("notbool",),
    "display_style_sensor_pixel_symbol": ("wrongsymbol",),
    "display_style_dipole_size": (-1,),  # float>=0
    "display_style_dipole_pivot": ("wrongpivot",),  # middle, tail, tip
    "display_style_markers_marker_size": (-1,),  # float>=0
    "display_style_markers_marker_color": ("wrongcolor",),
    "display_style_markers_marker_symbol": ("wrongsymbol",),
}

# dict of good input.
# This is just for check. dict keys should not be tuples in general, but the test will iterate
# over the values for each key
good_inputs = {
    "checkinputs": (True, False),  # bool
    "edgesize": (1e-9, 2),  # float>0
    "itercylinder": (10,),  # int>0
    "display_autosizefactor": (1,),  # float>0
    "display_animation_maxfps": (10,),  # int>0
    "display_animation_maxframes": (200,),  # int>0
    "display_animation_slider":  (True, False),  # bool
    "display_backend": ("matplotlib", "plotly"),  # str typo
    "display_colorsequence": (
        ["#2E91E5", "#0D2A63"],
        ["blue", "red"],
    ),  # ]),  # iterable of colors
    "display_style_base_path_line_width": (0, 1),  # float>=0
    "display_style_base_path_line_style": LINESTYLES_MATPLOTLIB_TO_PLOTLY.keys(),
    "display_style_base_path_line_color": ("blue", "#2E91E5"),  # color
    "display_style_base_path_marker_size": (0, 1),  # float>=0
    "display_style_base_path_marker_symbol": SYMBOLS_MATPLOTLIB_TO_PLOTLY.keys(),
    "display_style_base_path_marker_color": ("blue", "#2E91E5"),  # color
    "display_style_base_description_show": (True, False),  # bool
    "display_style_base_description_text": (
        True,
        object,
        "a string",
    ),  # DOES NOT RAISE, transforms everything into str
    "display_style_base_opacity": (0, 0.5, 1),  # 0<=float<=1
    "display_style_base_mesh3d_show": (True, False),
    "display_style_base_mesh3d_data": (
        dict(x=[1], y=[1], z=[1], i=[1], j=[1], k=[1]),
    ),  # dict with x,y,z,i,j,k
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
    "display_style_current_arrow_show": (True, False),
    "display_style_current_arrow_size": (0, 1),  # float>=0
    "display_style_current_arrow_width": (0, 1),  # float>=0
    "display_style_sensor_size": (0, 1),  # float>=0
    "display_style_sensor_pixel_size": (0, 1),  # float>=0
    "display_style_sensor_pixel_color": ("blue", "#2E91E5"),
    "display_style_sensor_pixel_symbol": SYMBOLS_MATPLOTLIB_TO_PLOTLY.keys(),
    "display_style_dipole_size": (0, 1),  # float>=0
    "display_style_dipole_pivot": (
        "middle",
        "tail",
        "tip",
    ),  # pivot middle, tail, tip
    "display_style_markers_marker_size": (0, 1),  # float>=0
    "display_style_markers_marker_color": ("blue", "#2E91E5"),
    "display_style_markers_marker_symbol": SYMBOLS_MATPLOTLIB_TO_PLOTLY.keys(),
}


def test_defaults():
    """test setting and resetting the config"""
    magpy.defaults.itercylinder = 15
    assert magpy.defaults.itercylinder == 15, "setting config failed"
    magpy.defaults.reset()
    assert magpy.defaults.itercylinder == 50, "resetting config failed"


def test_defaults_bad_inputs():
    """testing defaults setting on bad inputs"""
    c = DefaultConfig().reset()
    for k, tup in bad_inputs.items():
        for v in tup:
            if "description_text" not in k:
                if "color" in k and "transition" not in k and "mode" not in k:
                    # color attributes use a the color validator, which raises a ValueError
                    errortype = ValueError
                else:
                    # all other parameters raise AssertionError
                    errortype = AssertionError
                with pytest.raises(errortype):
                    c.update(**{k: v})


def test_defaults_good_inputs():
    """testing defaults setting on bad inputs"""
    c = DefaultConfig()
    for k, tup in good_inputs.items():
        for v1 in tup:
            c.update(**{k: v1})
            v0 = c
            for v in k.split("_"):
                v0 = getattr(v0, v)
            if "color" in k and isinstance(v1, str):
                v1 = v1.lower()  # hex color gets lowered
            elif "description_text" in k:
                # for a desc text, any object is valid and is transformed into a string
                v1 = str(v1)
            assert v0 == v1, f"{k} should be {v1}, but received {v0} instead"


def test_bad_input_classes():
    """testing properties which take classes as properties"""
    with pytest.raises(ValueError):
        magpy.defaults.display = "wrong input"
    with pytest.raises(ValueError):
        magpy.defaults.display.animation = "wrong input"
    with pytest.raises(ValueError):
        magpy.defaults.display.style = "wrong input"
    c = DisplayStyle().reset()
    style_classes = {
        "base",
        "base_description",
        "base_mesh3d",
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
    }
    for s in style_classes:
        with pytest.raises(ValueError):
            c.update(**{s: "bad class"})

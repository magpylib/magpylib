import pytest
import magpylib as magpy
from magpylib._lib.config import DefaultConfig
from magpylib._lib.default_utils import (
    LINESTYLES_MATPLOTLIB_TO_PLOTLY,
    SYMBOLS_MATPLOTLIB_TO_PLOTLY,
)

bad_inputs = {
    "checkinputs": -1,  # bool
    "edgesize": 0,  # float>0
    "itercylinder": 0.1,  # int>0
    "display_autosizefactor": 0,  # float>0
    "display_animation_maxfps": 0,  # int>0
    "display_animation_maxframes": 0,  # int>0
    "display_backend": "plotty",  # str typo
    "display_colorsequence": ["#2E91E5", "wrongcolor"],  # iterable of colors
    "display_opacity": 2,  # 0<=float<=1
    "display_styles_base_path_line_width": -1,  # float>=0
    "display_styles_base_path_line_style": "wrongstyle",
    "display_styles_base_path_line_color": "wrongcolor",  # color
    "display_styles_base_path_marker_size": -1,  # float>=0
    "display_styles_base_path_marker_symbol": "wrongsymbol",
    "display_styles_base_path_marker_color": "wrongcolor",  # color
    "display_styles_base_description_show": "notbool",  # bool
    "display_styles_base_description_text": False,  # DOES NOT RAISE, transforms everything into str
    "display_styles_base_opacity": -1,  # 0<=float<=1
    "display_styles_base_mesh3d_show": "notbool",
    "display_styles_base_mesh3d_data": dict(x=1, y=1),  # dict with x,y,z,i,j,k
    "display_styles_base_color": "wrongcolor",  # color
    "display_styles_magnets_magnetization_show": "notbool",
    "display_styles_magnets_magnetization_size": -1,  # float>=0
    "display_styles_magnets_magnetization_color_north": "#wrongcolor",
    "display_styles_magnets_magnetization_color_middle": "#wrongcolor",
    "display_styles_magnets_magnetization_color_south": "#wrongcolor",
    "display_styles_magnets_magnetization_color_transition": -0.2,  # 0<=float<=1
    "display_styles_currents_current_show": "notbool",
    "display_styles_currents_current_size": -1,  # float>=0
    "display_styles_currents_current_width": -1,  # float>=0
    "display_styles_sensors_size": -1,  # float>=0
    "display_styles_sensors_pixel_size": -1,  # float>=0
    "display_styles_sensors_pixel_color": "notbool",
    "display_styles_sensors_pixel_symbol": "wrongsymbol",
    "display_styles_dipoles_size": -1,  # float>=0
    "display_styles_dipoles_pivot": "wrongpivot",  # pivot middle, tail, tip
    "display_styles_markers_marker_size": -1,  # float>=0
    "display_styles_markers_marker_color": "wrongcolor",
    "display_styles_markers_marker_symbol": "wrongsymbol",
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
    "display_backend": ("matplotlib", "plotly"),  # str typo
    "display_colorsequence": (
        ["#2E91E5", "#0D2A63"],
        ["blue", "red"],
    ),  # ]),  # iterable of colors
    "display_opacity": (0, 0.5, 1),  # 0<=float<=1
    "display_styles_base_path_line_width": (0, 1),  # float>=0
    "display_styles_base_path_line_style": LINESTYLES_MATPLOTLIB_TO_PLOTLY.keys(),
    "display_styles_base_path_line_color": ("blue", "#2E91E5"),  # color
    "display_styles_base_path_marker_size": (0,1),  # float>=0
    "display_styles_base_path_marker_symbol": SYMBOLS_MATPLOTLIB_TO_PLOTLY.keys(),
    "display_styles_base_path_marker_color": ("blue", "#2E91E5"),  # color
    "display_styles_base_description_show": (True, False),  # bool
    "display_styles_base_description_text": (
        True,
        object,
        "a string",
    ),  # DOES NOT RAISE, transforms everything into str
    "display_styles_base_opacity": (0, 0.5, 1),  # 0<=float<=1
    "display_styles_base_mesh3d_show": (True, False),
    "display_styles_base_mesh3d_data": (
        dict(x=[1], y=[1], z=[1], i=[1], j=[1], k=[1]),
    ),  # dict with x,y,z,i,j,k
    "display_styles_base_color": ("blue", "#2E91E5"),  # color
    "display_styles_magnets_magnetization_show": (True, False),
    "display_styles_magnets_magnetization_size": (0, 1),  # float>=0
    "display_styles_magnets_magnetization_color_north": ("blue", "#2E91E5"),
    "display_styles_magnets_magnetization_color_middle": ("blue", "#2E91E5"),
    "display_styles_magnets_magnetization_color_south": ("blue", "#2E91E5"),
    "display_styles_magnets_magnetization_color_transition": (0, 0.5, 1),  # 0<=float<=1
    "display_styles_currents_current_show": (True, False),
    "display_styles_currents_current_size": (0, 1),  # float>=0
    "display_styles_currents_current_width": (0, 1),  # float>=0
    "display_styles_sensors_size": (0, 1),  # float>=0
    "display_styles_sensors_pixel_size": (0, 1),  # float>=0
    "display_styles_sensors_pixel_color": ("blue", "#2E91E5"),
    "display_styles_sensors_pixel_symbol": SYMBOLS_MATPLOTLIB_TO_PLOTLY.keys(),
    "display_styles_dipoles_size": (0, 1),  # float>=0
    "display_styles_dipoles_pivot": (
        "middle",
        "tail",
        "tip",
    ),  # pivot middle, tail, tip
    "display_styles_markers_marker_size": (0, 1),  # float>=0
    "display_styles_markers_marker_color": ("blue", "#2E91E5"),
    "display_styles_markers_marker_symbol": SYMBOLS_MATPLOTLIB_TO_PLOTLY.keys(),
}


def test_defaults():
    """test setting and resetting the config"""
    magpy.defaults.itercylinder = 15
    assert magpy.defaults.itercylinder == 15, "setting config failed"
    magpy.defaults.reset()
    assert magpy.defaults.itercylinder == 50, "resetting config failed"


def test_defaults_bad_inputs():
    """testing defaults setting on bad inputs"""
    c = DefaultConfig()
    for k, v in bad_inputs.items():
        if "description_text" not in k:
            if "color" in k and "transition" not in k:
                #color attributes use a the color validator, which raises a ValueError
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
            for v in k.split('_'):
                v0 = getattr(v0, v)
            if 'color' in k and isinstance(v1,str):
                v1 = v1.lower() # hex color gets lowered
            elif 'description_text' in k:
                v1 = str(v1) # for a desc text, any object is valid and is transformed into a string
            assert v0 == v1, f"{k} should be {v1}, but received {v0} instead"

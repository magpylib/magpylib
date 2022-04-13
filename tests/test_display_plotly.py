import numpy as np
import plotly.graph_objects as go
import pytest

import magpylib as magpy
from magpylib._src.display.plotly.plotly_display import get_plotly_traces
from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib.magnet import Cuboid
from magpylib.magnet import Cylinder
from magpylib.magnet import CylinderSegment
from magpylib.magnet import Sphere

# pylint: disable=assignment-from-no-return


def test_Cylinder_display():
    """testing display"""
    magpy.defaults.display.backend = "plotly"
    fig = go.Figure()
    src = Cylinder((1, 2, 3), (1, 2))
    x = src.show(canvas=fig, style_path_frames=15)
    assert x is None, "path should revert to True"

    src.move(np.linspace((0.4, 0.4, 0.4), (12.4, 12.4, 12.4), 33), start=-1)
    x = src.show(
        canvas=fig,
        style_path_show=False,
        style_magnetization_show=True,
        style_magnetization_color_mode="tricycle",
    )
    assert x is None, "display test fail"


def test_CylinderSegment_display():
    """testing display"""
    magpy.defaults.display.backend = "plotly"
    fig = go.Figure()
    src = CylinderSegment((1, 2, 3), (2, 4, 5, 30, 40))
    x = src.show(canvas=fig, style_path_frames=15)
    assert x is None, "path should revert to True"

    src.move(np.linspace((0.4, 0.4, 0.4), (12.4, 12.4, 12.4), 33), start=-1)
    x = src.show(
        canvas=fig,
        style_path_show=False,
        style_magnetization_show=True,
        style_magnetization_color_mode="bicolor",
    )
    assert x is None, "display test fail"


def test_Sphere_display():
    """testing display"""
    magpy.defaults.display.backend = "plotly"
    fig = go.Figure()
    src = Sphere((1, 2, 3), 2)
    x = src.show(canvas=fig, style_path_frames=15)
    assert x is None, "path should revert to True"

    src.move(np.linspace((0.4, 0.4, 0.4), (8, 8, 8), 33), start=-1)
    x = src.show(canvas=fig, style_path_show=False, style_magnetization_show=True)
    assert x is None, "display test fail"


def test_Cuboid_display():
    """testing display"""
    magpy.defaults.display.backend = "plotly"
    src = Cuboid((1, 2, 3), (1, 2, 3))
    src.move(np.linspace((0.1, 0.1, 0.1), (2, 2, 2), 20), start=-1)
    x = src.show(style_path_frames=5, style_magnetization_show=True, renderer="json")
    assert x is None, "display test fail"

    fig = go.Figure()
    x = src.show(canvas=fig, style_path_show=False, style_magnetization_show=True)
    assert x is None, "display test fail"


def test_Sensor_display():
    """testing display"""
    magpy.defaults.display.backend = "plotly"
    fig = go.Figure()
    sens_nopix = magpy.Sensor()
    x = sens_nopix.show(canvas=fig, style_description_text="mysensor")
    assert x is None, "display test fail"
    sens = magpy.Sensor(pixel=[(1, 2, 3), (2, 3, 4)])
    sens.move(np.linspace((0.4, 0.4, 0.4), (12.4, 12.4, 12.4), 33), start=-1)
    sens.style.arrows.z.color = "magenta"
    sens.style.arrows.z.show = False
    x = sens.show(canvas=fig, markers=[(100, 100, 100)], style_path_frames=15)
    assert x is None, "display test fail"
    x = sens.show(canvas=fig, markers=[(100, 100, 100)], style_path_show=False)
    assert x is None, "display test fail"


def test_Loop_display():
    """testing display for Loop source"""
    magpy.defaults.display.backend = "plotly"
    fig = go.Figure()
    src = magpy.current.Loop(current=1, diameter=1)
    x = src.show(canvas=fig)
    assert x is None, "display test fail"

    src.rotate_from_angax([5] * 35, "x", anchor=(1, 2, 3))
    x = src.show(canvas=fig, style_path_frames=3)
    assert x is None, "display test fail"


def test_col_display():
    """testing display"""
    # pylint: disable=assignment-from-no-return
    magpy.defaults.display.backend = "plotly"
    fig = go.Figure()
    pm1 = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    pm2 = pm1.copy(position=(2, 0, 0))
    pm3 = pm1.copy(position=(4, 0, 0))
    nested_col = (pm1 + pm2 + pm3).set_children_styles(color="magenta")
    x = nested_col.show(canvas=fig)
    assert x is None, "collection display test fail"


def test_dipole_display():
    """testing display"""
    # pylint: disable=assignment-from-no-return
    magpy.defaults.display.backend = "plotly"
    fig = go.Figure()
    dip1 = magpy.misc.Dipole(moment=(1, 2, 3), position=(1, 1, 1))
    dip2 = magpy.misc.Dipole(moment=(1, 2, 3), position=(2, 2, 2))
    dip3 = magpy.misc.Dipole(moment=(1, 2, 3), position=(3, 3, 3))
    dip2.move(np.linspace((0.4, 0.4, 0.4), (2, 2, 2), 5), start=-1)
    x = dip1.show(canvas=fig, style_pivot="tail")
    assert x is None, "display test fail"
    x = dip2.show(canvas=fig, style_path_frames=2, style_pivot="tip")
    assert x is None, "display test fail"
    x = dip3.show(canvas=fig, style_path_frames=2, style_pivot="middle")
    assert x is None, "display test fail"


def test_circular_line_display():
    """testing display"""
    # pylint: disable=assignment-from-no-return
    magpy.defaults.display.backend = "plotly"
    fig = go.Figure()
    src1 = magpy.current.Loop(1, 2)
    src2 = magpy.current.Loop(1, 2)
    src1.move(np.linspace((0.4, 0.4, 0.4), (2, 2, 2), 5), start=-1)
    src3 = magpy.current.Line(1, [(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    src4 = magpy.current.Line(1, [(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    src3.move([(0.4, 0.4, 0.4)] * 5, start=-1)
    x = src1.show(canvas=fig, style_path_frames=2, style_arrow_show=False)
    assert x is None, "display test fail"
    x = src2.show(canvas=fig)
    assert x is None, "display test fail"
    x = src3.show(canvas=fig, style_arrow_show=False)
    assert x is None, "display test fail"
    x = src4.show(canvas=fig, style_path_frames=2)
    assert x is None, "display test fail"


def test_display_bad_style_kwargs():
    """test if some magic kwargs are invalid"""
    magpy.defaults.display.backend = "plotly"
    fig = go.Figure()
    with pytest.raises(ValueError):
        magpy.show(canvas=fig, markers=[(1, 2, 3)], style_bad_style_kwarg=None)


def test_draw_unsupported_obj():
    """test if a object which is not directly supported by magpylib can be plotted"""
    magpy.defaults.display.backend = "plotly"

    class UnkwnownNoPosition:
        """Dummy Class"""

    class Unkwnown1DPosition:
        """Dummy Class"""

        position = [0, 0, 0]

    class Unkwnown2DPosition:
        """Dummy Class"""

        position = [[0, 0, 0]]
        orientation = None

    with pytest.raises(AttributeError):
        get_plotly_traces(UnkwnownNoPosition())

    traces = get_plotly_traces(Unkwnown1DPosition)
    assert (
        traces[0]["type"] == "scatter3d"
    ), "make trace has failed, should be 'scatter3d'"

    traces = get_plotly_traces(Unkwnown2DPosition)
    assert (
        traces[0]["type"] == "scatter3d"
    ), "make trace has failed, should be 'scatter3d'"


def test_extra_model3d():
    """test diplay when object has an extra model object attached"""
    magpy.defaults.display.backend = "plotly"
    cuboid = Cuboid((1, 2, 3), (1, 2, 3))
    cuboid.move(np.linspace((0.4, 0.4, 0.4), (12.4, 12.4, 12.4), 33), start=-1)
    cuboid.style.model3d.showdefault = False
    cuboid.style.model3d.data = [
        {
            "backend": "plotly",
            "constructor": "Scatter3d",
            "kwargs": {
                "x": [-1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1],
                "y": [-1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
                "z": [-1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1],
                "mode": "lines",
            },
            "show": True,
        },
        {
            "backend": "plotly",
            "constructor": "Mesh3d",
            "kwargs": {
                "i": [7, 0, 0, 0, 4, 4, 2, 6, 4, 0, 3, 7],
                "j": [0, 7, 1, 2, 6, 7, 1, 2, 5, 5, 2, 2],
                "k": [3, 4, 2, 3, 5, 6, 5, 5, 0, 1, 7, 6],
                "x": [-1, -1, 1, 1, -1, -1, 1, 1],
                "y": [-1, 1, 1, -1, -1, 1, 1, -1],
                "z": [-1, -1, -1, -1, 1, 1, 1, 1],
                "facecolor": ["red"] * 12,
            },
            "show": True,
        },
    ]
    fig = go.Figure()
    x = cuboid.show(canvas=fig, style=dict(model3d_showdefault=True))
    assert x is None, "display test fail"
    cuboid.style.model3d.data[0].show = False
    x = cuboid.show(canvas=fig)
    assert x is None, "display test fail"
    coll = magpy.Collection(cuboid)
    coll.rotate_from_angax(45, "z")
    x = magpy.show(
        coll,
        canvas=fig,
        animation=True,
        style=dict(model3d_showdefault=False),
    )
    assert x is None, "display test fail"
    my_callable_kwargs = lambda: {
        "x": [-1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1],
        "y": [-1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
        "z": [-1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1],
        "mode": "lines",
    }
    cuboid.style.model3d.add_trace(
        **{
            "backend": "plotly",
            "constructor": "Scatter3d",
            "kwargs": my_callable_kwargs,
            "show": True,
        }
    )
    cuboid.style.model3d.data[0].show = False
    x = cuboid.show(
        canvas=fig,
        style_path_show=False,
        style=dict(model3d_showdefault=False),
    )
    assert x is None, "display test fail"


def test_CustomSource_display():
    """testing display"""
    fig = go.Figure()
    cs = magpy.misc.CustomSource(style=dict(color="blue"))
    x = cs.show(canvas=fig, backend="plotly")
    assert x is None, "display test fail"


def test_empty_display():
    """should not fail if nothing to display"""
    fig = go.Figure()
    x = magpy.show(canvas=fig, backend="plotly")
    assert x is None, "empty display plotly test fail"


def test_display_warnings():
    """should display some animation warnings"""
    magpy.defaults.display.backend = "plotly"
    magpy.defaults.display.animation.maxfps = 2
    magpy.defaults.display.animation.maxframes = 2
    src = Cuboid((1, 2, 3), (1, 2, 3))
    src.move(np.linspace((0.4, 0.4, 0.4), (4, 4, 4), 10), start=-1)
    fig = go.Figure()

    with pytest.warns(UserWarning):  # animation_fps to big warning
        src.show(canvas=fig, animation=5, animation_fps=3)
    with pytest.warns(UserWarning):  # max frames surpassed
        src.show(canvas=fig, animation=True, animation_time=2, animation_fps=1)
    src = Cuboid((1, 2, 3), (1, 2, 3))
    with pytest.warns(UserWarning):  # no object path detected
        src.show(canvas=fig, style_path_frames=[], animation=True)


def test_bad_animation_value():
    """should fail if animation is not a boolean or a positive number"""
    magpy.defaults.display.backend = "plotly"
    magpy.defaults.display.animation.maxfps = 2
    magpy.defaults.display.animation.maxframes = 2
    src = Cuboid((1, 2, 3), (1, 2, 3))
    src.move(np.linspace((0.4, 0.4, 0.4), (4, 4, 4), 10), start=-1)
    fig = go.Figure()

    with pytest.raises(MagpylibBadUserInput):
        src.show(canvas=fig, animation=-1)

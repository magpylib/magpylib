import numpy as np
import plotly.graph_objects as go
import pytest
import magpylib as magpy
from magpylib.magnet import Cylinder, Cuboid, Sphere, CylinderSegment
from magpylib._src.display.plotly.plotly_display import get_plotly_traces

# pylint: disable=assignment-from-no-return


def test_Cylinder_display():
    """testing display"""
    magpy.defaults.display.backend = "plotly"
    fig = go.Figure()
    src = Cylinder((1, 2, 3), (1, 2))
    x = src.display(canvas=fig, path=15)
    assert x is None, "path should revert to True"

    src.move(np.linspace((.4,.4,.4), (12.4,12.4,12.4), 33), start=-1)
    x = src.display(
        canvas=fig,
        path=False,
        style_magnetization_show=True,
        style_magnetization_color_mode="tricycle",
    )
    assert x is None, "display test fail"


def test_CylinderSegment_display():
    """testing display"""
    magpy.defaults.display.backend = "plotly"
    fig = go.Figure()
    src = CylinderSegment((1, 2, 3), (2, 4, 5, 30, 40))
    x = src.display(canvas=fig, path=15)
    assert x is None, "path should revert to True"

    src.move(np.linspace((.4,.4,.4), (12.4,12.4,12.4), 33), start=-1)
    x = src.display(
        canvas=fig,
        path=False,
        style_magnetization_show=True,
        style_magnetization_color_mode="bicolor",
    )
    assert x is None, "display test fail"


def test_Sphere_display():
    """testing display"""
    magpy.defaults.display.backend = "plotly"
    fig = go.Figure()
    src = Sphere((1, 2, 3), 2)
    x = src.display(canvas=fig, path=15)
    assert x is None, "path should revert to True"

    src.move(np.linspace((.4,.4,.4), (8,8,8), 33), start=-1)
    x = src.display(canvas=fig, path=False, style_magnetization_show=True)
    assert x is None, "display test fail"


def test_Cuboid_display():
    """testing display"""
    magpy.defaults.display.backend = "plotly"
    src = Cuboid((1, 2, 3), (1, 2, 3))
    src.move(np.linspace((.1,.1,.1), (2,2,2), 20), start=-1)
    x = src.display(path=5, style_magnetization_show=True, renderer="json")
    assert x is None, "display test fail"

    fig = go.Figure()
    x = src.display(canvas=fig, path=False, style_magnetization_show=True)
    assert x is None, "display test fail"


def test_Sensor_display():
    """testing display"""
    magpy.defaults.display.backend = "plotly"
    fig = go.Figure()
    sens_nopix = magpy.Sensor()
    x = sens_nopix.display(canvas=fig, style_description_text="mysensor")
    assert x is None, "display test fail"
    sens = magpy.Sensor(pixel=[(1, 2, 3), (2, 3, 4)])
    sens.move(np.linspace((.4,.4,.4), (12.4,12.4,12.4), 33), start=-1)
    x = sens.display(canvas=fig, markers=[(100, 100, 100)], path=15)
    assert x is None, "display test fail"
    x = sens.display(canvas=fig, markers=[(100, 100, 100)], path=False)
    assert x is None, "display test fail"


def test_Loop_display():
    """testing display for Loop source"""
    magpy.defaults.display.backend = "plotly"
    fig = go.Figure()
    src = magpy.current.Loop(current=1, diameter=1)
    x = src.display(canvas=fig)
    assert x is None, "display test fail"

    src.rotate_from_angax([5] * 35, "x", anchor=(1, 2, 3))
    x = src.display(canvas=fig, path=3)
    assert x is None, "display test fail"


def test_col_display():
    """testing display"""
    # pylint: disable=assignment-from-no-return
    magpy.defaults.display.backend = "plotly"
    fig = go.Figure()
    pm1 = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    pm2 = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    col = magpy.Collection(pm1, pm2)
    x = col.display(canvas=fig)
    assert x is None, "colletion display test fail"


def test_dipole_display():
    """testing display"""
    # pylint: disable=assignment-from-no-return
    magpy.defaults.display.backend = "plotly"
    fig = go.Figure()
    dip1 = magpy.misc.Dipole(moment=(1, 2, 3), position=(1, 1, 1))
    dip2 = magpy.misc.Dipole(moment=(1, 2, 3), position=(2, 2, 2))
    dip3 = magpy.misc.Dipole(moment=(1, 2, 3), position=(3, 3, 3))
    dip2.move(np.linspace((.4,.4,.4), (2,2,2), 5), start=-1)
    x = dip1.display(canvas=fig, style_pivot="tail")
    assert x is None, "display test fail"
    x = dip2.display(canvas=fig, path=2, style_pivot="tip")
    assert x is None, "display test fail"
    x = dip3.display(canvas=fig, path=2, style_pivot="middle")
    assert x is None, "display test fail"


def test_circular_line_display():
    """testing display"""
    # pylint: disable=assignment-from-no-return
    magpy.defaults.display.backend = "plotly"
    fig = go.Figure()
    src1 = magpy.current.Loop(1, 2)
    src2 = magpy.current.Loop(1, 2)
    src1.move(np.linspace((.4,.4,.4), (2,2,2), 5), start=-1)
    src3 = magpy.current.Line(1, [(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    src4 = magpy.current.Line(1, [(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    src3.move([(0.4, 0.4, 0.4)] * 5, start=-1)
    x = src1.display(canvas=fig, path=2, style_arrow_show=False)
    assert x is None, "display test fail"
    x = src2.display(canvas=fig)
    assert x is None, "display test fail"
    x = src3.display(canvas=fig, style_arrow_show=False)
    assert x is None, "display test fail"
    x = src4.display(canvas=fig, path=2)
    assert x is None, "display test fail"


def test_display_bad_style_kwargs():
    """test if some magic kwargs are invalid"""
    magpy.defaults.display.backend = "plotly"
    fig = go.Figure()
    with pytest.raises(ValueError):
        magpy.display(canvas=fig, markers=[(1, 2, 3)], style_bad_style_kwarg=None)


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
    cuboid.move(np.linspace((.4,.4,.4), (12.4,12.4,12.4), 33), start=-1)
    cuboid.style.model3d.extra = [
        {
            "backend": "plotly",
            "trace": {
                "type": "scatter3d",
                "x": [-1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1],
                "y": [-1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
                "z": [-1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1],
                "mode": "lines",
            },
            "show": True,
        },
        {
            "backend": "plotly",
            "trace": {
                "type": "mesh3d",
                "i": [7, 0, 0, 0, 4, 4, 2, 6, 4, 0, 3, 7],
                "j": [0, 7, 1, 2, 6, 7, 1, 2, 5, 5, 2, 2],
                "k": [3, 4, 2, 3, 5, 6, 5, 5, 0, 1, 7, 6],
                "x": [-1, -1, 1, 1, -1, -1, 1, 1],
                "y": [-1, 1, 1, -1, -1, 1, 1, -1],
                "z": [-1, -1, -1, -1, 1, 1, 1, 1],
            },
            "show": True,
        },
    ]
    fig = go.Figure()
    x = cuboid.display(canvas=fig, style=dict(model3d_show=True))
    assert x is None, "display test fail"
    cuboid.style.model3d.extra[0].show = False
    x = cuboid.display(canvas=fig)
    assert x is None, "display test fail"
    x = cuboid.display(
        canvas=fig,
        path="animate",
        style=dict(model3d_show=False),
    )
    assert x is None, "display test fail"
    cuboid.style.model3d.extra = {
            "backend": "plotly",
            "trace": {
                "type": "scatter3d",
                "x": [-1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1],
                "y": [-1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
                "z": [-1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1],
                "mode": "lines",
            },
            "show": True,
    }
    cuboid.style.model3d.extra[0].show = False
    x = cuboid.display(
        canvas=fig,
        path=False,
        style=dict(model3d_show=False),
    )
    assert x is None, "display test fail"


def test_CustomSource_display():
    """testing display"""
    fig = go.Figure()
    cs = magpy.misc.CustomSource(style=dict(color='blue'))
    x = cs.display(canvas=fig, backend="plotly")
    assert x is None, "display test fail"


def test_empty_display():
    """should not fail if nothing to display"""
    fig = go.Figure()
    x = magpy.display(canvas=fig, backend="plotly")
    assert x is None, "empty display plotly test fail"


def test_display_warnings():
    """should display some animation warnings"""
    magpy.defaults.display.backend = "plotly"
    magpy.defaults.display.animation.maxfps = 2
    magpy.defaults.display.animation.maxframes = 2
    src = Cuboid((1, 2, 3), (1, 2, 3))
    src.move(np.linspace((.4,.4,.4), (4,4,4), 10), start=-1)
    fig = go.Figure()

    with pytest.warns(UserWarning):  # animate_fps to big warning
        src.display(canvas=fig, path="animate", animate_time=2, animate_fps=3)
    with pytest.warns(UserWarning):  # max frames surpassed
        src.display(canvas=fig, path="animate", animate_time=2, animate_fps=1)
    src = Cuboid((1, 2, 3), (1, 2, 3))
    with pytest.warns(UserWarning):  # no objet path detected
        src.display(canvas=fig, path="animate")

import numpy as np
import plotly.graph_objects as go
import pytest

import magpylib as magpy
from magpylib._src.exceptions import MagpylibBadUserInput

# pylint: disable=assignment-from-no-return
# pylint: disable=no-member


def test_Cylinder_display():
    """testing display"""
    magpy.defaults.display.backend = "plotly"
    fig = go.Figure()
    src = magpy.magnet.Cylinder((1, 2, 3), (1, 2))
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
    src = magpy.magnet.CylinderSegment((1, 2, 3), (2, 4, 5, 30, 40))
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
    src = magpy.magnet.Sphere((1, 2, 3), 2)
    x = src.show(canvas=fig, style_path_frames=15)
    assert x is None, "path should revert to True"

    src.move(np.linspace((0.4, 0.4, 0.4), (8, 8, 8), 33), start=-1)
    x = src.show(canvas=fig, style_path_show=False, style_magnetization_show=True)
    assert x is None, "display test fail"


def test_Cuboid_display():
    """testing display"""
    magpy.defaults.display.backend = "plotly"
    src = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
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


def test_Circle_display():
    """testing display for Circle source"""
    magpy.defaults.display.backend = "plotly"
    fig = go.Figure()
    src = magpy.current.Circle(current=1, diameter=1)
    x = src.show(canvas=fig)
    assert x is None, "display test fail"

    src.rotate_from_angax([5] * 35, "x", anchor=(1, 2, 3))
    x = src.show(canvas=fig, style_path_frames=3)
    assert x is None, "display test fail"


def test_Triangle_display():
    """testing display for Triangle source"""
    # this test is necessary to cover the case where the backend can display mag arrows and
    # color gradient must be deactivated
    verts = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    src = magpy.misc.Triangle(magnetization=(100, 200, 300), vertices=verts)
    src.show(style_magnetization_mode="arrow", return_fig=True)


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
    src1 = magpy.current.Circle(1, 2)
    src2 = magpy.current.Circle(1, 2)
    src1.move(np.linspace((0.4, 0.4, 0.4), (2, 2, 2), 5), start=-1)
    src3 = magpy.current.Polyline(1, [(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    src4 = magpy.current.Polyline(1, [(0, 0, 0), (1, 1, 1), (2, 2, 2)])
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


def test_extra_model3d():
    """test diplay when object has an extra model object attached"""
    magpy.defaults.display.backend = "plotly"
    cuboid = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    cuboid.move(np.linspace((0.4, 0.4, 0.4), (12.4, 12.4, 12.4), 33), start=-1)
    cuboid.style.model3d.showdefault = False
    cuboid.style.model3d.data = [
        {
            "backend": "generic",
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
    cuboid.show(canvas=fig, style={"model3d_showdefault": True})

    cuboid.style.model3d.data[0].show = False
    cuboid.show(canvas=fig)

    coll = magpy.Collection(cuboid)
    coll.rotate_from_angax(45, "z")
    magpy.show(
        coll,
        canvas=fig,
        animation=True,
        style={"model3d_showdefault": False},
    )

    def my_callable_kwargs():
        return {
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
    cuboid.show(
        canvas=fig,
        style_path_show=False,
        style={"model3d_showdefault": False},
    )


def test_CustomSource_display():
    """testing display"""
    fig = go.Figure()
    cs = magpy.misc.CustomSource(style={"color": "blue"})
    x = cs.show(canvas=fig, backend="plotly")
    assert x is None, "display test fail"


def test_empty_display():
    """should not fail if nothing to display"""
    fig = magpy.show(backend="plotly", return_fig=True)
    assert isinstance(fig, go.Figure), "empty display plotly test fail"


def test_display_warnings():
    """should display some animation warnings"""
    magpy.defaults.display.backend = "plotly"
    magpy.defaults.display.animation.maxfps = 2
    magpy.defaults.display.animation.maxframes = 2
    src = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    src.move(np.linspace((0.4, 0.4, 0.4), (4, 4, 4), 10), start=-1)
    fig = go.Figure()

    with pytest.warns(UserWarning):  # animation_fps to big warning
        src.show(canvas=fig, animation=5, animation_fps=3)
    with pytest.warns(UserWarning):  # max frames surpassed
        src.show(canvas=fig, animation=True, animation_time=2, animation_fps=1)
    src = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    with pytest.warns(UserWarning):  # no object path detected
        src.show(canvas=fig, style_path_frames=[], animation=True)


def test_bad_animation_value():
    """should fail if animation is not a boolean or a positive number"""
    magpy.defaults.display.backend = "plotly"
    magpy.defaults.display.animation.maxfps = 2
    magpy.defaults.display.animation.maxframes = 2
    src = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    src.move(np.linspace((0.4, 0.4, 0.4), (4, 4, 4), 10), start=-1)
    fig = go.Figure()

    with pytest.raises(MagpylibBadUserInput):
        src.show(canvas=fig, animation=-1)


def test_subplots():
    """test subplots"""
    sensors = magpy.Collection(
        [
            magpy.Sensor(
                pixel=np.linspace((x, 0, -0.2), (x, 0, 0.2), 2), style_label=str(x)
            )
            for x in np.linspace(0, 10, 11)
        ]
    )
    cyl1 = magpy.magnet.Cylinder(magnetization=(100, 0, 0), dimension=(1, 2))

    # define paths
    sensors.position = np.linspace((0, 0, -3), (0, 0, 3), 100)
    cyl1.position = (4, 0, 0)
    cyl1.rotate_from_angax(angle=np.linspace(0, 300, 100), start=0, axis="z", anchor=0)
    objs = cyl1, sensors

    # with implicit axes
    fig = go.Figure()
    with magpy.show_context(
        backend="plotly", canvas=fig, animation=False, sumup=False, pixel_agg="mean"
    ) as s:
        s.show(
            *objs, col=1, output="B", style_path_frames=10, sumup=False, pixel_agg=None
        )

    # bad output value
    with pytest.raises(
        ValueError, match=r"The `output` parameter must start with 'B' or 'H'.*"
    ):
        magpy.show(*objs, canvas=fig, output="bad_output")


def test_legends():
    """test legends"""
    f = 0.5
    N = 3
    xs = f * np.array([-1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1])
    ys = f * np.array([-1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1])
    zs = f * np.array([-1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1])
    trace_plotly = {
        "backend": "plotly",
        "constructor": "scatter3d",
        "kwargs": {"x": xs, "y": ys, "z": zs, "mode": "lines"},
    }
    c = magpy.magnet.Cuboid((0, 0, 1000), (1, 1, 1), style_label="Plotly extra trace")
    c.style.model3d.add_trace(trace_plotly)

    fig = magpy.show(
        c,
        backend="plotly",
        style_path_frames=2,
        style_legend_show=False,
        # style_model3d_showdefault=False,
        return_fig=True,
    )
    assert [t.name for t in fig.data] == ["Plotly extra trace (1mm|1mm|1mm)"] * 2
    assert [t.showlegend for t in fig.data] == [False, False]

    fig = magpy.show(
        c,
        backend="plotly",
        style_path_frames=2,
        # style_legend_show=False,
        # style_model3d_showdefault=False,
        return_fig=True,
    )
    assert [t.name for t in fig.data] == ["Plotly extra trace (1mm|1mm|1mm)"] * 2
    assert [t.showlegend for t in fig.data] == [True, False]

    fig = magpy.show(
        c,
        backend="plotly",
        style_path_frames=2,
        # style_legend_show=False,
        style_model3d_showdefault=False,
        return_fig=True,
    )
    assert [t.name for t in fig.data] == ["Plotly extra trace (1mm|1mm|1mm)"]
    assert [t.showlegend for t in fig.data] == [True]

    c.rotate_from_angax([10 * i for i in range(N)], "y", start=0, anchor=(0, 0, 10))
    fig = magpy.show(
        c,
        backend="plotly",
        style_path_frames=2,
        # style_legend_show=False,
        # style_model3d_showdefault=False,
        return_fig=True,
    )
    assert [t.name for t in fig.data] == ["Plotly extra trace (1mm|1mm|1mm)"] * 4
    assert [t.showlegend for t in fig.data] == [True, False, False, False]

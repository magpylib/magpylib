import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pytest

import magpylib as magpy
from magpylib.graphics.model3d import make_Cuboid
from magpylib.magnet import Cuboid
from magpylib.magnet import Cylinder
from magpylib.magnet import CylinderSegment
from magpylib.magnet import Sphere

# pylint: disable=assignment-from-no-return

magpy.defaults.reset()


def test_Cylinder_display():
    """testing display"""
    ax = plt.subplot(projection="3d")
    src = Cylinder((1, 2, 3), (1, 2))
    x = src.show(canvas=ax, style_path_frames=15, backend="matplotlib")
    assert x is None, "path should revert to True"
    src.move(np.linspace((0.4, 0.4, 0.4), (12, 12, 12), 30), start=-1)
    x = src.show(
        canvas=ax, style_path_show=False, show_direction=True, backend="matplotlib"
    )
    assert x is None, "display test fail"

    x = src.show(
        canvas=ax, style_path_frames=[], show_direction=True, backend="matplotlib"
    )
    assert x is None, "ind>path_len, should display last position"

    x = src.show(
        canvas=ax,
        style_path_frames=[1, 5, 6],
        show_direction=True,
        backend="matplotlib",
    )
    assert x is None, "should display 1,5,6 position"


def test_CylinderSegment_display():
    """testing display"""
    ax = plt.subplot(projection="3d")
    src = CylinderSegment((1, 2, 3), (2, 4, 5, 30, 40))
    x = src.show(canvas=ax, style_path_frames=15)
    assert x is None, "path should revert to True"

    src.move(np.linspace((0.4, 0.4, 0.4), (13.2, 13.2, 13.2), 33), start=-1)
    x = src.show(canvas=ax, style_path_show=False, show_direction=True)
    assert x is None, "display test fail"


def test_Sphere_display():
    """testing display"""
    ax = plt.subplot(projection="3d")
    src = Sphere((1, 2, 3), 2)
    x = src.show(canvas=ax, style_path_frames=15)
    assert x is None, "path should revert to True"

    src.move(np.linspace((0.4, 0.4, 0.4), (8, 8, 8), 20), start=-1)
    x = src.show(canvas=ax, style_path_show=False, show_direction=True)
    assert x is None, "display test fail"


def test_Cuboid_display():
    """testing display"""
    src = Cuboid((1, 2, 3), (1, 2, 3))
    src.move(np.linspace((0.1, 0.1, 0.1), (2, 2, 2), 20), start=-1)
    plt.ion()
    x = src.show(style_path_frames=5, show_direction=True)
    plt.close()
    assert x is None, "display test fail"

    ax = plt.subplot(projection="3d")
    x = src.show(canvas=ax, style_path_show=False, show_direction=True)
    assert x is None, "display test fail"


def test_Sensor_display():
    """testing display"""
    ax = plt.subplot(projection="3d")
    sens = magpy.Sensor(pixel=[(1, 2, 3), (2, 3, 4)])
    sens.style.arrows.z.color = "magenta"
    sens.style.arrows.z.show = False
    poz = np.linspace((0.4, 0.4, 0.4), (13.2, 13.2, 13.2), 33)
    sens.move(poz, start=-1)
    x = sens.show(canvas=ax, markers=[(100, 100, 100)], style_path_frames=15)
    assert x is None, "display test fail"

    x = sens.show(canvas=ax, markers=[(100, 100, 100)], style_path_show=False)
    assert x is None, "display test fail"


def test_CustomSource_display():
    """testing display"""
    ax = plt.subplot(projection="3d")
    cs = magpy.misc.CustomSource()
    x = cs.show(canvas=ax)
    assert x is None, "display test fail"


def test_Loop_display():
    """testing display for Loop source"""
    ax = plt.subplot(projection="3d")
    src = magpy.current.Loop(current=1, diameter=1)
    x = src.show(canvas=ax)
    assert x is None, "display test fail"

    src.rotate_from_angax([5] * 35, "x", anchor=(1, 2, 3))
    x = src.show(canvas=ax, style_path_frames=3)
    assert x is None, "display test fail"


def test_col_display():
    """testing display"""
    # pylint: disable=assignment-from-no-return
    ax = plt.subplot(projection="3d")
    pm1 = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    pm2 = pm1.copy(position=(2, 0, 0))
    pm3 = pm1.copy(position=(4, 0, 0))
    nested_col = (pm1 + pm2 + pm3).set_children_styles(color="magenta")
    x = nested_col.show(canvas=ax)
    assert x is None, "colletion display test fail"


def test_dipole_display():
    """testing display"""
    # pylint: disable=assignment-from-no-return
    ax2 = plt.subplot(projection="3d")
    dip = magpy.misc.Dipole(moment=(1, 2, 3), position=(2, 2, 2))
    dip2 = magpy.misc.Dipole(moment=(1, 2, 3), position=(2, 2, 2))
    dip2.move(np.linspace((0.4, 0.4, 0.4), (2, 2, 2), 5), start=-1)
    x = dip.show(canvas=ax2)
    assert x is None, "display test fail"
    x = dip.show(canvas=ax2, style_path_frames=2)
    assert x is None, "display test fail"


def test_circular_line_display():
    """testing display"""
    # pylint: disable=assignment-from-no-return
    ax2 = plt.subplot(projection="3d")
    src1 = magpy.current.Loop(1, 2)
    src2 = magpy.current.Loop(1, 2)
    src1.move(np.linspace((0.4, 0.4, 0.4), (2, 2, 2), 5), start=-1)
    src3 = magpy.current.Line(1, [(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    src4 = magpy.current.Line(1, [(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    src3.move([(0.4, 0.4, 0.4)] * 5, start=-1)
    x = src1.show(canvas=ax2, style_path_frames=2, style_arrow_size=0)
    assert x is None, "display test fail"
    x = src2.show(canvas=ax2)
    assert x is None, "display test fail"
    x = src3.show(canvas=ax2, style_arrow_size=0)
    assert x is None, "display test fail"
    x = src4.show(canvas=ax2, style_path_frames=2)
    assert x is None, "display test fail"


def test_matplotlib_animation_warning():
    """animation=True with matplotlib should raise UserWarning"""
    ax = plt.subplot(projection="3d")
    sens = magpy.Sensor(pixel=[(1, 2, 3), (2, 3, 4)])
    sens.move(np.linspace((0.4, 0.4, 0.4), (12.4, 12.4, 12.4), 33), start=-1)
    with pytest.warns(UserWarning):
        sens.show(canvas=ax, animation=True)


def test_matplotlib_model3d_extra():
    """test display extra model3d"""

    # using "plot"
    xs, ys, zs = [(1, 2)] * 3
    trace1 = dict(
        backend="matplotlib",
        constructor="plot",
        args=(xs, ys, zs),
        kwargs=dict(ls="-"),
    )
    obj1 = magpy.misc.Dipole(moment=(0, 0, 1))
    obj1.style.model3d.add_trace(**trace1)

    # using "plot_surface"
    u, v = np.mgrid[0 : 2 * np.pi : 6j, 0 : np.pi : 6j]
    xs = np.cos(u) * np.sin(v)
    ys = np.sin(u) * np.sin(v)
    zs = np.cos(v)
    trace2 = dict(
        backend="matplotlib",
        constructor="plot_surface",
        args=(xs, ys, zs),
        kwargs=dict(
            cmap=plt.cm.YlGnBu_r,
        ),
    )
    obj2 = magpy.Collection()
    obj2.style.model3d.add_trace(**trace2)

    # using "plot_trisurf"
    u, v = np.mgrid[0 : 2 * np.pi : 6j, -0.5:0.5:6j]
    u, v = u.flatten(), v.flatten()
    xs = (1 + 0.5 * v * np.cos(u / 2.0)) * np.cos(u)
    ys = (1 + 0.5 * v * np.cos(u / 2.0)) * np.sin(u)
    zs = 0.5 * v * np.sin(u / 2.0)
    tri = mtri.Triangulation(u, v)
    trace3 = dict(
        backend="matplotlib",
        constructor="plot_trisurf",
        args=lambda: (xs, ys, zs),  # test callable args
        kwargs=dict(
            triangles=tri.triangles,
            cmap=plt.cm.Spectral,
        ),
    )
    obj3 = magpy.misc.CustomSource(style_model3d_showdefault=False, position=(3, 0, 0))
    obj3.style.model3d.add_trace(**trace3)

    ax = plt.subplot(projection="3d")
    x = magpy.show(obj1, obj2, obj3, canvas=ax)
    assert x is None, "display test fail"


def test_matplotlib_model3d_extra_bad_input():
    """test display extra model3d"""

    xs, ys, zs = [(1, 2)] * 3
    trace = dict(
        backend="matplotlib",
        constructor="plot",
        kwargs={"xs": xs, "ys": ys, "zs": zs},
        coordsargs={"x": "xs", "y": "ys", "z": "Z"},  # bad Z input
    )
    obj = magpy.misc.Dipole(moment=(0, 0, 1))
    with pytest.raises(ValueError):
        obj.style.model3d.add_trace(**trace)
        ax = plt.subplot(projection="3d")
        obj.show(canvas=ax)


def test_matplotlib_model3d_extra_updatefunc():
    """test display extra model3d"""
    ax = plt.subplot(projection="3d")
    obj = magpy.misc.Dipole(moment=(0, 0, 1))
    updatefunc = lambda: make_Cuboid("matplotlib", position=(2, 0, 0))
    obj.style.model3d.data = updatefunc
    ax = plt.subplot(projection="3d")
    obj.show(canvas=ax)

    with pytest.raises(ValueError):
        updatefunc = "not callable"
        obj.style.model3d.add_trace(updatefunc)

    with pytest.raises(AssertionError):
        updatefunc = "not callable"
        obj.style.model3d.add_trace(updatefunc=updatefunc)

    with pytest.raises(AssertionError):
        updatefunc = lambda: "bad output type"
        obj.style.model3d.add_trace(updatefunc=updatefunc)

    with pytest.raises(AssertionError):
        updatefunc = lambda: {"bad_key": "some_value"}
        obj.style.model3d.add_trace(updatefunc=updatefunc)


def test_empty_display():
    """should not fail if nothing to display"""
    ax = plt.subplot(projection="3d")
    x = magpy.show(canvas=ax, backend="matplotlib")
    assert x is None, "empty display matplotlib test fail"


def test_graphics_model_mpl():
    """test base extra graphics with mpl"""
    ax = plt.subplot(projection="3d")
    c = magpy.magnet.Cuboid((0, 1, 0), (1, 1, 1))
    c.rotate_from_angax(33, "x", anchor=0)
    c.style.model3d.add_trace(**make_Cuboid("matplotlib", position=(2, 0, 0)))
    c.show(canvas=ax, style_path_frames=1, backend="matplotlib")

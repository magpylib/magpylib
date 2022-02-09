import pytest
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy
from magpylib.magnet import Cylinder, Cuboid, Sphere, CylinderSegment
from magpylib._src.display.plotly.plotly_base_traces import make_BaseCuboid

# pylint: disable=assignment-from-no-return


def test_Cylinder_display():
    """testing display"""
    ax = plt.subplot(projection="3d")
    src = Cylinder((1, 2, 3), (1, 2))
    x = src.show(canvas=ax, style_path_frames=15)
    assert x is None, "path should revert to True"
    src.move(np.linspace((.4,.4,.4), (12,12,12), 30), start=-1)
    x = src.show(canvas=ax, style_path_show=False, show_direction=True)
    assert x is None, "display test fail"

    x = src.show(canvas=ax, style_path_frames=[], show_direction=True)
    assert x is None, "ind>path_len, should display last position"

    x = src.show(canvas=ax, style_path_frames=[1, 5, 6], show_direction=True)
    assert x is None, "should display 1,5,6 position"


def test_CylinderSegment_display():
    """testing display"""
    ax = plt.subplot(projection="3d")
    src = CylinderSegment((1, 2, 3), (2, 4, 5, 30, 40))
    x = src.show(canvas=ax, style_path_frames=15)
    assert x is None, "path should revert to True"

    src.move(np.linspace((.4,.4,.4), (13.2,13.2,13.2), 33), start=-1)
    x = src.show(canvas=ax, style_path_show=False, show_direction=True)
    assert x is None, "display test fail"


def test_Sphere_display():
    """testing display"""
    ax = plt.subplot(projection="3d")
    src = Sphere((1, 2, 3), 2)
    x = src.show(canvas=ax, style_path_frames=15)
    assert x is None, "path should revert to True"

    src.move(np.linspace((.4,.4,.4), (8,8,8), 20), start=-1)
    x = src.show(canvas=ax, style_path_show=False, show_direction=True)
    assert x is None, "display test fail"


def test_Cuboid_display():
    """testing display"""
    src = Cuboid((1, 2, 3), (1, 2, 3))
    src.move(np.linspace((.1,.1,.1), (2,2,2), 20), start=-1)
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
    sens.style.arrows.z.color = 'magenta'
    sens.style.arrows.z.show = False
    poz = np.linspace((.4,.4,.4), (13.2,13.2,13.2), 33)
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
    pm2 = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    col = magpy.Collection(pm1, pm2)
    x = col.show(canvas=ax)
    assert x is None, "colletion display test fail"


def test_dipole_display():
    """testing display"""
    # pylint: disable=assignment-from-no-return
    ax2 = plt.subplot(projection="3d")
    dip = magpy.misc.Dipole(moment=(1, 2, 3), position=(2, 2, 2))
    dip2 = magpy.misc.Dipole(moment=(1, 2, 3), position=(2, 2, 2))
    dip2.move(np.linspace((.4,.4,.4), (2,2,2), 5), start=-1)
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
    src1.move(np.linspace((.4,.4,.4), (2,2,2), 5), start=-1)
    src3 = magpy.current.Line(1, [(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    src4 = magpy.current.Line(1, [(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    src3.move([(.4,.4,.4)]*5, start=-1)
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
    sens.move(np.linspace((.4,.4,.4), (12.4,12.4,12.4), 33), start=-1)
    with pytest.warns(UserWarning):
        sens.show(canvas=ax, animation=True)


def test_matplotlib_model3d_extra():
    """test display extra model3d"""
    cuboid = magpy.magnet.Cuboid(
        magnetization=(1, 0, 0), dimension=(3, 3, 3), position=(10, 0, 0)
    ).rotate_from_angax(np.linspace(72,360,5), "z", anchor=(0, 0, 0), start=0)
    ax = plt.subplot(projection="3d")
    cuboid.style.model3d.showdefault = False
    cuboid.style.model3d.data = [
        {
            "backend": "matplotlib",
            "trace": {
                "type": "plot",
                "xs": [-1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1],
                "ys": [-1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
                "zs": [-1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1],
                "ls": "-",
            },
            "show": True,
        }
    ]
    with pytest.raises(ValueError): # should fail because of invalid coordsargs
        x = cuboid.show(canvas=ax, style_path_frames=1)
    cuboid.style.model3d.data[0].coordsargs = {"x": "xs", "y": "ys", "z": "zs"}
    x = cuboid.show(canvas=ax, style_path_frames=1)

    assert x is None, "display test fail"

    cube = make_BaseCuboid()
    i,j,k = [cube[k] for k in 'ijk']
    x,y,z = [cube[k] for k in 'xyz']
    triangles = np.array([i, j, k]).T
    trace = dict(type="plot_trisurf", args=(x, y, z), triangles=triangles)
    cuboid.style.model3d.showdefault = False
    cuboid.style.model3d.data = [
        {
            "backend": "matplotlib",
            "coordsargs": {"x": "args[0]", "y": "args[1]", "z": "args[2]"},
            "trace": trace,
            "show": True,
        }
    ]
    x = cuboid.show(canvas=ax, style_path_frames=1)
    assert x is None, "display test fail"


def test_empty_display():
    """should not fail if nothing to display"""
    ax = plt.subplot(projection="3d")
    x = magpy.show(canvas=ax, backend="matplotlib")
    assert x is None, "empty display matplotlib test fail"

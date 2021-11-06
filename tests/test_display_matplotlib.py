import pytest
import matplotlib.pyplot as plt
import magpylib as magpy
from magpylib.magnet import Cylinder, Cuboid, Sphere, CylinderSegment
from magpylib._lib.display.plotly_draw import make_BaseCuboid

# pylint: disable=assignment-from-no-return


def test_Cylinder_display():
    """testing display"""
    ax = plt.subplot(projection="3d")
    src = Cylinder((1, 2, 3), (1, 2))
    x = src.display(canvas=ax, path=15)
    assert x is None, "path should revert to True"
    path_len = 30
    src.move([(0.4, 0.4, 0.4)] * path_len, increment=True)
    x = src.display(canvas=ax, path=False, show_direction=True)
    assert x is None, "display test fail"

    x = src.display(canvas=ax, path=[], show_direction=True)
    assert x is None, "ind>path_len, should display last position"

    x = src.display(canvas=ax, path=[1, 5, 6], show_direction=True)
    assert x is None, "should display 1,5,6 position"


def test_CylinderSegment_display():
    """testing display"""
    ax = plt.subplot(projection="3d")
    src = CylinderSegment((1, 2, 3), (2, 4, 5, 30, 40))
    x = src.display(canvas=ax, path=15)
    assert x is None, "path should revert to True"

    src.move([(0.4, 0.4, 0.4)] * 33, increment=True)
    x = src.display(canvas=ax, path=False, show_direction=True)
    assert x is None, "display test fail"


def test_Sphere_display():
    """testing display"""
    ax = plt.subplot(projection="3d")
    src = Sphere((1, 2, 3), 2)
    x = src.display(canvas=ax, path=15)
    assert x is None, "path should revert to True"

    src.move([(0.4, 0.4, 0.4)] * 20, increment=True)
    x = src.display(canvas=ax, path=False, show_direction=True)
    assert x is None, "display test fail"


def test_Cuboid_display():
    """testing display"""
    src = Cuboid((1, 2, 3), (1, 2, 3))
    src.move([(0.1, 0.1, 0.1)] * 20, increment=True)
    plt.ion()
    x = src.display(path=5, show_direction=True)
    plt.close()
    assert x is None, "display test fail"

    ax = plt.subplot(projection="3d")
    x = src.display(canvas=ax, path=False, show_direction=True)
    assert x is None, "display test fail"


def test_Sensor_display():
    """testing display"""
    ax = plt.subplot(projection="3d")
    sens = magpy.Sensor(pixel=[(1, 2, 3), (2, 3, 4)])
    sens.move([(0.4, 0.4, 0.4)] * 33, increment=True)
    x = sens.display(canvas=ax, markers=[(100, 100, 100)], path=15)
    assert x is None, "display test fail"

    x = sens.display(canvas=ax, markers=[(100, 100, 100)], path=False)
    assert x is None, "display test fail"


def test_Circular_display():
    """testing display for Circular source"""
    ax = plt.subplot(projection="3d")
    src = magpy.current.Circular(current=1, diameter=1)
    x = src.display(canvas=ax)
    assert x is None, "display test fail"

    src.rotate_from_angax([5] * 35, "x", anchor=(1, 2, 3))
    x = src.display(canvas=ax, path=3)
    assert x is None, "display test fail"


def test_col_display():
    """testing display"""
    # pylint: disable=assignment-from-no-return
    ax = plt.subplot(projection="3d")
    pm1 = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    pm2 = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    col = magpy.Collection(pm1, pm2)
    x = col.display(canvas=ax)
    assert x is None, "colletion display test fail"


def test_dipole_display():
    """testing display"""
    # pylint: disable=assignment-from-no-return
    ax2 = plt.subplot(projection="3d")
    dip = magpy.misc.Dipole(moment=(1, 2, 3), position=(2, 2, 2))
    dip2 = magpy.misc.Dipole(moment=(1, 2, 3), position=(2, 2, 2))
    dip2.move([(0.4, 0.4, 0.4)] * 5, increment=True)
    x = dip.display(canvas=ax2)
    assert x is None, "display test fail"
    x = dip.display(canvas=ax2, path=2)
    assert x is None, "display test fail"


def test_circular_line_display():
    """testing display"""
    # pylint: disable=assignment-from-no-return
    ax2 = plt.subplot(projection="3d")
    src1 = magpy.current.Circular(1, 2)
    src2 = magpy.current.Circular(1, 2)
    src1.move([(0.4, 0.4, 0.4)] * 5, increment=True)
    src3 = magpy.current.Line(1, [(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    src4 = magpy.current.Line(1, [(0, 0, 0), (1, 1, 1), (2, 2, 2)])
    src3.move([(0.4, 0.4, 0.4)] * 5, increment=False)
    x = src1.display(canvas=ax2, path=2, style_current_size=0)
    assert x is None, "display test fail"
    x = src2.display(canvas=ax2)
    assert x is None, "display test fail"
    x = src3.display(canvas=ax2, style_current_size=0)
    assert x is None, "display test fail"
    x = src4.display(canvas=ax2, path=2)
    assert x is None, "display test fail"


def test_matplotlib_animation_warning():
    """animate with matplotlib should raise UserWarning"""
    ax = plt.subplot(projection="3d")
    sens = magpy.Sensor(pixel=[(1, 2, 3), (2, 3, 4)])
    sens.move([(0.4, 0.4, 0.4)] * 33, increment=True)
    with pytest.warns(UserWarning):
        sens.display(canvas=ax, path="animate")


def test_matplotlib_mesh3d_warning():
    """if obj has mesh3d with matplotlib should raise UserWarning"""
    data = make_BaseCuboid(pos=(4, 0, 0), dim=(2, 2, 2))
    ax = plt.subplot(projection="3d")
    sens = magpy.Sensor(
        pixel=[(1, 2, 3), (2, 3, 4)],
        style=dict(mesh3d_data=data, mesh3d_show="inplace"),
    )
    sens.move([(0.4, 0.4, 0.4)] * 33, increment=True)
    with pytest.warns(UserWarning):
        sens.display(canvas=ax)

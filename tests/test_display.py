import matplotlib.pyplot as plt
import magpylib as mag3
from magpylib.magnet import Cylinder, Box, Sphere

# pylint: disable=assignment-from-no-return

def test_Cylinder_display():
    """ testing display
    """
    ax = plt.subplot(projection='3d')
    src = Cylinder((1,2,3),(1,2))
    x = src.display(axis=ax, show_path=15)
    assert x is None, 'show_path should revert to True'

    src.move([(.4,.4,.4)]*33, increment=True)
    x = src.display(axis=ax, show_path=False, show_direction=True)
    assert x is None, 'display test fail'


def test_Cylinder_display_dim5():
    """ testing display
    """
    ax = plt.subplot(projection='3d')
    src = Cylinder((1,2,3),(1,2,30,40,5))
    x = src.display(axis=ax, show_path=15)
    assert x is None, 'show_path should revert to True'

    src.move([(.4,.4,.4)]*33, increment=True)
    x = src.display(axis=ax, show_path=False, show_direction=True)
    assert x is None, 'display test fail'



def test_Sphere_display():
    """ testing display
    """
    ax = plt.subplot(projection='3d')
    src = Sphere((1,2,3),2)
    x = src.display(axis=ax, show_path=15)
    assert x is None, 'show_path should revert to True'

    src.move([(.4,.4,.4)]*20, increment=True)
    x = src.display(axis=ax, show_path=False, show_direction=True)
    assert x is None, 'display test fail'


def test_Box_display():
    """ testing display
    """
    src = Box((1,2,3),(1,2,3))
    src.move([(.1,.1,.1)]*20, increment=True)
    plt.ion()
    x = src.display(show_path=5, show_direction=True)
    plt.close()
    assert x is None, 'display test fail'

    ax = plt.subplot(projection='3d')
    x = src.display(axis=ax, show_path=False, show_direction=True)
    assert x is None, 'display test fail'


def test_Sensor_display():
    """ testing display
    """
    ax = plt.subplot(projection='3d')
    sens = mag3.Sensor(pixel=[(1,2,3),(2,3,4)])
    sens.move([(.4,.4,.4)]*33, increment=True)
    x = sens.display(axis=ax, markers=[(100,100,100)], show_path=15)
    assert x is None, 'display test fail'

    x = sens.display(axis=ax, markers=[(100,100,100)], show_path=False)
    assert x is None, 'display test fail'


def test_Circular_display():
    """ testing display for Circular source
    """
    ax = plt.subplot(projection='3d')
    src = mag3.current.Circular(current=1, diameter=1)
    x = src.display(axis=ax)
    assert x is None, 'display test fail'

    src.rotate_from_angax([5]*35, 'x', anchor=(1,2,3))
    x = src.display(axis=ax, show_path=3)
    assert x is None, 'display test fail'


def test_col_display():
    """ testing display
    """
    # pylint: disable=assignment-from-no-return
    ax = plt.subplot(projection='3d')
    pm1 = mag3.magnet.Box((1,2,3),(1,2,3))
    pm2 = mag3.magnet.Box((1,2,3),(1,2,3))
    col = mag3.Collection(pm1,pm2)
    x = col.display(axis=ax)
    assert x is None, 'colletion display test fail'


def test_dipole_display():
    """ testing display
    """
    # pylint: disable=assignment-from-no-return
    ax2 = plt.subplot(projection='3d')
    dip = mag3.misc.Dipole(moment=(1,2,3), position=(2,2,2))
    dip2 = mag3.misc.Dipole(moment=(1,2,3), position=(2,2,2))
    dip2.move([(.4,.4,.4)]*5, increment=True)
    x = dip.display(axis=ax2)
    assert x is None, 'display test fail'
    x = dip.display(axis=ax2, show_path=2)
    assert x is None, 'display test fail'


def test_circular_line_display():
    """ testing display
    """
    # pylint: disable=assignment-from-no-return
    ax2 = plt.subplot(projection='3d')
    src1 = mag3.current.Circular(1,2)
    src2 = mag3.current.Circular(1,2)
    src1.move([(.4,.4,.4)]*5, increment=True)
    src3 = mag3.current.Line(1, [(0,0,0),(1,1,1),(2,2,2)])
    src4 = mag3.current.Line(1, [(0,0,0),(1,1,1),(2,2,2)])
    src3.move([(.4,.4,.4)]*5, increment=False)
    x = src1.display(axis=ax2, show_path=2)
    assert x is None, 'display test fail'
    x = src2.display(axis=ax2)
    assert x is None, 'display test fail'
    x = src3.display(axis=ax2)
    assert x is None, 'display test fail'
    x = src4.display(axis=ax2, show_path=2)
    assert x is None, 'display test fail'

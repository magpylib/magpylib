import matplotlib.pyplot as plt
import magpylib as mag3
from magpylib.magnet import Cylinder, Box, Sphere

# pylint: disable=assignment-from-no-return

def test_Cylinder_display():
    """ testing display
    """
    ax = plt.subplot(projection='3d')
    src = Cylinder((1,2,3),(1,2))
    x = src.display(axis=ax,show_path=15)
    assert x is None, 'show_path should revert to True'

    src.move_by((4,4,4),steps=33)
    x = src.display(axis=ax,show_path=False)
    assert x is None, 'display test fail'


def test_Sphere_display():
    """ testing display
    """
    ax = plt.subplot(projection='3d')
    src = Sphere((1,2,3),2)
    x = src.display(axis=ax,show_path=15)
    assert x is None, 'show_path should revert to True'

    src.move_by((4,4,4),steps=33)
    x = src.display(axis=ax,show_path=False)
    assert x is None, 'display test fail'


def test_Box_display():
    """ testing display
    """
    src = Box((1,2,3),(1,2,3))
    src.move_by((3,3,3),steps=15)
    plt.ion()
    x = src.display(show_path=5, direc=True)
    plt.close()
    assert x is None, 'display test fail'

    ax = plt.subplot(projection='3d')
    x = src.display(axis=ax, show_path=False, direc=True)
    assert x is None, 'display test fail'


def test_Sensor_display():
    """ testing display
    """
    ax = plt.subplot(projection='3d')
    sens = mag3.Sensor(pos_pix=[(1,2,3),(2,3,4)])
    sens.move_by((10,1,10),steps=33)
    x = sens.display(axis=ax, markers=[(100,100,100)], show_path=15)
    assert x is None, 'display test fail'

    x = sens.display(axis=ax, markers=[(100,100,100)], show_path=False)
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
    dip = mag3.misc.Dipole(moment=(1,2,3), pos=(2,2,2))
    dip2 = mag3.misc.Dipole(moment=(1,2,3), pos=(2,2,2))
    dip2.move_by((1,2,3), steps=5)
    x = dip.display(axis=ax2)
    assert x is None, 'display test fail'
    x = dip.display(axis=ax2, show_path=2)
    assert x is None, 'display test fail'

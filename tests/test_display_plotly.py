import plotly.graph_objects as go
import magpylib as magpy
from magpylib.magnet import Cylinder, Cuboid, Sphere, CylinderSegment

# pylint: disable=assignment-from-no-return

def test_Cylinder_display():
    """ testing display
    """
    magpy.defaults.display.backend = 'plotly'
    fig = go.Figure()
    src = Cylinder((1,2,3),(1,2))
    x = src.display(canvas=fig, path=15)
    assert x is None, 'path should revert to True'

    src.move([(.4,.4,.4)]*33, increment=True)
    x = src.display(canvas=fig, path=False, style_magnetization_show=True)
    assert x is None, 'display test fail'


def test_CylinderSegment_display():
    """ testing display
    """
    magpy.defaults.display.backend = 'plotly'
    fig = go.Figure()
    src = CylinderSegment((1,2,3),(2,4,5,30,40))
    x = src.display(canvas=fig, path=15)
    assert x is None, 'path should revert to True'

    src.move([(.4,.4,.4)]*33, increment=True)
    x = src.display(canvas=fig, path=False, style_magnetization_show=True)
    assert x is None, 'display test fail'



def test_Sphere_display():
    """ testing display
    """
    magpy.defaults.display.backend = 'plotly'
    fig = go.Figure()
    src = Sphere((1,2,3),2)
    x = src.display(canvas=fig, path=15)
    assert x is None, 'path should revert to True'

    src.move([(.4,.4,.4)]*20, increment=True)
    x = src.display(canvas=fig, path=False, style_magnetization_show=True)
    assert x is None, 'display test fail'


def test_Cuboid_display():
    """ testing display
    """
    magpy.defaults.display.backend = 'plotly'
    src = Cuboid((1,2,3),(1,2,3))
    src.move([(.1,.1,.1)]*20, increment=True)
    x = src.display(path=5, style_magnetization_show=True)
    assert x is None, 'display test fail'

    fig = go.Figure()
    x = src.display(canvas=fig, path=False, style_magnetization_show=True)
    assert x is None, 'display test fail'


def test_Sensor_display():
    """ testing display
    """
    magpy.defaults.display.backend = 'plotly'
    fig = go.Figure()
    sens = magpy.Sensor(pixel=[(1,2,3),(2,3,4)])
    sens.move([(.4,.4,.4)]*33, increment=True)
    x = sens.display(canvas=fig, markers=[(100,100,100)], path=15)
    assert x is None, 'display test fail'

    x = sens.display(canvas=fig, markers=[(100,100,100)], path=False)
    assert x is None, 'display test fail'


def test_Circular_display():
    """ testing display for Circular source
    """
    magpy.defaults.display.backend = 'plotly'
    fig = go.Figure()
    src = magpy.current.Circular(current=1, diameter=1)
    x = src.display(canvas=fig)
    assert x is None, 'display test fail'

    src.rotate_from_angax([5]*35, 'x', anchor=(1,2,3))
    x = src.display(canvas=fig, path=3)
    assert x is None, 'display test fail'


def test_col_display():
    """ testing display
    """
    # pylint: disable=assignment-from-no-return
    magpy.defaults.display.backend = 'plotly'
    fig = go.Figure()
    pm1 = magpy.magnet.Cuboid((1,2,3),(1,2,3))
    pm2 = magpy.magnet.Cuboid((1,2,3),(1,2,3))
    col = magpy.Collection(pm1,pm2)
    x = col.display(canvas=fig)
    assert x is None, 'colletion display test fail'


def test_dipole_display():
    """ testing display
    """
    # pylint: disable=assignment-from-no-return
    magpy.defaults.display.backend = 'plotly'
    fig = go.Figure()
    dip = magpy.misc.Dipole(moment=(1,2,3), position=(2,2,2))
    dip2 = magpy.misc.Dipole(moment=(1,2,3), position=(2,2,2))
    dip2.move([(.4,.4,.4)]*5, increment=True)
    x = dip.display(canvas=fig)
    assert x is None, 'display test fail'
    x = dip.display(canvas=fig, path=2)
    assert x is None, 'display test fail'


def test_circular_line_display():
    """ testing display
    """
    # pylint: disable=assignment-from-no-return
    magpy.defaults.display.backend = 'plotly'
    fig = go.Figure()
    src1 = magpy.current.Circular(1,2)
    src2 = magpy.current.Circular(1,2)
    src1.move([(.4,.4,.4)]*5, increment=True)
    src3 = magpy.current.Line(1, [(0,0,0),(1,1,1),(2,2,2)])
    src4 = magpy.current.Line(1, [(0,0,0),(1,1,1),(2,2,2)])
    src3.move([(.4,.4,.4)]*5, increment=False)
    x = src1.display(canvas=fig, path=2)
    assert x is None, 'display test fail'
    x = src2.display(canvas=fig)
    assert x is None, 'display test fail'
    x = src3.display(canvas=fig)
    assert x is None, 'display test fail'
    x = src4.display(canvas=fig, path=2)
    assert x is None, 'display test fail'

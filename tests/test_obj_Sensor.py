import numpy as np
import magpylib as magpy


def test_sensor1():
    """ self-consistent test of the sensor class
    """
    pm = magpy.magnet.Cuboid((11,22,33),(1,2,3))
    angs = np.linspace(0,555,44)
    possis = [(3*np.cos(t/180*np.pi),3*np.sin(t/180*np.pi),1) for t in angs]
    sens = magpy.Sensor()
    sens.move((3,0,1))
    sens.rotate_from_angax(angs, 'z', start=0, anchor=0)
    sens.rotate_from_angax(-angs, 'z', start=0)

    B1 = pm.getB(possis)
    B2 = sens.getB(pm)

    assert B1.shape==B2.shape, 'FAIL sensor shape'
    assert np.allclose(B1,B2), 'FAIL sensor values'


def test_sensor2():
    """ self-consistent test of the sensor class
    """
    pm = magpy.magnet.Cuboid((11,22,33),(1,2,3))
    poz = np.linspace(0,5,33)
    poss1 = [(t,0,2) for t in poz]
    poss2 = [(t,0,3) for t in poz]
    poss3 = [(t,0,4) for t in poz]
    B1 = np.array([pm.getB(poss) for poss in [poss1,poss2,poss3]])
    B1 = np.swapaxes(B1,0,1)

    sens = magpy.Sensor(pixel=[(0,0,2),(0,0,3),(0,0,4)])
    sens.move([(t,0,0) for t in poz], start=0)
    B2 = sens.getB(pm)

    assert B1.shape==B2.shape, 'FAIL sensor shape'
    assert np.allclose(B1,B2), 'FAIL sensor values'


def test_Sensor_getB_specs():
    """ test input of sens getB
    """
    sens1 = magpy.Sensor(pixel=(4,4,4))
    pm1 = magpy.magnet.Cylinder((111,222,333),(1,2))

    B1 = sens1.getB(pm1)
    B2 = magpy.getB(pm1,sens1)
    assert np.allclose(B1,B2), 'should be same'


def test_Sensor_squeeze():
    """ testing squeeze output
    """
    src = magpy.magnet.Sphere((1,1,1),1)
    sensor = magpy.Sensor(pixel=[(1,2,3),(1,2,3)])
    B = sensor.getB(src)
    assert B.shape==(2,3)
    H = sensor.getH(src)
    assert H.shape==(2,3)

    B = sensor.getB(src,squeeze=False)
    assert B.shape==(1,1,1,2,3)
    H = sensor.getH(src,squeeze=False)
    assert H.shape==(1,1,1,2,3)


def test_repr():
    """ test __repr__
    """
    sens = magpy.Sensor()
    assert sens.__repr__()[:6]== 'Sensor', 'Sensor repr failed'

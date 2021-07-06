import magpylib as mag3
from magpylib.magnet import Cylinder


def test_Cylinder_add():
    """ testing __add__
    """
    src1 = Cylinder((1,2,3),(1,2))
    src2 = Cylinder((1,2,3),(1,2))
    col = src1 + src2
    assert isinstance(col,mag3.Collection), 'adding cylinder fail'


def test_Cylinder_squeeze():
    """ testing squeeze output
    """
    src1 = Cylinder((1,1,1),(1,1))
    sensor = mag3.Sensor(pixel=[(1,2,3),(1,2,3)])
    B = src1.getB(sensor)
    assert B.shape==(2,3)
    H = src1.getH(sensor)
    assert H.shape==(2,3)

    B = src1.getB(sensor,squeeze=False)
    assert B.shape==(1,1,1,2,3)
    H = src1.getH(sensor,squeeze=False)
    assert H.shape==(1,1,1,2,3)


def test_repr():
    """ test __repr__
    """
    pm2 = mag3.magnet.Cylinder((1,2,3),(2,3))
    assert pm2.__repr__()[:8] == 'Cylinder', 'Cylinder repr failed'

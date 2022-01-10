import numpy as np
import magpylib as magpy
from magpylib.magnet import Cylinder
from magpylib._src.fields.field_wrap_BH_level2_dict import getB_dict, getH_dict


def test_Cylinder_add():
    """ testing __add__
    """
    src1 = Cylinder((1,2,3),(1,2))
    src2 = Cylinder((1,2,3),(1,2))
    col = src1 + src2
    assert isinstance(col,magpy.Collection), 'adding cylinder fail'


def test_Cylinder_squeeze():
    """ testing squeeze output
    """
    src1 = Cylinder((1,1,1),(1,1))
    sensor = magpy.Sensor(pixel=[(1,2,3),(1,2,3)])
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
    pm2 = magpy.magnet.Cylinder((1,2,3),(2,3))
    assert pm2.__repr__()[:8] == 'Cylinder', 'Cylinder repr failed'

def test_repr2():
    """ test __repr__
    """
    pm2 = magpy.magnet.CylinderSegment((1,2,3),(2,3,1,0,45))
    assert pm2.__repr__()[:15] == 'CylinderSegment', 'CylinderSegment repr failed'


def test_Cylinder_getBH():
    """
    test Cylinder geB and getH with diffeent inputs
    vs the vectorized form
    """
    mag = (22,33,44)
    poso = [
        (.123,.234,.345),
        (-.123,.234,.345),
        (.123,-.234,.345),
        (.123,.234,-.345),
        (-.123,-.234,.345),
        (-.123,.234,-.345),
        (.123,-.234,-.345),
        (-.123,-.234,-.345),
        (12,13,14),
        (-12,13,14),
        (12,-13,14),
        (12,13,-14),
        (-12,-13,14),
        (12,-13,-14),
        (-12,13,-14),
        (-12,-13,-14)]

    dim2 = [(1,2), (2,3), (3,4)]
    dim5 = [(0,.5,2,0,360), (0,1,3,0,360), (0,1.5,4,0,360)]

    for d2,d5 in zip(dim2,dim5):

        src1 = magpy.magnet.Cylinder(mag, d2)
        src2 = magpy.magnet.CylinderSegment(mag, d5)
        B0 = src1.getB(poso)
        H0 = src1.getH(poso)

        B1 = src2.getB(poso)
        H1 = src2.getH(poso)

        B2 = getB_dict(
            source_type='Cylinder',
            magnetization=mag,
            dimension=d2,
            observer=poso)
        H2 = getH_dict(
            source_type='Cylinder',
            magnetization=mag,
            dimension=d2,
            observer=poso)

        B3 = getB_dict(
            source_type='CylinderSegment',
            magnetization=mag,
            dimension=d5,
            observer=poso)
        H3 = getH_dict(
            source_type='CylinderSegment',
            magnetization=mag,
            dimension=d5,
            observer=poso)

        assert np.allclose(B1, B2)
        assert np.allclose(B1, B3)
        assert np.allclose(B1, B0)

        assert np.allclose(H1, H2)
        assert np.allclose(H1, H3)
        assert np.allclose(H1, H0)

import numpy as np

import magpylib as magpy


def test_Cylinder_add():
    """testing __add__"""
    src1 = magpy.magnet.Cylinder((1, 2, 3), (1, 2))
    src2 = magpy.magnet.Cylinder((1, 2, 3), (1, 2))
    col = src1 + src2
    assert isinstance(col, magpy.Collection), "adding cylinder fail"


def test_Cylinder_squeeze():
    """testing squeeze output"""
    src1 = magpy.magnet.Cylinder((1, 1, 1), (1, 1))
    sensor = magpy.Sensor(pixel=[(1, 2, 3), (1, 2, 3)])
    B = src1.getB(sensor)
    assert B.shape == (2, 3)
    H = src1.getH(sensor)
    assert H.shape == (2, 3)

    B = src1.getB(sensor, squeeze=False)
    assert B.shape == (1, 1, 1, 2, 3)
    H = src1.getH(sensor, squeeze=False)
    assert H.shape == (1, 1, 1, 2, 3)


def test_repr():
    """test __repr__"""
    pm2 = magpy.magnet.Cylinder((1, 2, 3), (2, 3))
    assert repr(pm2)[:8] == "Cylinder", "Cylinder repr failed"


def test_repr2():
    """test __repr__"""
    pm2 = magpy.magnet.CylinderSegment((1, 2, 3), (2, 3, 1, 0, 45))
    assert repr(pm2)[:15] == "CylinderSegment", "CylinderSegment repr failed"


def test_Cylinder_getBH():
    """
    test Cylinder getB and getH with different inputs
    vs the vectorized form
    """
    mag = (22, 33, 44)
    poso = [
        (0.123, 0.234, 0.345),
        (-0.123, 0.234, 0.345),
        (0.123, -0.234, 0.345),
        (0.123, 0.234, -0.345),
        (-0.123, -0.234, 0.345),
        (-0.123, 0.234, -0.345),
        (0.123, -0.234, -0.345),
        (-0.123, -0.234, -0.345),
        (12, 13, 14),
        (-12, 13, 14),
        (12, -13, 14),
        (12, 13, -14),
        (-12, -13, 14),
        (12, -13, -14),
        (-12, 13, -14),
        (-12, -13, -14),
    ]

    dim2 = [(1, 2), (2, 3), (3, 4)]
    dim5 = [(0, 0.5, 2, 0, 360), (0, 1, 3, 0, 360), (0.0000001, 1.5, 4, 0, 360)]

    for d2, d5 in zip(dim2, dim5):
        src1 = magpy.magnet.Cylinder(mag, d2)
        src2 = magpy.magnet.CylinderSegment(mag, d5)
        B0 = src1.getB(poso)
        H0 = src1.getH(poso)

        B1 = src2.getB(poso)
        H1 = src2.getH(poso)

        B2 = magpy.getB(
            "Cylinder",
            poso,
            magnetization=mag,
            dimension=d2,
        )
        H2 = magpy.getH(
            "Cylinder",
            poso,
            magnetization=mag,
            dimension=d2,
        )

        B3 = magpy.getB(
            "CylinderSegment",
            poso,
            magnetization=mag,
            dimension=d5,
        )
        H3 = magpy.getH(
            "CylinderSegment",
            poso,
            magnetization=mag,
            dimension=d5,
        )

        assert np.allclose(B1, B2)
        assert np.allclose(B1, B3)
        assert np.allclose(B1, B0)

        assert np.allclose(H1, H2)
        assert np.allclose(H1, H3)
        assert np.allclose(H1, H0)

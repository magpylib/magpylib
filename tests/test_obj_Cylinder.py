import numpy as np

import magpylib as magpy


def test_Cylinder_add():
    """testing __add__"""
    src1 = magpy.magnet.Cylinder(polarization=(1, 2, 3), dimension=(1, 2))
    src2 = magpy.magnet.Cylinder(polarization=(1, 2, 3), dimension=(1, 2))
    col = src1 + src2
    assert isinstance(col, magpy.Collection), "adding cylinder fail"


def test_Cylinder_squeeze():
    """testing squeeze output"""
    src1 = magpy.magnet.Cylinder(polarization=(1, 1, 1), dimension=(1, 1))
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
    pm2 = magpy.magnet.Cylinder(polarization=(1, 2, 3), dimension=(2, 3))
    assert repr(pm2)[:8] == "Cylinder", "Cylinder repr failed"


def test_repr2():
    """test __repr__"""
    pm2 = magpy.magnet.CylinderSegment(
        polarization=(1, 2, 3), dimension=(2, 3, 1, 0, 45)
    )
    assert repr(pm2)[:15] == "CylinderSegment", "CylinderSegment repr failed"


def test_Cylinder_getBH():
    """
    test Cylinder getB and getH with different inputs
    vs the vectorized form
    """
    pol = (22, 33, 44)
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
        src1 = magpy.magnet.Cylinder(polarization=pol, dimension=d2)
        src2 = magpy.magnet.CylinderSegment(polarization=pol, dimension=d5)
        B0 = src1.getB(poso)
        H0 = src1.getH(poso)

        B1 = src2.getB(poso)
        H1 = src2.getH(poso)

        B2 = magpy.getB(
            "Cylinder",
            poso,
            polarization=pol,
            dimension=d2,
        )
        H2 = magpy.getH(
            "Cylinder",
            poso,
            polarization=pol,
            dimension=d2,
        )

        B3 = magpy.getB(
            "CylinderSegment",
            poso,
            polarization=pol,
            dimension=d5,
        )
        H3 = magpy.getH(
            "CylinderSegment",
            poso,
            polarization=pol,
            dimension=d5,
        )

        np.testing.assert_allclose(B1, B2)
        np.testing.assert_allclose(B1, B3)
        np.testing.assert_allclose(B1, B0)

        np.testing.assert_allclose(H1, H2)
        np.testing.assert_allclose(H1, H3)
        np.testing.assert_allclose(H1, H0)


def test_getM():
    """getM test"""
    m0 = (0, 0, 0)
    m1 = (10, 200, 3000)
    cyl = magpy.magnet.Cylinder(dimension=(2, 2), magnetization=m1)
    obs = [
        (2, 2, 2),
        (0, 0, 0),
        (0.5, 0.5, 0.5),
        (3, 0, 0),
    ]
    sens = magpy.Sensor(pixel=obs)

    M1 = cyl.getM(obs)
    M2 = magpy.getM(cyl, sens)
    M3 = sens.getM(cyl)

    Mtest = np.array([m0, m1, m1, m0])

    np.testing.assert_allclose(M1, Mtest)
    np.testing.assert_allclose(M2, Mtest)
    np.testing.assert_allclose(M3, Mtest)


def test_getJ():
    """getM test"""
    j0 = (0, 0, 0)
    j1 = (0.1, 0.2, 0.3)
    cyl = magpy.magnet.Cylinder(
        dimension=(2, 2),
        polarization=j1,
    )
    obs = [
        (-2, 2, -2),
        (0, 0, 0),
        (-0.5, -0.5, 0.5),
        (-3, 0, 0),
    ]
    sens = magpy.Sensor(pixel=obs)

    J1 = cyl.getJ(obs)
    J2 = magpy.getJ(cyl, sens)
    J3 = sens.getJ(cyl)
    Jtest = np.array([j0, j1, j1, j0])

    np.testing.assert_allclose(J1, Jtest)
    np.testing.assert_allclose(J2, Jtest)
    np.testing.assert_allclose(J3, Jtest)

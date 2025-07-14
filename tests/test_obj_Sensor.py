import numpy as np

import magpylib as magpy


def test_sensor1():
    """self-consistent test of the sensor class"""
    pm = magpy.magnet.Cuboid(polarization=(11, 22, 33), dimension=(1, 2, 3))
    angs = np.linspace(0, 555, 44)
    possis = [
        (3 * np.cos(t / 180 * np.pi), 3 * np.sin(t / 180 * np.pi), 1) for t in angs
    ]
    sens = magpy.Sensor()
    sens.move((3, 0, 1))
    sens.rotate_from_angax(angs, "z", start=0, anchor=0)
    sens.rotate_from_angax(-angs, "z", start=0)

    B1 = pm.getB(possis)
    B2 = sens.getB(pm)

    np.testing.assert_allclose(B1, B2)


def test_sensor2():
    """self-consistent test of the sensor class"""
    pm = magpy.magnet.Cuboid(polarization=(11, 22, 33), dimension=(1, 2, 3))
    poz = np.linspace(0, 5, 33)
    poss1 = [(t, 0, 2) for t in poz]
    poss2 = [(t, 0, 3) for t in poz]
    poss3 = [(t, 0, 4) for t in poz]
    B1 = np.array([pm.getB(poss) for poss in [poss1, poss2, poss3]])
    B1 = np.swapaxes(B1, 0, 1)

    sens = magpy.Sensor(pixel=[(0, 0, 2), (0, 0, 3), (0, 0, 4)])
    sens.move([(t, 0, 0) for t in poz], start=0)
    B2 = sens.getB(pm)

    np.testing.assert_allclose(B1, B2)


def test_Sensor_getB_specs():
    """test input of sens getB"""
    sens1 = magpy.Sensor(pixel=(4, 4, 4))
    pm1 = magpy.magnet.Cylinder(polarization=(111, 222, 333), dimension=(1, 2))

    B1 = sens1.getB(pm1)
    B2 = magpy.getB(pm1, sens1)
    np.testing.assert_allclose(B1, B2)


def test_Sensor_squeeze():
    """testing squeeze output"""
    src = magpy.magnet.Sphere(polarization=(1, 1, 1), diameter=1)
    sensor = magpy.Sensor(pixel=[(1, 2, 3), (1, 2, 3)])
    B = sensor.getB(src)
    assert B.shape == (2, 3)
    H = sensor.getH(src)
    assert H.shape == (2, 3)

    B = sensor.getB(src, squeeze=False)
    assert B.shape == (1, 1, 1, 2, 3)
    H = sensor.getH(src, squeeze=False)
    assert H.shape == (1, 1, 1, 2, 3)


def test_repr():
    """test __repr__"""
    sens = magpy.Sensor()
    assert repr(sens)[:6] == "Sensor", "Sensor repr failed"


def test_pixel1():
    """
    squeeze=False Bfield minimal shape is (1,1,1,1,3)
    logic: single sensor, scalar path, single source all generate
    1 for squeeze=False Bshape. Bare pixel should do the same
    """
    src = magpy.misc.Dipole(moment=(1, 2, 3))

    # squeeze=False Bshape of nbare pixel must be (1,1,1,1,3)
    np.testing.assert_allclose(
        src.getB(magpy.Sensor(pixel=(1, 2, 3)), squeeze=False).shape,
        (1, 1, 1, 1, 3),
    )

    # squeeze=False Bshape of [(1,2,3)] must then also be (1,1,1,1,3)
    src = magpy.misc.Dipole(moment=(1, 2, 3))
    np.testing.assert_allclose(
        src.getB(magpy.Sensor(pixel=[(1, 2, 3)]), squeeze=False).shape,
        (1, 1, 1, 1, 3),
    )

    # squeeze=False Bshape of [[(1,2,3)]] must be (1,1,1,1,1,3)
    np.testing.assert_allclose(
        src.getB(magpy.Sensor(pixel=[[(1, 2, 3)]]), squeeze=False).shape,
        (1, 1, 1, 1, 1, 3),
    )


def test_pixel2():
    """
    Sensor(pixel=pos_vec).pixel should always return pos_vec
    """

    p0 = (1, 2, 3)
    p1 = [(1, 2, 3)]
    p2 = [[(1, 2, 3)]]

    # input pos_vec == Sensor(pixel=pos_vec)
    for pos_vec in [p0, p1, p2]:
        np.testing.assert_allclose(
            magpy.Sensor(pixel=pos_vec).pixel,
            pos_vec,
        )


def test_pixel3():
    """
    There should be complete equivalence between pos_vec and
    Sensor(pixel=pos_vec) inputs
    """
    src = magpy.misc.Dipole(moment=(1, 2, 3))

    p0 = (1, 2, 3)
    p1 = [(1, 2, 3)]
    p2 = [[(1, 2, 3)]]
    for pos_vec in [p0, p1, p2]:
        np.testing.assert_allclose(
            src.getB(magpy.Sensor(pixel=pos_vec), squeeze=False),
            src.getB(pos_vec, squeeze=False),
        )

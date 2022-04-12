import numpy as np

import magpylib as magpy


def test_getB_interfaces1():
    """self-consistent test of different possibilities for computing the field"""
    src = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    src.move(np.linspace((0.1, 0.2, 0.3), (1, 2, 3), 10), start=-1)
    poso = [[(-1, -1, -1)] * 2] * 2
    sens = magpy.Sensor(pixel=poso)
    B = magpy.getB(
        "Cuboid",
        (-1, -1, -1),
        position=src.position,
        magnetization=(1, 2, 3),
        dimension=(1, 2, 3),
    )
    B1 = np.tile(B, (2, 2, 1, 1))
    B1 = np.swapaxes(B1, 0, 2)

    B_test = magpy.getB(src, sens)
    np.testing.assert_allclose(B1, B_test)

    B_test = src.getB(poso)
    np.testing.assert_allclose(B1, B_test)

    B_test = src.getB(sens)
    np.testing.assert_allclose(B1, B_test)

    B_test = sens.getB(src)


def test_getB_interfaces2():
    """self-consistent test of different possibilities for computing the field"""
    src = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    src.move(np.linspace((0.1, 0.2, 0.3), (1, 2, 3), 10), start=-1)
    poso = [[(-1, -1, -1)] * 2] * 2
    sens = magpy.Sensor(pixel=poso)
    B = magpy.getB(
        "Cuboid",
        (-1, -1, -1),
        position=src.position,
        magnetization=(1, 2, 3),
        dimension=(1, 2, 3),
    )

    B2 = np.tile(B, (2, 2, 2, 1, 1))
    B2 = np.swapaxes(B2, 1, 3)

    B_test = magpy.getB([src, src], sens)
    np.testing.assert_allclose(B2, B_test)

    B_test = sens.getB([src, src])
    np.testing.assert_allclose(B2, B_test)


def test_getB_interfaces3():
    """self-consistent test of different possibilities for computing the field"""
    src = magpy.magnet.Cuboid((1, 2, 3), (1, 2, 3))
    src.move(np.linspace((0.1, 0.2, 0.3), (1, 2, 3), 10), start=-1)
    poso = [[(-1, -1, -1)] * 2] * 2
    sens = magpy.Sensor(pixel=poso)
    B = magpy.getB(
        "Cuboid",
        (-1, -1, -1),
        position=src.position,
        magnetization=(1, 2, 3),
        dimension=(1, 2, 3),
    )

    B3 = np.tile(B, (2, 2, 2, 1, 1))
    B3 = np.swapaxes(B3, 0, 3)

    B_test = magpy.getB(src, [sens, sens])
    np.testing.assert_allclose(B3, B_test)

    B_test = src.getB([poso, poso])
    np.testing.assert_allclose(B3, B_test)

    B_test = src.getB([sens, sens])
    np.testing.assert_allclose(B3, B_test)


def test_getH_interfaces1():
    """self-consistent test of different possibilities for computing the field"""
    mag = (22, -33, 44)
    dim = (3, 2, 3)
    src = magpy.magnet.Cuboid(mag, dim)
    src.move(np.linspace((0.1, 0.2, 0.3), (1, 2, 3), 10), start=-1)

    poso = [[(-1, -2, -3)] * 2] * 2
    sens = magpy.Sensor(pixel=poso)

    H = magpy.getH(
        "Cuboid",
        (-1, -2, -3),
        position=src.position,
        magnetization=mag,
        dimension=dim,
    )
    H1 = np.tile(H, (2, 2, 1, 1))
    H1 = np.swapaxes(H1, 0, 2)

    H_test = magpy.getH(src, sens)
    np.testing.assert_allclose(H1, H_test)

    H_test = src.getH(poso)
    np.testing.assert_allclose(H1, H_test)

    H_test = src.getH(sens)
    np.testing.assert_allclose(H1, H_test)

    H_test = sens.getH(src)
    np.testing.assert_allclose(H1, H_test)


def test_getH_interfaces2():
    """self-consistent test of different possibilities for computing the field"""
    mag = (22, -33, 44)
    dim = (3, 2, 3)
    src = magpy.magnet.Cuboid(mag, dim)
    src.move(np.linspace((0.1, 0.2, 0.3), (1, 2, 3), 10), start=-1)

    poso = [[(-1, -2, -3)] * 2] * 2
    sens = magpy.Sensor(pixel=poso)

    H = magpy.getH(
        "Cuboid",
        (-1, -2, -3),
        position=src.position,
        magnetization=mag,
        dimension=dim,
    )

    H2 = np.tile(H, (2, 2, 2, 1, 1))
    H2 = np.swapaxes(H2, 1, 3)

    H_test = magpy.getH([src, src], sens)
    np.testing.assert_allclose(H2, H_test)

    H_test = sens.getH([src, src])
    np.testing.assert_allclose(H2, H_test)


def test_getH_interfaces3():
    """self-consistent test of different possibilities for computing the field"""
    mag = (22, -33, 44)
    dim = (3, 2, 3)
    src = magpy.magnet.Cuboid(mag, dim)
    src.move(np.linspace((0.1, 0.2, 0.3), (1, 2, 3), 10), start=-1)

    poso = [[(-1, -2, -3)] * 2] * 2
    sens = magpy.Sensor(pixel=poso)

    H = magpy.getH(
        "Cuboid",
        (-1, -2, -3),
        position=src.position,
        magnetization=mag,
        dimension=dim,
    )

    H3 = np.tile(H, (2, 2, 2, 1, 1))
    H3 = np.swapaxes(H3, 0, 3)

    H_test = magpy.getH(src, [sens, sens])
    np.testing.assert_allclose(H3, H_test)

    H_test = src.getH([poso, poso])
    np.testing.assert_allclose(H3, H_test)

    H_test = src.getH([sens, sens])
    np.testing.assert_allclose(H3, H_test)

import unittest
import warnings

import numpy as np
import pytest

import magpylib as magpy
from magpylib._src.exceptions import MagpylibBadUserInput


def test_getB_level2_input_simple():
    """test functionality of getB_level2 to combine various
    inputs - simple position inputs
    """
    mag = (1, 2, 3)
    dim_cuboid = (1, 2, 3)
    dim_cyl = (1, 2)
    pm1 = magpy.magnet.Cuboid(polarization=mag, dimension=dim_cuboid)
    pm2 = magpy.magnet.Cuboid(polarization=mag, dimension=dim_cuboid)
    pm3 = magpy.magnet.Cylinder(polarization=mag, dimension=dim_cyl)
    pm4 = magpy.magnet.Cylinder(polarization=mag, dimension=dim_cyl)
    col1 = magpy.Collection(pm1.copy())
    col2 = magpy.Collection(pm1.copy(), pm2.copy())
    col3 = magpy.Collection(pm1.copy(), pm2.copy(), pm3.copy())
    col4 = magpy.Collection(pm1.copy(), pm2.copy(), pm3.copy(), pm4.copy())
    pos_obs = (1, 2, 3)
    sens1 = magpy.Sensor(position=pos_obs)
    sens2 = magpy.Sensor(pixel=pos_obs)
    sens3 = magpy.Sensor(position=(1, 2, 0), pixel=(0, 0, 3))

    fb1 = magpy.getB(pm1, pos_obs)
    fc1 = magpy.getB(pm3, pos_obs)
    fb2 = np.array([fb1, fb1])
    fc2 = np.array([fc1, fc1])

    for poso, fb, fc in zip(
        [pos_obs, sens1, sens2, sens3, [sens1, sens2]],
        [fb1, fb1, fb1, fb1, fb2],
        [fc1, fc1, fc1, fc1, fc2],
    ):
        src_obs_res = [
            [pm1, poso, fb],
            [pm3, poso, fc],
            [[pm1, pm2], poso, [fb, fb]],
            [[pm1, pm2, pm3], poso, [fb, fb, fc]],
            [col1, poso, fb],
            [col2, poso, 2 * fb],
            [col3, poso, 2 * fb + fc],
            [col4, poso, 2 * fb + 2 * fc],
            [[pm1, col1], poso, [fb, fb]],
            [[pm1, col1, col2, pm2, col4], poso, [fb, fb, 2 * fb, fb, 2 * fb + 2 * fc]],
        ]

        for sor in src_obs_res:
            sources, observers, result = sor
            result = np.array(result)

            B = magpy.getB(sources, observers)
            np.testing.assert_allclose(B, result)


def test_getB_level2_input_shape22():
    """test functionality of getB_level2 to combine various
    inputs - position input with shape (2,2)
    """
    mag = (1, 2, 3)
    dim_cuboid = (1, 2, 3)
    dim_cyl = (1, 2)

    def pm1():
        return magpy.magnet.Cuboid(polarization=mag, dimension=dim_cuboid)

    def pm2():
        return magpy.magnet.Cuboid(polarization=mag, dimension=dim_cuboid)

    def pm3():
        return magpy.magnet.Cylinder(polarization=mag, dimension=dim_cyl)

    def pm4():
        return magpy.magnet.Cylinder(polarization=mag, dimension=dim_cyl)

    col1 = magpy.Collection(pm1())
    col2 = magpy.Collection(pm1(), pm2())
    col3 = magpy.Collection(pm1(), pm2(), pm3())
    col4 = magpy.Collection(pm1(), pm2(), pm3(), pm4())
    pos_obs = [[(1, 2, 3), (1, 2, 3)], [(1, 2, 3), (1, 2, 3)]]
    sens1 = magpy.Sensor(pixel=pos_obs)

    fb22 = magpy.getB(pm1(), pos_obs)
    fc22 = magpy.getB(pm3(), pos_obs)

    for poso, fb, fc in zip(
        [pos_obs, sens1, [sens1, sens1, sens1]],
        [fb22, fb22, [fb22, fb22, fb22]],
        [fc22, fc22, [fc22, fc22, fc22]],
    ):
        fb = np.array(fb)
        fc = np.array(fc)
        src_obs_res = [
            [pm1(), poso, fb],
            [pm3(), poso, fc],
            [[pm1(), pm2()], poso, [fb, fb]],
            [[pm1(), pm2(), pm3()], poso, [fb, fb, fc]],
            [col1, poso, fb],
            [col2, poso, 2 * fb],
            [col3, poso, 2 * fb + fc],
            [col4, poso, 2 * fb + 2 * fc],
            [[pm1(), col1], poso, [fb, fb]],
            [
                [pm1(), col1, col2, pm2(), col4],
                poso,
                [fb, fb, 2 * fb, fb, 2 * fb + 2 * fc],
            ],
        ]

        for sor in src_obs_res:
            sources, observers, result = sor
            result = np.array(result)
            B = magpy.getB(sources, observers)
            np.testing.assert_allclose(B, result)


def test_getB_level2_input_path():
    """test functionality of getB_level2 to combine various
    inputs - input objects with path
    """
    mag = (1, 2, 3)
    dim_cuboid = (1, 2, 3)
    pm1 = magpy.magnet.Cuboid(polarization=mag, dimension=dim_cuboid)
    pm2 = magpy.magnet.Cuboid(polarization=mag, dimension=dim_cuboid)
    sens1 = magpy.Sensor()
    sens2 = magpy.Sensor(pixel=[(0, 0, 0), (0, 0, 1), (0, 0, 2)])

    fb = pm1.getB([(x, 0, 0) for x in np.linspace(0, -1, 11)])

    possis = np.linspace((0.1, 0, 0), (1, 0, 0), 10)
    pm1.move(possis)
    B = magpy.getB(pm1, (0, 0, 0))
    result = fb
    np.testing.assert_allclose(B, result)

    B = magpy.getB(pm1, sens1)
    result = fb
    np.testing.assert_allclose(B, result)

    B = magpy.getB([pm1, pm1], sens1)
    result = np.array([fb, fb])
    np.testing.assert_allclose(B, result)

    fb = pm2.getB([[(x, 0, 0), (x, 0, 0)] for x in np.linspace(0, -1, 11)])
    B = magpy.getB([pm1, pm1], [sens1, sens1])
    result = np.array([fb, fb])
    np.testing.assert_allclose(B, result)

    fb = pm2.getB(
        [[[(x, 0, 0), (x, 0, 1), (x, 0, 2)]] * 2 for x in np.linspace(0, -1, 11)]
    )
    B = magpy.getB([pm1, pm1], [sens2, sens2])
    result = np.array([fb, fb])
    np.testing.assert_allclose(B, result)


def test_path_tile():
    """Test if auto-tiled paths of objects will properly be reset
    in getB_level2 before returning
    """
    pm1 = magpy.magnet.Cuboid(polarization=(11, 22, 33), dimension=(1, 2, 3))
    pm2 = magpy.magnet.Cuboid(polarization=(11, 22, 33), dimension=(1, 2, 3))
    poz = np.linspace((10 / 33, 10 / 33, 10 / 33), (10, 10, 10), 33)
    pm2.move(poz)

    path1p = pm1.position
    path1r = pm1.orientation

    path2p = pm2.position
    path2r = pm2.orientation

    _ = magpy.getB([pm1, pm2], [0, 0, 0])

    np.testing.assert_array_equal(
        path1p,
        pm1.position,
        err_msg="FAILED: getB modified object path",
    )
    np.testing.assert_array_equal(
        path1r.as_quat(),
        pm1.orientation.as_quat(),
        err_msg="FAILED: getB modified object path",
    )
    np.testing.assert_array_equal(
        path2p,
        pm2.position,
        err_msg="FAILED: getB modified object path",
    )
    np.testing.assert_array_equal(
        path2r.as_quat(),
        pm2.orientation.as_quat(),
        err_msg="FAILED: getB modified object path",
    )


def test_sensor_rotation1():
    """Test simple sensor rotation using sin/cos"""
    src = magpy.magnet.Cuboid(polarization=(1, 0, 0), dimension=(1, 1, 1))
    sens = magpy.Sensor(position=(1, 0, 0))
    sens.rotate_from_angax(np.linspace(0, 360, 56)[1:], "z", start=1, anchor=None)
    B = src.getB(sens)

    B0 = B[0, 0]
    Brot = np.array(
        [
            (B0 * np.cos(phi), -B0 * np.sin(phi), 0)
            for phi in np.linspace(0, 2 * np.pi, 56)
        ]
    )

    np.testing.assert_allclose(B, Brot, rtol=1e-05, atol=1e-08)


def test_sensor_rotation2():
    """test sensor rotations with different combinations of inputs mag/col + sens/pos"""
    src = magpy.magnet.Cuboid(
        polarization=(1, 0, 0), dimension=(1, 1, 1), position=(0, 0, 2)
    )
    src2 = magpy.magnet.Cuboid(
        polarization=(1, 0, 0), dimension=(1, 1, 1), position=(0, 0, 2)
    )
    col = magpy.Collection(src, src2)

    poss = (0, 0, 0)
    sens = magpy.Sensor(pixel=poss)
    sens.rotate_from_angax([45, 90], "z")

    sens2 = magpy.Sensor(pixel=poss)
    sens2.rotate_from_angax(-45, "z")

    x1 = np.array([-9.82, 0, 0]) * 1e-3
    x2 = np.array([-6.94, 6.94, 0]) * 1e-3
    x3 = np.array([0, 9.82, 0]) * 1e-3
    x1b = np.array([-19.64, 0, 0]) * 1e-3
    x2b = np.array([-13.89, 13.89, 0]) * 1e-3
    x3b = np.array([0, 19.64, 0]) * 1e-3

    B = magpy.getB(src, poss, squeeze=True)
    Btest = x1
    np.testing.assert_allclose(
        B,
        Btest,
        rtol=1e-4,
        atol=1e-5,
        err_msg="FAIL: mag  +  pos",
    )

    B = magpy.getB([src], [sens], squeeze=True)
    Btest = np.array([x1, x2, x3])
    np.testing.assert_allclose(
        B,
        Btest,
        rtol=1e-4,
        atol=1e-5,
        err_msg="FAIL: mag  +  sens_rot_path",
    )

    B = magpy.getB([src], [sens, poss], squeeze=True)
    Btest = np.array([[x1, x1], [x2, x1], [x3, x1]])
    np.testing.assert_allclose(
        B,
        Btest,
        rtol=1e-4,
        atol=1e-5,
        err_msg="FAIL: mag  +  sens_rot_path, pos",
    )

    B = magpy.getB([src, col], [sens, poss], squeeze=True)
    Btest = np.array(
        [[[x1, x1], [x2, x1], [x3, x1]], [[x1b, x1b], [x2b, x1b], [x3b, x1b]]]
    )
    np.testing.assert_allclose(
        B,
        Btest,
        rtol=1e-4,
        atol=1e-5,
        err_msg="FAIL: mag,col  +  sens_rot_path, pos",
    )


def test_sensor_rotation3():
    """testing rotated static sensor path"""
    # case static sensor rot
    src = magpy.magnet.Cuboid(polarization=(1, 0, 0), dimension=(1, 1, 1))
    sens = magpy.Sensor()
    sens.rotate_from_angax(45, "z")
    B0 = magpy.getB(src, sens)
    B0t = np.tile(B0, (12, 1))

    sens.move([(0, 0, 0)] * 12, start=-1)
    Bpath = magpy.getB(src, sens)

    np.testing.assert_allclose(B0t, Bpath)


def test_object_tiling():
    """test if object tiling works when input paths are of various lengths"""
    # pylint: disable=no-member
    src1 = magpy.current.Circle(current=1, diameter=1)
    src1.rotate_from_angax(np.linspace(1, 31, 31), "x", anchor=(0, 1, 0), start=-1)

    src2 = magpy.magnet.Cuboid(
        polarization=(1, 1, 1), dimension=(1, 1, 1), position=(1, 1, 1)
    )
    src2.move([(1, 1, 1)] * 21, start=-1)

    src3 = magpy.magnet.Cuboid(
        polarization=(1, 1, 1), dimension=(1, 1, 1), position=(1, 1, 1)
    )
    src4 = magpy.magnet.Cuboid(
        polarization=(1, 1, 1), dimension=(1, 1, 1), position=(1, 1, 1)
    )

    col = magpy.Collection(src3, src4)
    src3.move([(1, 1, 1)] * 12, start=-1)
    src4.move([(1, 1, 1)] * 31, start=-1)

    possis = [[1, 2, 3]] * 5
    sens = magpy.Sensor(pixel=possis)

    assert src1.position.shape == (31, 3), "a1"
    assert src2.position.shape == (21, 3), "a2"
    assert src3.position.shape == (12, 3), "a3"
    assert src4.position.shape == (31, 3), "a4"
    assert sens.position.shape == (3,), "a5"

    assert src1.orientation.as_quat().shape == (31, 4), "b1"
    assert src2.orientation.as_quat().shape == (21, 4), "b2"
    assert src3.orientation.as_quat().shape == (12, 4), "b3"
    assert src4.orientation.as_quat().shape == (31, 4), "b4"
    assert sens.orientation.as_quat().shape == (4,), "b5"

    B = magpy.getB([src1, src2, col], [sens, possis])
    assert B.shape == (3, 31, 2, 5, 3)

    assert src1.position.shape == (31, 3), "c1"
    assert src2.position.shape == (21, 3), "c2"
    assert src3.position.shape == (12, 3), "c3"
    assert src4.position.shape == (31, 3), "c4"
    assert sens.position.shape == (3,), "c5"

    assert src1.orientation.as_quat().shape == (31, 4), "d1"
    assert src2.orientation.as_quat().shape == (21, 4), "d2"
    assert src3.orientation.as_quat().shape == (12, 4), "d3"
    assert src4.orientation.as_quat().shape == (31, 4), "d4"
    assert sens.orientation.as_quat().shape == (4,), "d5"


def test_superposition_vs_tiling():
    """test superposition vs tiling, see issue #507"""

    loop = magpy.current.Circle(current=10000, diameter=20, position=(1, 20, 10))
    loop.rotate_from_angax([45, 90], "x")

    sphere1 = magpy.magnet.Sphere(
        polarization=(0, 0, 1), diameter=1, position=(20, 10, 1)
    )
    sphere1.rotate_from_angax([45, 90], "y")

    sphere2 = magpy.magnet.Sphere(
        polarization=(1, 0, 0), diameter=2, position=(10, 20, 1)
    )
    sphere2.rotate_from_angax([45, 90], "y")

    loop_collection = magpy.Collection(loop, sphere1, sphere2)

    observer_positions = [[0, 0, 0], [1, 1, 1]]

    B1 = magpy.getB(loop, observer_positions)
    B2 = magpy.getB(sphere1, observer_positions)
    B3 = magpy.getB(sphere2, observer_positions)
    superposed_B = B1 + B2 + B3

    collection_B = magpy.getB(loop_collection, observer_positions)

    np.testing.assert_allclose(superposed_B, collection_B)


def test_squeeze_sumup():
    """make sure that sumup does not lead to false output shape"""

    s = magpy.Sensor(pixel=(1, 2, 3))
    ss = magpy.magnet.Sphere(polarization=(1, 2, 3), diameter=1)

    B1 = magpy.getB(ss, s, squeeze=False)
    B2 = magpy.getB(ss, s, squeeze=False, sumup=True)

    assert B1.shape == B2.shape


def test_pixel_agg():
    """test pixel aggregator"""
    src1 = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1)).move(
        [[1, 0, 0]]
    )
    sens1 = magpy.Sensor(
        position=(0, 0, 1), pixel=np.zeros((4, 5, 3)), style_label="sens1 pixel(4,5)"
    )
    sens2 = sens1.copy(position=(0, 0, 2), style_label="sens2 pixel(4,5)")
    sens3 = sens1.copy(position=(0, 0, 3), style_label="sens3 pixel(4,5)")
    sens_col = magpy.Collection(sens1, sens2, sens3)

    B1 = magpy.getB(src1, sens_col, squeeze=False, pixel_agg=None)
    np.testing.assert_array_equal(B1.shape, (1, 2, 3, 4, 5, 3))

    B2 = magpy.getB(src1, sens_col, squeeze=False, pixel_agg="mean")
    np.testing.assert_array_equal(B2.shape, (1, 2, 3, 1, 3))

    B3 = magpy.getB(src1, sens_col, squeeze=True, pixel_agg=None)
    np.testing.assert_array_equal(B3.shape, (2, 3, 4, 5, 3))

    B4 = magpy.getB(src1, sens_col, squeeze=True, pixel_agg="mean")
    np.testing.assert_array_equal(B4.shape, (2, 3, 3))


def test_pixel_agg_heterogeneous_pixel_shapes():
    """test pixel aggregator with heterogeneous pixel shapes"""
    src1 = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    src2 = magpy.magnet.Sphere(polarization=(0, 0, 1), diameter=1, position=(2, 0, 0))
    sens1 = magpy.Sensor(
        position=(0, 0, 1), pixel=[0, 0, 0], style_label="sens1, pixel.shape = (3,)"
    )
    sens2 = sens1.copy(
        position=(0, 0, 2), pixel=[1, 1, 1], style_label="sens2,  pixel.shape = (3,)"
    )
    sens3 = sens1.copy(
        position=(0, 0, 3), pixel=[2, 2, 2], style_label="sens3,  pixel.shape = (3,)"
    )
    sens4 = sens1.copy(style_label="sens4,  pixel.shape = (3,)")
    sens5 = sens2.copy(
        pixel=np.zeros((4, 5, 3)) + 1, style_label="sens5,  pixel.shape = (3,)"
    )
    sens6 = sens3.copy(
        pixel=np.zeros((4, 5, 1, 3)) + 2, style_label="sens6,  pixel.shape = (4,5,1,3)"
    )
    src_col = magpy.Collection(src1, src2)
    sens_col1 = magpy.Collection(sens1, sens2, sens3)
    sens_col2 = magpy.Collection(sens4, sens5, sens6)
    sens_col1.rotate_from_angax([45], "z", anchor=(5, 0, 0))
    sens_col2.rotate_from_angax([45], "z", anchor=(5, 0, 0))

    # different pixel shapes withoug pixel_agg should raise an error
    with pytest.raises(MagpylibBadUserInput):
        magpy.getB(src1, sens_col2, pixel_agg=None)

    # bad pixel_agg numpy reference
    with pytest.raises(AttributeError):
        magpy.getB(src1, sens_col2, pixel_agg="bad_aggregator")

    # good pixel_agg numpy reference, but non-reducing function
    with pytest.raises(AttributeError):
        magpy.getB(src1, sens_col2, pixel_agg="array")

    B1 = magpy.getB(src1, sens_col1, squeeze=False, pixel_agg="max")
    np.testing.assert_array_equal(B1.shape, (1, 2, 3, 1, 3))

    B2 = magpy.getB(src1, sens_col2, squeeze=False, pixel_agg="max")
    np.testing.assert_array_equal(B2.shape, (1, 2, 3, 1, 3))

    B3 = magpy.getB(src1, sens_col1, squeeze=True)
    np.testing.assert_array_equal(B3.shape, (2, 3, 3))

    B4 = magpy.getB(src1, sens_col2, squeeze=True, pixel_agg="mean")
    np.testing.assert_array_equal(B4.shape, (2, 3, 3))

    # B3 and B4 should deliver the same results since pixel all have the same
    # positions respectively for each sensor, so mean equals single value
    np.testing.assert_allclose(B3, B4)

    # Testing autmatic vs manual aggregation (mean) with different pixel shapes
    B_by_sens_agg_1 = magpy.getB(src_col, sens_col2, squeeze=False, pixel_agg="mean")
    B_by_sens_agg_2 = []
    for sens in sens_col2:
        B = magpy.getB(src_col, sens, squeeze=False)
        B = B.mean(axis=tuple(range(3 - B.ndim, -1)))
        B = np.expand_dims(B, axis=-2)
        B_by_sens_agg_2.append(B)
    B_by_sens_agg_2 = np.concatenate(B_by_sens_agg_2, axis=2)

    np.testing.assert_allclose(B_by_sens_agg_1, B_by_sens_agg_2)


def test_pixel_agg3():
    """test for various inputs"""
    B1 = np.array([0.03122074, 0.03122074, 0.03122074])

    e0 = np.array((1, 1, 1))
    e1 = [(1, 1, 1)]
    e2 = [(1, 1, 1)] * 2
    e3 = [(1, 1, 1)] * 3

    s0 = magpy.magnet.Cuboid(polarization=e0, dimension=e0)
    c0 = magpy.Collection(s0)
    s1 = magpy.magnet.Cuboid(polarization=e0, dimension=e0)
    s2 = magpy.magnet.Cuboid(polarization=-e0, dimension=e0)
    c1 = magpy.Collection(c0, s1, s2)

    x0 = magpy.Sensor(pixel=e0)
    x1 = magpy.Sensor(pixel=e1)
    x2 = magpy.Sensor(pixel=e2)
    x3 = magpy.Sensor(pixel=e3)

    c2 = x0 + x1 + x2 + x3

    for src, src_sh in zip([s0, c0, [s0, c0], c1, [s0, c0, c1, s1]], [1, 1, 2, 1, 4]):
        for obs, obs_sh in zip(
            [e0, e1, e2, e3, x0, x1, x2, x3, c2, [x0, x2, x3]], [1] * 8 + [4, 3]
        ):
            for px_agg in ["mean", "average", "min"]:
                np.testing.assert_allclose(
                    magpy.getB(src, obs, pixel_agg=px_agg),
                    np.squeeze(np.tile(B1, (src_sh, obs_sh, 1))),
                    rtol=1e-5,
                    atol=1e-8,
                )

    # same check with a path
    s0.position = [(0, 0, 0)] * 5
    for src, src_sh in zip([s0, c0, [s0, c0], c1, [s0, c0, c1, s1]], [1, 1, 2, 1, 4]):
        for obs, obs_sh in zip(
            [e0, e1, e2, e3, x0, x1, x2, x3, c2, [x0, x2, x3]], [1] * 8 + [4, 3]
        ):
            for px_agg in ["mean", "average", "min"]:
                np.testing.assert_allclose(
                    magpy.getB(src, obs, pixel_agg=px_agg),
                    np.squeeze(np.tile(B1, (src_sh, 5, obs_sh, 1))),
                    rtol=1e-5,
                    atol=1e-8,
                )


##############################################################
def warnme1():
    """test if in_out warning is thrown"""
    sp = magpy.magnet.Sphere(
        polarization=(1, 2, 3),
        diameter=1,
    )
    sp.getB((1, 1, 1), in_out="inside")


def warnme2():
    """test if in_out warning is thrown"""
    sp = magpy.magnet.Sphere(
        polarization=(1, 2, 3),
        diameter=1,
    )
    magpy.getH([sp, sp], (1, 1, 1), in_out="inside")


class TestExceptions(unittest.TestCase):
    """test class for exception testing"""

    def test_warning(self):
        """whatever"""
        self.assertWarns(UserWarning, warnme1)
        self.assertWarns(UserWarning, warnme2)


##############################################################


def do_not_warnme1():
    """test if in_out warning is thrown"""
    sp = magpy.magnet.Sphere(
        polarization=(1, 2, 3),
        diameter=1,
    )
    tetra = magpy.magnet.Tetrahedron(
        polarization=(1, 2, 3),
        vertices=[(1, 2, 3), (0, 0, 0), (1, 0, 0), (0, 1, 0)],
    )
    magpy.getH([sp, tetra], (1, 1, 1), in_out="inside")


def do_not_warnme2():
    """test if in_out warning is thrown"""
    tetra = magpy.magnet.Tetrahedron(
        polarization=(1, 2, 3),
        vertices=[(1, 2, 3), (0, 0, 0), (1, 0, 0), (0, 1, 0)],
    )
    magpy.getH(tetra, (1, 1, 1), in_out="inside")


def test_do_not_warn():
    """do not warn"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        do_not_warnme1()
        do_not_warnme2()
        if len(w) > 0:
            pytest.fail("WARNING SHOULD NOT HAVE BEEN RAISED")

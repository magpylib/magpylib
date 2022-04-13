import numpy as np
from numpy.testing import assert_allclose

from magpylib._src.fields.field_BH_cuboid import magnet_cuboid_field
from magpylib._src.fields.field_BH_cylinder_segment import magnet_cylinder_segment_field
from magpylib._src.fields.field_BH_dipole import dipole_field
from magpylib._src.fields.field_BH_line import current_line_field
from magpylib._src.fields.field_BH_line import field_BH_line_from_vert
from magpylib._src.fields.field_BH_loop import current_loop_field
from magpylib._src.fields.field_BH_sphere import magnet_sphere_field


def test_magnet_cuboid_Bfield():
    """test cuboid field"""
    mag = np.array(
        [
            (0, 0, 0),
            (1, 2, 3),
            (1, 2, 3),
            (1, 2, 3),
            (1, 2, 3),
            (2, 2, 2),
            (2, 2, 2),
            (1, 1, 1),
            (1, 1, 1),
        ]
    )
    dim = np.array(
        [
            (1, 2, 3),
            (-1, -2, 2),
            (1, 2, 2),
            (0, 2, 2),
            (1, 2, 3),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
            (2, 2, 2),
        ]
    )
    pos = np.array(
        [
            (1, 2, 3),
            (1, -1, 0),
            (1, -1, 0),
            (1, -1, 0),
            (1, 2, 3),
            (1, 1 + 1e-14, 0),
            (1, 1, 1),
            (1, -1, 2),
            (1 + 1e-14, -1, 2),
        ]
    )
    B = magnet_cuboid_field("B", pos, mag, dim)

    Btest = [
        [0.0, 0.0, 0.0],
        [-0.14174376, -0.16976459, -0.20427478],
        [-0.14174376, -0.16976459, -0.20427478],
        [0.0, 0.0, 0.0],
        [0.02596336, 0.04530334, 0.05840059],
        [np.inf, np.inf, -0.29516724],
        [0.0, 0.0, 0.0],
        [-0.0009913, -0.08747071, 0.04890262],
        [-0.0009913, -0.08747071, 0.04890262],
    ]

    np.testing.assert_allclose(B, Btest, rtol=1e-5)


def test_magnet_cuboid_field_mag0():
    """test cuboid field magnetization=0"""
    n = 10
    mag = np.zeros((n, 3))
    dim = np.random.rand(n, 3)
    pos = np.random.rand(n, 3)
    B = magnet_cuboid_field("B", pos, mag, dim)
    assert_allclose(mag, B)


def test_field_BH_cylinder_tile_mag0():
    """test cylinder_tile field magnetization=0"""
    n = 10
    mag = np.zeros((n, 3))
    r1, r2, h, phi1, phi2 = np.random.rand(5, n)
    r2 = r1 + r2
    phi2 = phi1 + phi2
    dim = np.array([r1, r2, h, phi1, phi2]).T
    pos = np.random.rand(n, 3)
    B = magnet_cylinder_segment_field("B", pos, mag, dim)
    assert_allclose(mag, B)


def test_field_sphere_vs_v2():
    """testing against old version"""
    result_v2 = np.array(
        [
            [22.0, 44.0, 66.0],
            [22.0, 44.0, 66.0],
            [38.47035383, 30.77628307, 23.0822123],
            [0.60933932, 0.43524237, 1.04458169],
            [22.0, 44.0, 66.0],
            [-0.09071337, -0.18142674, -0.02093385],
            [-0.17444878, -0.0139559, -0.10466927],
        ]
    )

    dim = np.array([1.23] * 7)
    mag = np.array([(33, 66, 99)] * 7)
    poso = np.array(
        [
            (0, 0, 0),
            (0.2, 0.2, 0.2),
            (0.4, 0.4, 0.4),
            (-1, -1, -2),
            (0.1, 0.1, 0.1),
            (1, 2, -3),
            (-3, 2, 1),
        ]
    )
    B = magnet_sphere_field("B", poso, mag, dim)

    np.testing.assert_allclose(result_v2, B, rtol=1e-6)


def test_magnet_sphere_field_mag0():
    """test cuboid field magnetization=0"""
    n = 10
    mag = np.zeros((n, 3))
    dim = np.random.rand(n)
    pos = np.random.rand(n, 3)
    B = magnet_sphere_field("B", pos, mag, dim)
    assert_allclose(mag, B)


def test_field_dipole1():
    """Test standard dipole field output computed with mathematica"""
    poso = np.array([(1, 2, 3), (-1, 2, 3)])
    mom = np.array([(2, 3, 4), (0, -3, -2)])
    B = dipole_field("B", poso, mom) * np.pi
    Btest = np.array(
        [
            (0.01090862, 0.02658977, 0.04227091),
            (0.0122722, -0.01022683, -0.02727156),
        ]
    )

    assert_allclose(B, Btest, rtol=1e-6)


def test_field_dipole2():
    """test nan return when pos_obs=0"""
    moment = np.array([(100, 200, 300)] * 2 + [(0, 0, 0)] * 2)
    observer = np.array([(0, 0, 0), (1, 2, 3)] * 2)
    B = dipole_field("B", observer, moment)

    assert all(np.isinf(B[0]))
    assert_allclose(
        B[1:], [[0.3038282, 0.6076564, 0.91148459], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )


def test_field_loop():
    """test if field function gives correct outputs"""
    # from hyperphysics
    # current = 1A
    # loop radius = 1mm
    # B at center = 0.6283185307179586 mT
    # B at 1mm on zaxis = 0.22214414690791835 mT
    pos_test_hyper = [[0, 0, 0], [0, 0, 1]]
    Btest_hyper = [[0, 0, 0.6283185307179586], [0, 0, 0.22214414690791835]]

    # from magpylib 2
    pos_test_mag2 = [
        [1, 2, 3],
        [-3, 2, 1],
        [1, -0.2, 0.3],
        [1, 0.2, -1],
        [-0.1, -0.2, 3],
        [-1, 0.2, -0.3],
        [3, -3, -3],
        [-2, -0.2, -0.3],
    ]
    Btest_mag2 = [
        [0.44179833, 0.88359665, 0.71546231],
        [-0.53137126, 0.35424751, -0.59895825],
        [72.87320789, -14.57464158, 22.07633404],
        [-13.75612867, -2.75122573, 11.36467552],
        [-0.10884885, -0.21769769, 2.41206364],
        [72.87320789, -14.57464158, 22.07633404],
        [-0.27939151, 0.27939151, 0.01220605],
        [3.25697271, 0.32569727, -5.49353046],
    ]

    pos_test = np.array(pos_test_hyper + pos_test_mag2)
    Btest = np.array(Btest_hyper + Btest_mag2)

    current = np.array([1, 1] + [123] * 8)
    dim = np.array([2, 2] + [2] * 8)

    B = current_loop_field("B", pos_test, current, dim)

    assert_allclose(B, Btest, rtol=1e-6)

    Htest = Btest * 10 / 4 / np.pi
    H = current_loop_field("H", pos_test, current, dim)
    assert_allclose(H, Htest, rtol=1e-6)


def test_field_loop2():
    """test if field function accepts correct inputs"""
    curr = np.array([1])
    dim = np.array([2])
    poso = np.array([[0, 0, 0]])
    B = current_loop_field("B", poso, curr, dim)

    curr = np.array([1] * 2)
    dim = np.array([2] * 2)
    poso = np.array([[0, 0, 0]] * 2)
    B2 = current_loop_field("B", poso, curr, dim)

    assert_allclose(B, (B2[0],))
    assert_allclose(B, (B2[1],))


def test_field_loop_specials():
    """test loop special cases"""
    cur = np.array([1, 1, 1, 1, 0, 2])
    dia = np.array([2, 2, 0, 0, 2, 2])
    obs = np.array([(0, 0, 0), (1, 0, 0), (0, 0, 0), (1, 0, 0), (1, 0, 0), (0, 0, 0)])

    B = current_loop_field("B", obs, cur, dia)
    Btest = [
        [0, 0, 0.62831853],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1.25663706],
    ]
    assert_allclose(B, Btest)


def test_field_line():
    """test line current for all cases"""

    c1 = np.array([1])
    po1 = np.array([(1, 2, 3)])
    ps1 = np.array([(0, 0, 0)])
    pe1 = np.array([(2, 2, 2)])

    # only normal
    B1 = current_line_field("B", po1, c1, ps1, pe1)
    x1 = np.array([[0.02672612, -0.05345225, 0.02672612]])
    assert_allclose(x1, B1, rtol=1e-6)

    # only on_line
    po1b = np.array([(1, 1, 1)])
    B2 = current_line_field("B", po1b, c1, ps1, pe1)
    x2 = np.zeros((1, 3))
    assert_allclose(x2, B2, rtol=1e-6)

    # only zero-segment
    B3 = current_line_field("B", po1, c1, ps1, ps1)
    x3 = np.zeros((1, 3))
    assert_allclose(x3, B3, rtol=1e-6)

    # only on_line and zero_segment
    c2 = np.array([1] * 2)
    ps2 = np.array([(0, 0, 0)] * 2)
    pe2 = np.array([(0, 0, 0), (2, 2, 2)])
    po2 = np.array([(1, 2, 3), (1, 1, 1)])
    B4 = current_line_field("B", po2, c2, ps2, pe2)
    x4 = np.zeros((2, 3))
    assert_allclose(x4, B4, rtol=1e-6)

    # normal + zero_segment
    po2b = np.array([(1, 2, 3), (1, 2, 3)])
    B5 = current_line_field("B", po2b, c2, ps2, pe2)
    x5 = np.array([[0, 0, 0], [0.02672612, -0.05345225, 0.02672612]])
    assert_allclose(x5, B5, rtol=1e-6)

    # normal + on_line
    pe2b = np.array([(2, 2, 2)] * 2)
    B6 = current_line_field("B", po2, c2, ps2, pe2b)
    x6 = np.array([[0.02672612, -0.05345225, 0.02672612], [0, 0, 0]])
    assert_allclose(x6, B6, rtol=1e-6)

    # normal + zero_segment + on_line
    c4 = np.array([1] * 3)
    ps4 = np.array([(0, 0, 0)] * 3)
    pe4 = np.array([(0, 0, 0), (2, 2, 2), (2, 2, 2)])
    po4 = np.array([(1, 2, 3), (1, 2, 3), (1, 1, 1)])
    B7 = current_line_field("B", po4, c4, ps4, pe4)
    x7 = np.array([[0, 0, 0], [0.02672612, -0.05345225, 0.02672612], [0, 0, 0]])
    assert_allclose(x7, B7, rtol=1e-6)


def test_field_line_from_vert():
    """test the Line field from vertex input"""
    p = np.array([(1, 2, 2), (1, 2, 3), (-1, 0, -3)])
    curr = np.array([1, 5, -3])

    vert1 = np.array(
        [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3), (1, 2, 3), (-3, 4, -5)]
    )
    vert2 = np.array([(0, 0, 0), (3, 3, 3), (-3, 4, -5)])
    vert3 = np.array([(1, 2, 3), (-2, -3, 3), (3, 2, 1), (3, 3, 3)])

    pos_tiled = np.tile(p, (3, 1))
    B_vert = field_BH_line_from_vert("B", pos_tiled, curr, [vert1, vert2, vert3])

    B = []
    for i, vert in enumerate([vert1, vert2, vert3]):
        for pos in p:
            p1 = vert[:-1]
            p2 = vert[1:]
            po = np.array([pos] * (len(vert) - 1))
            cu = np.array([curr[i]] * (len(vert) - 1))
            B += [np.sum(current_line_field("B", po, cu, p1, p2), axis=0)]
    B = np.array(B)

    assert_allclose(B_vert, B)


def test_field_line_v4():
    """test current_line_Bfield() for all cases"""
    cur = np.array([1] * 7)
    start = np.array([(-1, 0, 0)] * 7)
    end = np.array([(1, 0, 0), (-1, 0, 0), (1, 0, 0), (-1, 0, 0)] + [(1, 0, 0)] * 3)
    obs = np.array(
        [
            (0, 0, 1),
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 1e-16),
            (2, 0, 1),
            (-2, 0, 1),
        ]
    )
    B = current_line_field("B", obs, cur, start, end)
    Btest = np.array(
        [
            [0, -0.14142136, 0],
            [0, 0.0, 0],
            [0, 0.0, 0],
            [0, 0.0, 0],
            [0, 0.0, 0],
            [0, -0.02415765, 0],
            [0, -0.02415765, 0],
        ]
    )
    np.testing.assert_allclose(B, Btest)

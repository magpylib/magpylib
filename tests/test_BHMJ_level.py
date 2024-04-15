import numpy as np
from numpy.testing import assert_allclose
from scipy.constants import mu_0 as MU0

from magpylib._src.fields.field_BH_circle import BHJM_circle
from magpylib._src.fields.field_BH_cuboid import BHJM_magnet_cuboid
from magpylib._src.fields.field_BH_cylinder import BHJM_magnet_cylinder
from magpylib._src.fields.field_BH_cylinder_segment import BHJM_cylinder_segment
from magpylib._src.fields.field_BH_dipole import BHJM_dipole
from magpylib._src.fields.field_BH_polyline import BHJM_current_polyline
from magpylib._src.fields.field_BH_polyline import current_vertices_field
from magpylib._src.fields.field_BH_sphere import BHJM_magnet_sphere
from magpylib._src.fields.field_BH_tetrahedron import BHJM_magnet_tetrahedron
from magpylib._src.fields.field_BH_triangle import BHJM_triangle
from magpylib._src.fields.field_BH_triangularmesh import BHJM_magnet_trimesh

#######################################################################################
#######################################################################################
#######################################################################################

# NEW V5 BASIC FIELD COMPUTATION TESTS


def helper_check_HBMJ_consistency(func, **kw):
    """
    helper function to check H,B,M,J field consistencies
    returns H, B, M, J
    """
    B = func(field="B", **kw)
    H = func(field="H", **kw)
    M = func(field="M", **kw)
    J = func(field="J", **kw)
    np.testing.assert_allclose(M * MU0, J)
    np.testing.assert_allclose(B, MU0 * H + J)
    return H, B, M, J


def test_BHJM_magnet_cuboid():
    """test cuboid field"""
    pol = np.array(
        [
            (0, 0, 0),
            (1, 2, 3),
            (1, 2, 3),
            (1, 2, 3),
            (1, 2, 3),
            (1, 2, 3),
        ]
    )
    dim = np.array(
        [
            (1, 2, 3),
            (-1, -2, 2),
            (1, 2, 2),
            (0, 2, 2),
            (1, 2, 3),
            (3, 3, 3),
        ]
    )
    obs = np.array(
        [
            (1, 2, 3),
            (1, -1, 0),
            (1, -1, 0),
            (1, -1, 0),
            (1, 2, 3),
            (0, 0, 0),  # inside
        ]
    )
    kw = {
        "observers": obs,
        "polarization": pol,
        "dimension": dim,
    }
    H, B, _, J = helper_check_HBMJ_consistency(BHJM_magnet_cuboid, **kw)

    Btest = [
        [0.0, 0.0, 0.0],
        [-0.14174376, -0.16976459, -0.20427478],
        [-0.14174376, -0.16976459, -0.20427478],
        [0.0, 0.0, 0.0],
        [0.02596336, 0.04530334, 0.05840059],
        [0.66666667, 1.33333333, 2.0],
    ]
    np.testing.assert_allclose(B, Btest, rtol=1e-5)

    Htest = [
        [0.0, 0.0, 0.0],
        [-112796.09804171, -135094.37189185, -162556.70519527],
        [-112796.09804171, -135094.37189185, -162556.70519527],
        [0.0, 0.0, 0.0],
        [20660.98851314, 36051.25202256, 46473.71425434],
        [-265258.23848649, -530516.47697298, -795774.71545948],
    ]
    np.testing.assert_allclose(H, Htest, rtol=1e-5)

    Jtest = np.array([(0, 0, 0)] * 5 + [(1, 2, 3)])
    np.testing.assert_allclose(J, Jtest, rtol=1e-5)

    # H_inout = BHJM_magnet_cuboid(field="H", in_out="outside", **kw)
    # Htest_inout = Htest + Jtest / MU0
    # np.testing.assert_allclose(H_inout, Htest_inout, rtol=1e-5)


def test_BHJM_magnet_cylinder():
    """test cylinder field computation"""
    pol = np.array(
        [
            (0, 0, 0),
            (1, 2, 3),
            (3, 2, -1),
            (1, 1, 1),
        ]
    )
    dim = np.array(
        [
            (1, 2),
            (2, 2),
            (1, 2),
            (3, 3),
        ]
    )
    obs = np.array(
        [
            (1, 2, 3),
            (1, -1, 0),
            (1, 1, 1),
            (0, 0, 0),  # inside
        ]
    )

    kw = {
        "observers": obs,
        "polarization": pol,
        "dimension": dim,
    }
    H, B, _, J = helper_check_HBMJ_consistency(BHJM_magnet_cylinder, **kw)

    Btest = [
        [0.0, 0.0, 0.0],
        [-0.36846057, -0.10171405, -0.33006492],
        [0.05331225, 0.07895873, 0.10406998],
        [0.64644661, 0.64644661, 0.70710678],
    ]
    np.testing.assert_allclose(B, Btest)

    Htest = [
        [0.0, 0.0, 0.0],
        [-293211.60229288, -80941.4714998, -262657.31858654],
        [42424.54100401, 62833.36365626, 82816.25721518],
        [-281348.8487991, -281348.8487991, -233077.01786129],
    ]
    np.testing.assert_allclose(H, Htest)

    Jtest = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (1, 1, 1)])
    np.testing.assert_allclose(J, Jtest)

    # H_inout = BHJM_magnet_cylinder(field="H", in_out="outside", **kw)
    # Htest_inout = Htest - Jtest / MU0
    # np.testing.assert_allclose(H_inout, Htest_inout, rtol=1e-5)


def test_BHJM_magnet_sphere():
    """test BHJM_magnet_sphere"""
    pol = np.array(
        [
            (0, 0, 0),
            (1, 2, 3),
            (2, 3, -1),
            (2, 3, -1),
        ]
    )
    dia = np.array([1, 2, 3, 4])
    obs = np.array(
        [
            (1, 2, 3),
            (1, -1, 0),
            (0, -1, 0),  # inside
            (1, -1, 0.5),  # inside
        ]
    )

    kw = {
        "observers": obs,
        "polarization": pol,
        "diameter": dia,
    }
    H, B, _, J = helper_check_HBMJ_consistency(BHJM_magnet_sphere, **kw)

    Btest = [
        [0.0, 0.0, 0.0],
        [-0.29462783, -0.05892557, -0.35355339],
        [1.33333333, 2.0, -0.66666667],
        [1.33333333, 2.0, -0.66666667],
    ]
    np.testing.assert_allclose(B, Btest)

    Htest = [
        [0.0, 0.0, 0.0],
        [-234457.37399925, -46891.47479985, -281348.8487991],
        [-530516.47697298, -795774.71545948, 265258.23848649],
        [-530516.47697298, -795774.71545948, 265258.23848649],
    ]
    np.testing.assert_allclose(H, Htest)

    Jtest = [(0, 0, 0), (0, 0, 0), (2, 3, -1), (2, 3, -1)]
    np.testing.assert_allclose(J, Jtest)


def test_field_cylinder_segment_BH():
    """CylinderSegment field test"""
    pol = np.array(
        [
            (0, 0, 0),
            (1, 2, 3),
            (2, 3, -1),
            (2, 3, -1),
        ]
    )
    dim = np.array(
        [
            (1, 2, 3, 10, 20),
            (1, 2, 3, 10, 20),
            (1, 3, 2, -50, 50),
            (0.1, 5, 2, 20, 370),
        ]
    )
    obs = np.array(
        [
            (1, 2, 3),
            (1, -1, 0),
            (0, -1, 0),
            (1, -1, 0.5),  # inside
        ]
    )

    kw = {
        "observers": obs,
        "polarization": pol,
        "dimension": dim,
    }
    H, B, _, J = helper_check_HBMJ_consistency(BHJM_cylinder_segment, **kw)

    Btest = [
        [0.0, 0.0, 0.0],
        [0.00762186, 0.04194934, -0.01974813],
        [0.52440702, -0.04650694, 0.09432828],
        [1.75574175, 2.58945648, -0.19025747],
    ]
    np.testing.assert_allclose(B, Btest, rtol=1e-6)

    Htest = [
        [0.0, 0.0, 0.0],
        [6065.28627343, 33382.22618218, -15715.05894253],
        [417309.84428576, -37009.05020239, 75064.06294505],
        [-194374.5385654, -326700.15326755, 644372.62925584],
    ]
    np.testing.assert_allclose(H, Htest, rtol=1e-6)

    Jtest = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (2, 3, -1)]
    np.testing.assert_allclose(J, Jtest)


def test_BHJM_triangle_BH():
    """Test of triangle field core function"""
    pol = np.array(
        [
            (0, 0, 0),
            (1, 2, 3),
            (2, -1, 1),
            (1, -1, 2),
        ]
    )
    vert = np.array(
        [
            [(0, 0, 0), (0, 1, 0), (1, 0, 0)],
            [(0, 0, 0), (0, 1, 0), (1, 0, 0)],
            [(1, 2, 3), (0, 1, -5), (1, 1, 5)],
            [(1, 2, 2), (0, 1, -1), (3, -1, 1)],
        ]
    )
    obs = np.array(
        [
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            (2, 3, 1),
        ]
    )

    kw = {
        "observers": obs,
        "polarization": pol,
        "vertices": vert,
    }
    H, B, _, J = helper_check_HBMJ_consistency(BHJM_triangle, **kw)

    Btest = [
        [0.0, 0.0, 0.0],
        [-0.02825571, -0.02825571, -0.04386991],
        [-0.34647603, 0.29421715, 0.06980312],
        [0.02041789, 0.05109073, 0.00218011],
    ]
    np.testing.assert_allclose(B, Btest, rtol=1e-06)

    Htest = [
        [0.0, 0.0, 0.0],
        [-22485.1813849, -22485.1813849, -34910.56834885],
        [-275716.86458395, 234130.57085866, 55547.55765999],
        [16248.03897974, 40656.7134656, 1734.8781397],
    ]
    np.testing.assert_allclose(H, Htest, rtol=1e-06)

    Jtest = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
    np.testing.assert_allclose(J, Jtest)


def test_magnet_tetrahedron_field_BH():
    """Test of tetrahedron field core function"""
    pol = np.array(
        [
            (0, 0, 0),
            (1, 2, 3),
            (-1, 0.5, 0.1),  # inside
            (2, 2, -1),
            (3, 2, 1),  # inside
        ]
    )
    vert = np.array(
        [
            [(0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)],
            [(0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)],
            [(-1, 0, -1), (1, 1, -1), (1, -1, -1), (0, 0, 1)],
            [(-1, 0, -1), (1, 1, -1), (1, -1, -1), (0, 0, 1)],
            [(-10, 0, -10), (10, 10, -10), (10, -10, -10), (0, 0, 10)],
        ]
    )
    obs = np.array(
        [
            (1, 1, 1),
            (1, 1, 1),
            (0, 0, 0),
            (2, 0, 0),
            (1, 2, 3),
        ]
    )
    kw = {
        "observers": obs,
        "polarization": pol,
        "vertices": vert,
    }
    H, B, _, J = helper_check_HBMJ_consistency(BHJM_magnet_tetrahedron, **kw)

    Btest = [
        [0.0, 0.0, 0.0],
        [0.02602367, 0.02081894, 0.0156142],
        [-0.69704332, 0.20326329, 0.11578416],
        [0.04004769, -0.03186713, 0.03854207],
        [2.09887014, 1.42758632, 0.8611617],
    ]
    np.testing.assert_allclose(B, Btest, rtol=1e-06)

    Htest = [
        [0.0, 0.0, 0.0],
        [20708.97827326, 16567.1826186, 12425.38696395],
        [241085.26350642, -236135.56979233, 12560.63814427],
        [31868.94160192, -25359.05664996, 30670.80436549],
        [-717096.35551784, -455512.33538799, -110484.00786285],
    ]
    np.testing.assert_allclose(H, Htest, rtol=1e-06)

    Jtest = [(0, 0, 0), (0, 0, 0), (-1, 0.5, 0.1), (0, 0, 0), (3, 2, 1)]
    np.testing.assert_allclose(J, Jtest, rtol=1e-06)


def test_BHJM_magnet_trimesh_BH():
    """Test of BHJM_magnet_trimesh core-like function"""

    mesh1 = [
        [
            [0.7439252734184265, 0.5922041535377502, 0.30962786078453064],
            [0.3820107579231262, -0.8248414397239685, -0.416778564453125],
            [-0.5555410385131836, 0.4872661232948303, -0.6737549901008606],
        ],
        [
            [-0.5555410385131836, 0.4872661232948303, -0.6737549901008606],
            [0.3820107579231262, -0.8248414397239685, -0.416778564453125],
            [-0.5703949332237244, -0.25462886691093445, 0.7809056639671326],
        ],
        [
            [0.7439252734184265, 0.5922041535377502, 0.30962786078453064],
            [-0.5703949332237244, -0.25462886691093445, 0.7809056639671326],
            [0.3820107579231262, -0.8248414397239685, -0.416778564453125],
        ],
        [
            [0.7439252734184265, 0.5922041535377502, 0.30962786078453064],
            [-0.5555410385131836, 0.4872661232948303, -0.6737549901008606],
            [-0.5703949332237244, -0.25462886691093445, 0.7809056639671326],
        ],
    ]
    mesh2 = [  # inside
        [
            [0.9744000434875488, 0.15463787317276, 0.16319207847118378],
            [-0.12062954157590866, -0.8440634608268738, -0.522499144077301],
            [-0.3775683045387268, 0.7685779929161072, -0.516459047794342],
        ],
        [
            [-0.3775683045387268, 0.7685779929161072, -0.516459047794342],
            [-0.12062954157590866, -0.8440634608268738, -0.522499144077301],
            [-0.47620221972465515, -0.0791524201631546, 0.8757661581039429],
        ],
        [
            [0.9744000434875488, 0.15463787317276, 0.16319207847118378],
            [-0.47620221972465515, -0.0791524201631546, 0.8757661581039429],
            [-0.12062954157590866, -0.8440634608268738, -0.522499144077301],
        ],
        [
            [0.9744000434875488, 0.15463787317276, 0.16319207847118378],
            [-0.3775683045387268, 0.7685779929161072, -0.516459047794342],
            [-0.47620221972465515, -0.0791524201631546, 0.8757661581039429],
        ],
    ]
    mesh = np.array([mesh1, mesh2])
    pol = np.array([(1, 2, 3), (3, 2, 1)])
    obs = np.array([(1, 2, 3), (0, 0, 0)])
    kw = {
        "observers": obs,
        "polarization": pol,
        "mesh": mesh,
    }
    H, B, _, J = helper_check_HBMJ_consistency(BHJM_magnet_trimesh, **kw)

    Btest = [
        [1.54452002e-03, 3.11861149e-03, 4.68477835e-03],
        [2.00000002e00, 1.33333333e00, 6.66666685e-01],
    ]
    np.testing.assert_allclose(B, Btest)

    Htest = [
        [1229.08998194, 2481.71216888, 3728.02815642],
        [-795774.70120171, -530516.47792526, -265258.22366805],
    ]
    np.testing.assert_allclose(H, Htest)

    Jtest = [(0, 0, 0), (3, 2, 1)]
    np.testing.assert_allclose(J, Jtest, rtol=1e-06)


def test_BHJM_circle():
    """Test of current circle field core function"""
    kw = {
        "observers": np.array([(1, 1, 1), (2, 2, 2), (3, 3, 3)]),
        "current": np.array([1, 1, 2]) * 1e3,
        "diameter": np.array([2, 4, 6]),
    }
    H, B, M, _ = helper_check_HBMJ_consistency(BHJM_circle, **kw)

    Btest = (
        np.array(
            [
                [0.06235974, 0.06235974, 0.02669778],
                [0.03117987, 0.03117987, 0.01334889],
                [0.04157316, 0.04157316, 0.01779852],
            ]
        )
        * 1e-3
    )
    np.testing.assert_allclose(B, Btest)

    Htest = (
        np.array(
            [
                [49624.3033947, 49624.3033947, 21245.41908818],
                [24812.15169735, 24812.15169735, 10622.70954409],
                [33082.8689298, 33082.8689298, 14163.61272545],
            ]
        )
        * 1e-3
    )
    np.testing.assert_allclose(H, Htest)

    Mtest = [(0, 0, 0)] * 3
    np.testing.assert_allclose(M, Mtest, rtol=1e-06)


def test_BHJM_current_polyline():
    """Test of current polyline field core function"""
    vert = np.array([(-1.5, 0, 0), (-0.5, 0, 0), (0.5, 0, 0), (1.5, 0, 0)])

    kw = {
        "observers": np.array([(0, 0, 1)] * 3),
        "current": np.array([1, 1, 1]),
        "segment_start": vert[:-1],
        "segment_end": vert[1:],
    }
    H, B, M, _ = helper_check_HBMJ_consistency(BHJM_current_polyline, **kw)

    Btest = (
        np.array(
            [
                [0.0, -0.03848367, 0.0],
                [0.0, -0.08944272, 0.0],
                [0.0, -0.03848367, 0.0],
            ]
        )
        * 1e-6
    )
    np.testing.assert_allclose(B, Btest, rtol=0, atol=1e-7)

    Htest = (
        np.array(
            [
                [0.0, -30624.33145161, 0.0],
                [0.0, -71176.25434172, 0.0],
                [0.0, -30624.33145161, 0.0],
            ]
        )
        * 1e-6
    )
    np.testing.assert_allclose(H, Htest, rtol=0, atol=1e-7)

    Mtest = [(0, 0, 0)] * 3
    np.testing.assert_allclose(M, Mtest, rtol=1e-06)


def test_BHJM_dipole():
    """Test of dipole field core function"""
    pol = np.array([(0, 0, 1), (1, 0, 1), (-1, 0.321, 0.123)])

    kw = {
        "observers": np.array([(1, 2, 3), (-1, -2, -3), (3, 3, -1)]),
        "moment": pol * 4 * np.pi / 3 / MU0,
    }
    H, B, M, _ = helper_check_HBMJ_consistency(BHJM_dipole, **kw)

    Btest = [
        [4.09073329e-03, 8.18146659e-03, 5.90883698e-03],
        [-9.09051843e-04, 1.09086221e-02, 9.99957028e-03],
        [-9.32067617e-05, -5.41001702e-03, 8.77626395e-04],
    ]
    np.testing.assert_allclose(B, Btest)

    Htest = [
        [3255.30212351, 6510.60424703, 4702.1030673],
        [-723.40047189, 8680.8056627, 7957.40519081],
        [-74.17158426, -4305.1547508, 698.3928945],
    ]
    np.testing.assert_allclose(H, Htest)

    Mtest = [(0, 0, 0)] * 3
    np.testing.assert_allclose(M, Mtest, rtol=1e-06)


#######################################################################################
#######################################################################################
#######################################################################################

# OTHER TESTS AND V4 TESTS


def test_field_loop_specials():
    """test loop special cases"""
    cur = np.array([1, 1, 1, 1, 0, 2])
    dia = np.array([2, 2, 0, 0, 2, 2])
    obs = np.array([(0, 0, 0), (1, 0, 0), (0, 0, 0), (1, 0, 0), (1, 0, 0), (0, 0, 0)])

    B = (
        BHJM_circle(
            field="B",
            observers=obs,
            diameter=dia,
            current=cur,
        )
        * 1e6
    )
    Btest = [
        [0, 0, 0.62831853],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1.25663706],
    ]
    assert_allclose(B, Btest)


def test_field_line_special_cases():
    """test line current for all cases"""

    c1 = np.array([1])
    po1 = np.array([(1, 2, 3)])
    ps1 = np.array([(0, 0, 0)])
    pe1 = np.array([(2, 2, 2)])

    # only normal
    B1 = (
        BHJM_current_polyline(
            field="B",
            observers=po1,
            current=c1,
            segment_start=ps1,
            segment_end=pe1,
        )
        * 1e6
    )
    x1 = np.array([[0.02672612, -0.05345225, 0.02672612]])
    assert_allclose(x1, B1, rtol=1e-6)

    # only on_line
    po1b = np.array([(1, 1, 1)])
    B2 = (
        BHJM_current_polyline(
            field="B",
            observers=po1b,
            current=c1,
            segment_start=ps1,
            segment_end=pe1,
        )
        * 1e6
    )
    x2 = np.zeros((1, 3))
    assert_allclose(x2, B2, rtol=1e-6)

    # only zero-segment
    B3 = (
        BHJM_current_polyline(
            field="B",
            observers=po1,
            current=c1,
            segment_start=ps1,
            segment_end=ps1,
        )
        * 1e6
    )
    x3 = np.zeros((1, 3))
    assert_allclose(x3, B3, rtol=1e-6)

    # only on_line and zero_segment
    c2 = np.array([1] * 2)
    ps2 = np.array([(0, 0, 0)] * 2)
    pe2 = np.array([(0, 0, 0), (2, 2, 2)])
    po2 = np.array([(1, 2, 3), (1, 1, 1)])
    B4 = (
        BHJM_current_polyline(
            field="B",
            observers=po2,
            current=c2,
            segment_start=ps2,
            segment_end=pe2,
        )
        * 1e6
    )
    x4 = np.zeros((2, 3))
    assert_allclose(x4, B4, rtol=1e-6)

    # normal + zero_segment
    po2b = np.array([(1, 2, 3), (1, 2, 3)])
    B5 = (
        BHJM_current_polyline(
            field="B",
            observers=po2b,
            current=c2,
            segment_start=ps2,
            segment_end=pe2,
        )
        * 1e6
    )
    x5 = np.array([[0, 0, 0], [0.02672612, -0.05345225, 0.02672612]])
    assert_allclose(x5, B5, rtol=1e-6)

    # normal + on_line
    pe2b = np.array([(2, 2, 2)] * 2)
    B6 = (
        BHJM_current_polyline(
            field="B",
            observers=po2,
            current=c2,
            segment_start=ps2,
            segment_end=pe2b,
        )
        * 1e6
    )
    x6 = np.array([[0.02672612, -0.05345225, 0.02672612], [0, 0, 0]])
    assert_allclose(x6, B6, rtol=1e-6)

    # normal + zero_segment + on_line
    c4 = np.array([1] * 3)
    ps4 = np.array([(0, 0, 0)] * 3)
    pe4 = np.array([(0, 0, 0), (2, 2, 2), (2, 2, 2)])
    po4 = np.array([(1, 2, 3), (1, 2, 3), (1, 1, 1)])
    B7 = (
        BHJM_current_polyline(
            field="B",
            observers=po4,
            current=c4,
            segment_start=ps4,
            segment_end=pe4,
        )
        * 1e6
    )
    x7 = np.array([[0, 0, 0], [0.02672612, -0.05345225, 0.02672612], [0, 0, 0]])
    assert_allclose(x7, B7, rtol=1e-6)


def test_field_loop2():
    """test if field function accepts correct inputs"""
    curr = np.array([1])
    dia = np.array([2])
    obs = np.array([[0, 0, 0]])
    B = BHJM_circle(
        field="B",
        observers=obs,
        current=curr,
        diameter=dia,
    )

    curr = np.array([1] * 2)
    dia = np.array([2] * 2)
    obs = np.array([[0, 0, 0]] * 2)
    B2 = BHJM_circle(
        field="B",
        observers=obs,
        current=curr,
        diameter=dia,
    )

    assert_allclose(B, (B2[0],))
    assert_allclose(B, (B2[1],))


def test_field_line_from_vert():
    """test the Polyline field from vertex input"""
    observers = np.array([(1, 2, 2), (1, 2, 3), (-1, 0, -3)])
    current = np.array([1, 5, -3])

    vertices = np.array(
        [
            np.array(
                [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3), (1, 2, 3), (-3, 4, -5)]
            ),
            np.array([(0, 0, 0), (3, 3, 3), (-3, 4, -5)]),
            np.array([(1, 2, 3), (-2, -3, 3), (3, 2, 1), (3, 3, 3)]),
        ],
        dtype="object",
    )

    B_vert = current_vertices_field(
        field="B",
        observers=observers,
        vertices=vertices,
        current=current,
    )

    B = []
    for obs, vert, curr in zip(observers, vertices, current):
        p1 = vert[:-1]
        p2 = vert[1:]
        po = np.array([obs] * (len(vert) - 1))
        cu = np.array([curr] * (len(vert) - 1))
        B += [
            np.sum(
                BHJM_current_polyline(
                    field="B",
                    observers=po,
                    current=cu,
                    segment_start=p1,
                    segment_end=p2,
                ),
                axis=0,
            )
        ]
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
    B = (
        BHJM_current_polyline(
            field="B",
            observers=obs,
            current=cur,
            segment_start=start,
            segment_end=end,
        )
        * 1e6
    )
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


def test_triangle5():
    """special case tests on edges - result is continuous and 0 for vertical component"""
    btest1 = [
        [26.29963526814195, 15.319834473660082, 0.0],
        [54.91549594789228, 41.20535983076747, 0.0],
        [32.25241487782939, 15.087161660417559, 0.0],
        [10.110611199952707, -11.41176203622237, 0.0],
        [-3.8084378251737285, -30.875600143560657, -0.0],
        [-15.636505140623612, -50.00854548249858, -0.0],
        [-27.928308992688645, -72.80800891847107, -0.0],
        [-45.34417750711242, -109.5871836961927, -0.0],
        [-36.33970306054345, 12.288824457077656, 0.0],
        [-16.984738462958845, 4.804383318447626, 0.0],
    ]

    btest2 = [
        [15.31983447366009, 26.299635268142033, 0.0],
        [41.20535983076747, 54.91549594789104, 0.0],
        [-72.61316618947018, 32.25241487782958, 0.0],
        [-54.07597251255013, 10.110611199952693, 0.0],
        [-44.104089712909634, -3.808437825173785, -0.0],
        [-36.78005591314963, -15.636505140623605, -0.0],
        [-30.143798442143236, -27.92830899268858, -0.0],
        [-21.886855846306176, -45.34417750711366, -0.0],
        [12.288824457077965, -36.33970306054315, 0.0],
        [4.80438331844773, -16.98473846295874, 0.0],
    ]

    n = 10
    ts = np.linspace(-1, 6, n)
    obs1 = np.array([(t, 0, 0) for t in ts])
    obs2 = np.array([(0, t, 0) for t in ts])
    mag = np.array([(111, 222, 333)] * n)
    ver = np.array([[(0, 0, 0), (0, 5, 0), (5, 0, 0)]] * n)

    b1 = (
        BHJM_triangle(
            field="H",
            observers=obs1,
            polarization=mag,
            vertices=ver,
        )
        * 1e-6
    )
    np.testing.assert_allclose(btest1, b1)
    b2 = (
        BHJM_triangle(
            field="H",
            observers=obs2,
            polarization=mag,
            vertices=ver,
        )
        * 1e-6
    )
    np.testing.assert_allclose(btest2, b2)


def test_triangle6():
    """special case tests on corners - result is nan"""
    obs1 = np.array([(0, 0, 0)])
    obs2 = np.array([(0, 5, 0)])
    obs3 = np.array([(5, 0, 0)])
    mag = np.array([(1, 2, 3)])
    ver = np.array([[(0, 0, 0), (0, 5, 0), (5, 0, 0)]])
    b1 = BHJM_triangle(
        field="B",
        observers=obs1,
        polarization=mag,
        vertices=ver,
    )
    b2 = BHJM_triangle(
        field="B",
        observers=obs2,
        polarization=mag,
        vertices=ver,
    )
    b3 = BHJM_triangle(
        field="B",
        observers=obs3,
        polarization=mag,
        vertices=ver,
    )
    for b in [b1, b2, b3]:
        np.testing.assert_equal(b, np.array([[np.nan] * 3]))

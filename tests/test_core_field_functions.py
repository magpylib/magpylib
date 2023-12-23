import numpy as np
import pytest
from numpy.testing import assert_allclose

import magpylib as magpy
from magpylib._src.exceptions import MagpylibDeprecationWarning
from magpylib._src.exceptions import MagpylibMissingInput
from magpylib._src.fields.field_BH_polyline import current_vertices_field
from magpylib._src.fields.field_BH_triangularmesh import magnet_trimesh_field
from magpylib._src.utility import MU0
from magpylib.core import current_circle_field
from magpylib.core import current_line_field
from magpylib.core import current_loop_field
from magpylib.core import current_polyline_field
from magpylib.core import dipole_field
from magpylib.core import magnet_cuboid_field
from magpylib.core import magnet_cylinder_field
from magpylib.core import magnet_cylinder_segment_field
from magpylib.core import magnet_sphere_field
from magpylib.core import magnet_tetrahedron_field
from magpylib.core import triangle_field


#######################################################################################
#######################################################################################
#######################################################################################

# BASIC FIELD COMPUTATION TESTS


def test_magnet_cuboid_field_BH():
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
            (0, 0, 0),
        ]
    )

    B = magnet_cuboid_field(
        field="B",
        observers=obs,
        polarization=pol,
        dimension=dim,
    )
    H = magnet_cuboid_field(
        field="H",
        observers=obs,
        polarization=pol,
        dimension=dim,
    )
    J = np.array([(0, 0, 0)] * 5 + [(1, 2, 3)])
    np.testing.assert_allclose(B, MU0 * H + J)

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


def test_magnet_cylinder_field_BH():
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
            (0, 0, 0),
        ]
    )
    B = magpy.core.magnet_cylinder_field(
        field="B",
        observers=obs,
        polarization=pol,
        dimension=dim,
    )
    H = magpy.core.magnet_cylinder_field(
        field="H",
        observers=obs,
        polarization=pol,
        dimension=dim,
    )
    J = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (1, 1, 1)])
    np.testing.assert_allclose(B, MU0 * H + J)

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


def test_magnet_sphere_field_BH():
    """test magnet_sphere_field"""
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
            (0, -1, 0),
            (1, -1, 0.5),
        ]
    )
    B = magnet_sphere_field(
        field="B",
        observers=obs,
        diameter=dia,
        polarization=pol,
    )
    H = magnet_sphere_field(
        field="H",
        observers=obs,
        diameter=dia,
        polarization=pol,
    )
    J = np.array([(0, 0, 0), (0, 0, 0), pol[2], pol[3]])
    np.testing.assert_allclose(B, MU0 * H + J)

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


def test_field_cylinder_segment_BH():
    """CylinderSegmetn field test"""
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
            (1, -1, 0.5),
        ]
    )
    B = magnet_cylinder_segment_field(
        field="B",
        observers=obs,
        dimension=dim,
        polarization=pol,
    )
    H = magnet_cylinder_segment_field(
        field="H",
        observers=obs,
        dimension=dim,
        polarization=pol,
    )
    J = np.array([(0, 0, 0)] * 3 + [pol[3]])
    np.testing.assert_allclose(B, MU0 * H + J)

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


def test_triangle_field_BH():
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
    B = triangle_field(
        field="B",
        observers=obs,
        vertices=vert,
        polarization=pol,
    )
    H = triangle_field(
        field="H",
        observers=obs,
        vertices=vert,
        polarization=pol,
    )
    np.testing.assert_allclose(B, MU0 * H)

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


def test_magnet_tetrahedron_field_BH():
    """Test of tetrahedron field core function"""
    pol = np.array(
        [
            (0, 0, 0),
            (1, 2, 3),
            (-1, 0.5, 0.1),
            (2, 2, -1),
        ]
    )
    vert = np.array(
        [
            [(0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)],
            [(0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)],
            [(-1, 0, -1), (1, 1, -1), (1, -1, -1), (0, 0, 1)],
            [(-1, 0, -1), (1, 1, -1), (1, -1, -1), (0, 0, 1)],
        ]
    )
    obs = np.array(
        [
            (1, 1, 1),
            (1, 1, 1),
            (0, 0, 0),
            (2, 0, 0),
        ]
    )
    B = magnet_tetrahedron_field(
        field="B",
        observers=obs,
        vertices=vert,
        polarization=pol,
    )
    H = magnet_tetrahedron_field(
        field="H",
        observers=obs,
        vertices=vert,
        polarization=pol,
    )
    J = np.array([(0, 0, 0)] * 2 + [pol[2]] + [(0, 0, 0)])
    np.testing.assert_allclose(B, MU0 * H + J)

    Btest = [
        [0.0, 0.0, 0.0],
        [0.02602367, 0.02081894, 0.0156142],
        [-0.69704332, 0.20326329, 0.11578416],
        [0.04004769, -0.03186713, 0.03854207],
    ]
    np.testing.assert_allclose(B, Btest, rtol=1e-06)

    Htest = [
        [0.0, 0.0, 0.0],
        [20708.97827326, 16567.1826186, 12425.38696395],
        [241085.26350642, -236135.56979233, 12560.63814427],
        [31868.94160192, -25359.05664996, 30670.80436549],
    ]
    np.testing.assert_allclose(H, Htest, rtol=1e-06)


def test_magnet_trimesh_field_BH():
    """Test of magnet_trimesh_field core-like function"""

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
    mesh2 = [
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
    meshes = np.array([mesh1, mesh2])
    pol = np.array([(1, 2, 3), (3, 2, 1)])
    obs = np.array([(1, 2, 3), (0, 0, 0)])
    B = magnet_trimesh_field(
        field="B",
        observers=obs,
        mesh=meshes,
        polarization=pol,
    )
    H = magnet_trimesh_field(
        field="H",
        observers=obs,
        mesh=meshes,
        polarization=pol,
    )
    J = np.array([(0, 0, 0), (3, 2, 1)])
    np.testing.assert_allclose(B, MU0 * H + J)

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


def test_current_circle_field_BH():
    """Test of current circle field core function"""
    B = magpy.core.current_circle_field(
        field="B",
        observers=np.array([(1, 1, 1), (2, 2, 2), (3, 3, 3)]),
        diameter=np.array([2, 4, 6]),
        current=np.array([1, 1, 2]) * 1e3,
    )
    H = magpy.core.current_circle_field(
        field="H",
        observers=np.array([(1, 1, 1), (2, 2, 2), (3, 3, 3)]),
        diameter=np.array([2, 4, 6]),
        current=np.array([1, 1, 2]) * 1e3,
    )
    np.testing.assert_allclose(B, MU0 * H)

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


# def test_current_polyline_field_BH():
#     """Test of current polyline field core function"""
#     vert=np.array([(-1.5,0,0), (-.5,0,0), (.5,0,0), (1.5,0,0)])
#     B = magpy.core.current_polyline_field(
#         field="B",
#         observers=np.array([(0,0,1)]*3),
#         segment_start=vert[:-1],
#         segment_end=vert[1:],
#         current=np.array([1, 1, 2]),
#     )
#     H = magpy.core.current_polyline_field(
#         field="H",
#         observers=np.array([(0,0,1)]*3),
#         segment_start=vert[:-1],
#         segment_end=vert[1:],
#         current=np.array([1, 1, 2]),
#     )
#     np.testing.assert_allclose(B, MU0 * H)

#     Btest = np.array([
#         [0.0, -0.03848367, 0.0],
#         [0.0, -0.08944272, 0.0],
#         [0.0, -0.03848367, 0.0],
#     ])*1e-6
#     np.testing.assert_allclose(B, Btest, rtol=0, atol=1e-7)

#     Htest = np.array([
#         [0.0, -30624.33145161, 0.0],
#         [0.0, -71176.25434172, 0.0],
#         [0.0, -30624.33145161, 0.0],
#     ])*1e-6
#     np.testing.assert_allclose(H, Htest, rtol=0, atol=1e-7)


def test_dipole_field_BH():
    """Test of dipole field core function"""
    obs = np.array([(1, 2, 3), (-1, -2, -3), (3, 3, -1)])
    pol = np.array([(0, 0, 1), (1, 0, 1), (-1, 0.321, 0.123)])
    mom = pol * 4 * np.pi / 3 / MU0

    B = magpy.core.dipole_field(
        field="B",
        observers=obs,
        moment=mom,
    )
    H = magpy.core.dipole_field(
        field="H",
        observers=obs,
        moment=mom,
    )
    np.testing.assert_allclose(B, MU0 * H)

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


#######################################################################################
#######################################################################################
#######################################################################################

# FIELD COMPUTATION PHYSICS CONSISTENCY TESTS


def test_core_phys_dipole_circle():
    """
    test dipole vs circular current loop
    moment = current * surface
    far field test
    """
    obs = np.array([(10, 20, 30), (-10, -20, 30)])
    dia = np.array([2, 2])
    curr = np.array([1e3, 1e3])
    mom = ((dia / 2) ** 2 * np.pi * curr * np.array([(0, 0, 1)] * 2).T).T

    B1 = magpy.core.current_circle_field(
        field="B",
        observers=obs,
        diameter=dia,
        current=curr,
    )
    B2 = magpy.core.dipole_field(
        field="B",
        observers=obs,
        moment=mom,
    )
    np.testing.assert_allclose(B1, B2, rtol=1e-02)


def test_core_phys_dipole_polyline():
    """
    test dipole VS square current loop
    moment = I x A, far field test
    """
    obs1 = np.array([(10, 20, 30)])
    obs4 = np.array([(10, 20, 30)] * 4)
    vert = np.array([(1, 1, 0), (1, -1, 0), (-1, -1, 0), (-1, 1, 0), (1, 1, 0)])
    curr1 = 1e3
    curr4 = np.array([curr1] * 4)
    mom = (4 * curr1 * np.array([(0, 0, 1)]).T).T

    B1 = magpy.core.current_polyline_field(
        field="B",
        observers=obs4,
        segment_start=vert[:-1],
        segment_end=vert[1:],
        current=curr4,
    )
    B1 = np.sum(B1, axis=0)
    B2 = magpy.core.dipole_field(
        field="B",
        observers=obs1,
        moment=mom,
    )[0]

    np.testing.assert_allclose(B1, -B2, rtol=1e-03)


def test_core_phys_circle_polyline():
    """approximate circle with polyline"""
    ts = np.linspace(0, 2 * np.pi, 300)
    vert = np.array([(np.sin(t), np.cos(t), 0) for t in ts])
    curr = np.array([1])
    curr99 = np.array([1] * 299)
    obs = np.array([(1, 2, 3)])
    obs99 = np.array([(1, 2, 3)] * 299)
    dia = np.array([2])

    H1 = magpy.core.current_circle_field(
        field="H",
        observers=obs,
        diameter=dia,
        current=curr,
    )[0]
    H2 = magpy.core.current_polyline_field(
        field="H",
        observers=obs99,
        segment_start=vert[:-1],
        segment_end=vert[1:],
        current=curr99,
    )
    H2 = np.sum(H2, axis=0)

    np.testing.assert_allclose(H1, -H2, rtol=1e-4)


def test_core_physics_dipole_sphere():
    """
    dipole and sphere field must be similar outside
    moment = magnetization * volume
    """
    obs = np.array([(1, 2, 3), (-2, -2, -2), (3, 5, -4), (5, 4, 0.1)])
    dia = np.array([2, 3, 0.1, 3.3])
    pol = np.array([(1, 2, 3), (0, 0, 1), (-1, -2, 0), (1, -1, 0.1)])
    mom = np.array([4 * (d / 2) ** 3 * np.pi / 3 * p / MU0 for d, p in zip(dia, pol)])

    B1 = magpy.core.magnet_sphere_field(
        field="B",
        observers=obs,
        diameter=dia,
        polarization=pol,
    )
    B2 = magpy.core.dipole_field(
        field="B",
        observers=obs,
        moment=mom,
    )
    np.testing.assert_allclose(B1, B2, rtol=0, atol=1e-16)


def test_core_physics_long_solenoid():
    """
    test if field from solenoid converges to long-solenoid field
    Bz = I*N/L
    """
    I = 1
    N = 10000
    L = 100
    B = magpy.core.current_circle_field(
        field="H",
        observers=np.linspace((0, 0, -L / 2), (0, 0, L / 2), N),
        diameter=np.array([2] * N),
        current=np.array([I] * N),
    )
    bz = np.sum(B, axis=0)[2]
    bz_test = N * I / L

    np.testing.assert_allclose(bz, bz_test, rtol=1e-3)


# dipole vs other magnets
# solenoid formula
# approximate magnets with currents

#######################################################################################
#######################################################################################
#######################################################################################

# FIELD COMPUTATION TESTS AGAINST OTHER SOFTWARE

# def test_field_dipole_VS_mathematica():
#     """Test standard dipole field output computed with mathematica"""
#     obs = np.array([(1, 2, 3), (-1, 2, 3)])
#     mom = np.array([(2, 3, 4), (0, -3, -2)])/MU0
#     B = magpy.core.dipole_field(
#         field="B",
#         observers=obs,
#         moment=mom,
#     )*np.pi
#     Btest = np.array(
#         [
#             (0.01090862, 0.02658977, 0.04227091),
#             (0.0122722, -0.01022683, -0.02727156),
#         ]
#     )
#     assert_allclose(B, Btest, rtol=1e-6)


def test_core_other_circle():
    """
    Compare Circle on-axis field vs e-magnetica & hyperphysics
    """
    dia = np.array([2] * 4)
    curr = np.array([1e3] * 4)  # A
    zs = [0, 1, 2, 3]
    obs = np.array([(0, 0, z) for z in zs])

    # values from e-magnetica
    Hz = [500, 176.8, 44.72, 15.81]
    Htest = [(0, 0, hz) for hz in Hz]

    H = magpy.core.current_circle_field(
        field="H",
        observers=obs,
        diameter=dia,
        current=curr,
    )
    np.testing.assert_allclose(H, Htest, rtol=1e-3)

    # values from hyperphysics
    Bz = [
        0.6283185307179586e-3,
        2.2214414690791835e-4,
        5.619851784832581e-5,
        1.9869176531592205e-5,
    ]
    Btest = [(0, 0, bz) for bz in Bz]

    B = magpy.core.current_circle_field(
        field="B",
        observers=obs,
        diameter=dia,
        current=curr,
    )
    np.testing.assert_allclose(B, Btest, rtol=1e-7)


#######################################################################################
#######################################################################################
#######################################################################################

# OLD FIELD COMPUTATION TESTS - SPECIAL CASES


def test_field_loop_specials():
    """test loop special cases"""
    cur = np.array([1, 1, 1, 1, 0, 2])
    dia = np.array([2, 2, 0, 0, 2, 2])
    obs = np.array([(0, 0, 0), (1, 0, 0), (0, 0, 0), (1, 0, 0), (1, 0, 0), (0, 0, 0)])

    B = current_circle_field(
        field="B",
        observers=obs,
        diameter=dia,
        current=cur,
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
    B1 = current_polyline_field(
        field="B",
        observers=po1,
        current=c1,
        segment_start=ps1,
        segment_end=pe1,
    )
    x1 = np.array([[0.02672612, -0.05345225, 0.02672612]])
    assert_allclose(x1, B1, rtol=1e-6)

    # only on_line
    po1b = np.array([(1, 1, 1)])
    B2 = current_polyline_field(
        field="B",
        observers=po1b,
        current=c1,
        segment_start=ps1,
        segment_end=pe1,
    )
    x2 = np.zeros((1, 3))
    assert_allclose(x2, B2, rtol=1e-6)

    # only zero-segment
    B3 = current_polyline_field(
        field="B",
        observers=po1,
        current=c1,
        segment_start=ps1,
        segment_end=ps1,
    )
    x3 = np.zeros((1, 3))
    assert_allclose(x3, B3, rtol=1e-6)

    # only on_line and zero_segment
    c2 = np.array([1] * 2)
    ps2 = np.array([(0, 0, 0)] * 2)
    pe2 = np.array([(0, 0, 0), (2, 2, 2)])
    po2 = np.array([(1, 2, 3), (1, 1, 1)])
    B4 = current_polyline_field(
        field="B",
        observers=po2,
        current=c2,
        segment_start=ps2,
        segment_end=pe2,
    )
    x4 = np.zeros((2, 3))
    assert_allclose(x4, B4, rtol=1e-6)

    # normal + zero_segment
    po2b = np.array([(1, 2, 3), (1, 2, 3)])
    B5 = current_polyline_field(
        field="B",
        observers=po2b,
        current=c2,
        segment_start=ps2,
        segment_end=pe2,
    )
    x5 = np.array([[0, 0, 0], [0.02672612, -0.05345225, 0.02672612]])
    assert_allclose(x5, B5, rtol=1e-6)

    # normal + on_line
    pe2b = np.array([(2, 2, 2)] * 2)
    B6 = current_polyline_field(
        field="B",
        observers=po2,
        current=c2,
        segment_start=ps2,
        segment_end=pe2b,
    )
    x6 = np.array([[0.02672612, -0.05345225, 0.02672612], [0, 0, 0]])
    assert_allclose(x6, B6, rtol=1e-6)

    # normal + zero_segment + on_line
    c4 = np.array([1] * 3)
    ps4 = np.array([(0, 0, 0)] * 3)
    pe4 = np.array([(0, 0, 0), (2, 2, 2), (2, 2, 2)])
    po4 = np.array([(1, 2, 3), (1, 2, 3), (1, 1, 1)])
    B7 = current_polyline_field(
        field="B",
        observers=po4,
        current=c4,
        segment_start=ps4,
        segment_end=pe4,
    )
    x7 = np.array([[0, 0, 0], [0.02672612, -0.05345225, 0.02672612], [0, 0, 0]])
    assert_allclose(x7, B7, rtol=1e-6)


#######################################################################################
#######################################################################################
#######################################################################################

# OTHER


# def test_field_loop2():
#     """test if field function accepts correct inputs"""
#     curr = np.array([1])
#     dim = np.array([2])
#     poso = np.array([[0, 0, 0]])
#     B = current_circle_field("B", poso, curr, dim)

#     curr = np.array([1] * 2)
#     dim = np.array([2] * 2)
#     poso = np.array([[0, 0, 0]] * 2)
#     B2 = current_circle_field("B", poso, curr, dim)

#     assert_allclose(B, (B2[0],))
#     assert_allclose(B, (B2[1],))


def test_line_deprecation():
    with pytest.warns(MagpylibDeprecationWarning):
        x = current_line_field("B", 1, 2, 3, 4)


def test_loop_deprecation():
    with pytest.warns(MagpylibDeprecationWarning):
        x = current_loop_field("B", 1, 2, 3, 4)


# def test_field_line_from_vert():
#     """test the Polyline field from vertex input"""
#     observers = np.array([(1, 2, 2), (1, 2, 3), (-1, 0, -3)])
#     current = np.array([1, 5, -3])

#     vertices = np.array(
#         [
#             np.array(
#                 [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3), (1, 2, 3), (-3, 4, -5)]
#             ),
#             np.array([(0, 0, 0), (3, 3, 3), (-3, 4, -5)]),
#             np.array([(1, 2, 3), (-2, -3, 3), (3, 2, 1), (3, 3, 3)]),
#         ],
#         dtype="object",
#     )

#     B_vert = current_vertices_field("B", observers, current, vertices)

#     B = []
#     for obs, vert, curr in zip(observers, vertices, current):
#         p1 = vert[:-1]
#         p2 = vert[1:]
#         po = np.array([obs] * (len(vert) - 1))
#         cu = np.array([curr] * (len(vert) - 1))
#         B += [np.sum(current_polyline_field("B", po, cu, p1, p2), axis=0)]
#     B = np.array(B)

#     assert_allclose(B_vert, B)


# def test_field_line_v4():
#     """test current_line_Bfield() for all cases"""
#     cur = np.array([1] * 7)
#     start = np.array([(-1, 0, 0)] * 7)
#     end = np.array([(1, 0, 0), (-1, 0, 0), (1, 0, 0), (-1, 0, 0)] + [(1, 0, 0)] * 3)
#     obs = np.array(
#         [
#             (0, 0, 1),
#             (0, 0, 0),
#             (0, 0, 0),
#             (0, 0, 0),
#             (0, 0, 1e-16),
#             (2, 0, 1),
#             (-2, 0, 1),
#         ]
#     )
#     B = current_polyline_field("B", obs, cur, start, end)
#     Btest = np.array(
#         [
#             [0, -0.14142136, 0],
#             [0, 0.0, 0],
#             [0, 0.0, 0],
#             [0, 0.0, 0],
#             [0, 0.0, 0],
#             [0, -0.02415765, 0],
#             [0, -0.02415765, 0],
#         ]
#     )
#     np.testing.assert_allclose(B, Btest)


# def test_triangle1():
#     """test core triangle VS cube"""
#     obs = np.array([(3, 4, 5)] * 4)
#     mag = np.array([(0, 0, 333)] * 4)
#     fac = np.array(
#         [
#             [(-1, -1, 1), (1, -1, 1), (-1, 1, 1)],  # top1
#             [(1, -1, -1), (-1, -1, -1), (-1, 1, -1)],  # bott1
#             [(1, -1, 1), (1, 1, 1), (-1, 1, 1)],  # top2
#             [(1, 1, -1), (1, -1, -1), (-1, 1, -1)],  # bott2
#         ]
#     )
#     b = magpy.core.triangle_field("B", obs, mag, fac)
#     b = np.sum(b, axis=0)

#     obs = np.array([(3, 4, 5)])
#     mag = np.array([(0, 0, 333)])
#     dim = np.array([(2, 2, 2)])
#     bb = magpy.core.magnet_cuboid_field("B", obs, mag, dim)[0]

#     np.testing.assert_allclose(b, bb)


# def test_triangle2():
#     """test core single triangle vs same surface split up into 4 triangular faces"""
#     obs = np.array([(3, 4, 5)])
#     mag = np.array([(111, 222, 333)])
#     fac = np.array(
#         [
#             [(0, 0, 0), (10, 0, 0), (0, 10, 0)],
#         ]
#     )
#     b = magpy.core.triangle_field("B", obs, mag, fac)
#     b = np.sum(b, axis=0)

#     obs = np.array([(3, 4, 5)] * 4)
#     mag = np.array([(111, 222, 333)] * 4)
#     fac = np.array(
#         [
#             [(0, 0, 0), (3, 0, 0), (0, 10, 0)],
#             [(3, 0, 0), (5, 0, 0), (0, 10, 0)],
#             [(5, 0, 0), (6, 0, 0), (0, 10, 0)],
#             [(6, 0, 0), (10, 0, 0), (0, 10, 0)],
#         ]
#     )
#     bb = magpy.core.triangle_field("B", obs, mag, fac)
#     bb = np.sum(bb, axis=0)

#     np.testing.assert_allclose(b, bb)


# def test_triangle3():
#     """test core tetrahedron vs cube"""
#     ver = np.array(
#         [
#             [(1, 1, -1), (1, 1, 1), (-1, 1, 1), (1, -1, 1)],
#             [(-1, -1, 1), (-1, 1, 1), (1, -1, 1), (1, -1, -1)],
#             [(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)],
#             [(-1, 1, -1), (1, -1, -1), (-1, -1, 1), (-1, 1, 1)],
#             [(1, -1, -1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)],
#             [(-1, 1, -1), (-1, 1, 1), (1, 1, -1), (1, -1, -1)],
#         ]
#     )

#     mags = [
#         [1.03595366, 0.42840487, 0.10797529],
#         [0.33513152, 1.61629547, 0.15959791],
#         [0.29904441, 1.32185041, 1.81218046],
#         [0.82665456, 1.86827489, 1.67338911],
#         [0.97619806, 1.52323106, 1.63628455],
#         [1.70290645, 1.49610608, 0.13878711],
#         [1.49886747, 1.55633919, 1.41351862],
#         [0.9959534, 0.62059942, 1.28616663],
#         [0.60114354, 0.96120344, 0.32009221],
#         [0.83133901, 0.7925518, 0.64574592],
#     ]

#     obss = [
#         [0.82811352, 1.77818627, 0.19819379],
#         [0.84147235, 1.10200857, 1.51687527],
#         [0.30751474, 0.89773196, 0.56468564],
#         [1.87437889, 1.55908581, 1.10579983],
#         [0.64810548, 1.38123846, 1.90576802],
#         [0.48981034, 0.09376294, 0.53717129],
#         [1.42826412, 0.30246674, 0.57649909],
#         [1.58376758, 1.70420478, 0.22894022],
#         [0.26791832, 0.36839769, 0.67934335],
#         [1.15140149, 0.10549875, 0.98304184],
#     ]

#     for mag in mags:
#         for obs in obss:
#             obs6 = np.tile(obs, (6, 1))
#             mag6 = np.tile(mag, (6, 1))
#             b = magpy.core.magnet_tetrahedron_field("B", obs6, mag6, ver)
#             h = magpy.core.magnet_tetrahedron_field("H", obs6, mag6, ver)
#             b = np.sum(b, axis=0)
#             h = np.sum(h, axis=0)

#             obs1 = np.reshape(obs, (1, 3))
#             mag1 = np.reshape(mag, (1, 3))
#             dim = np.array([(2, 2, 2)])
#             bb = magpy.core.magnet_cuboid_field("B", obs1, mag1, dim)[0]
#             hh = magpy.core.magnet_cuboid_field("H", obs1, mag1, dim)[0]
#             np.testing.assert_allclose(b, bb)
#             np.testing.assert_allclose(h, hh)


# def test_triangle4():
#     """test core tetrahedron vs cube"""
#     obs = np.array([(3, 4, 5)] * 6)
#     mag = np.array([(111, 222, 333)] * 6)
#     ver = np.array(
#         [
#             [(1, 1, -1), (1, 1, 1), (-1, 1, 1), (1, -1, 1)],
#             [(-1, -1, 1), (-1, 1, 1), (1, -1, 1), (1, -1, -1)],
#             [(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)],
#             [(-1, 1, -1), (1, -1, -1), (-1, -1, 1), (-1, 1, 1)],
#             [(1, -1, -1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)],
#             [(-1, 1, -1), (-1, 1, 1), (1, 1, -1), (1, -1, -1)],
#         ]
#     )
#     b = magpy.core.magnet_tetrahedron_field("B", obs, mag, ver)
#     b = np.sum(b, axis=0)

#     obs = np.array([(3, 4, 5)])
#     mag = np.array([(111, 222, 333)])
#     dim = np.array([(2, 2, 2)])
#     bb = magpy.core.magnet_cuboid_field("B", obs, mag, dim)[0]

#     np.testing.assert_allclose(b, bb)


# def test_triangle5():
#     """special case tests on edges - result is continuous and 0 for vertical component"""
#     btest1 = [
#         [26.29963526814195, 15.319834473660082, 0.0],
#         [54.91549594789228, 41.20535983076747, 0.0],
#         [32.25241487782939, 15.087161660417559, 0.0],
#         [10.110611199952707, -11.41176203622237, 0.0],
#         [-3.8084378251737285, -30.875600143560657, -0.0],
#         [-15.636505140623612, -50.00854548249858, -0.0],
#         [-27.928308992688645, -72.80800891847107, -0.0],
#         [-45.34417750711242, -109.5871836961927, -0.0],
#         [-36.33970306054345, 12.288824457077656, 0.0],
#         [-16.984738462958845, 4.804383318447626, 0.0],
#     ]

#     btest2 = [
#         [15.31983447366009, 26.299635268142033, 0.0],
#         [41.20535983076747, 54.91549594789104, 0.0],
#         [-72.61316618947018, 32.25241487782958, 0.0],
#         [-54.07597251255013, 10.110611199952693, 0.0],
#         [-44.104089712909634, -3.808437825173785, -0.0],
#         [-36.78005591314963, -15.636505140623605, -0.0],
#         [-30.143798442143236, -27.92830899268858, -0.0],
#         [-21.886855846306176, -45.34417750711366, -0.0],
#         [12.288824457077965, -36.33970306054315, 0.0],
#         [4.80438331844773, -16.98473846295874, 0.0],
#     ]

#     n = 10
#     ts = np.linspace(-1, 6, n)
#     obs1 = np.array([(t, 0, 0) for t in ts])
#     obs2 = np.array([(0, t, 0) for t in ts])
#     mag = np.array([(111, 222, 333)] * n)
#     ver = np.array([[(0, 0, 0), (0, 5, 0), (5, 0, 0)]] * n)

#     b1 = magpy.core.triangle_field("H", obs1, mag, ver)
#     np.testing.assert_allclose(btest1, b1)
#     b2 = magpy.core.triangle_field("H", obs2, mag, ver)
#     np.testing.assert_allclose(btest2, b2)


# def test_triangle6():
#     """special case tests on corners - result is nan"""
#     obs1 = np.array([(0, 0, 0)])
#     obs2 = np.array([(0, 5, 0)])
#     obs3 = np.array([(5, 0, 0)])
#     mag = np.array([(111, 222, 333)])
#     ver = np.array([[(0, 0, 0), (0, 5, 0), (5, 0, 0)]])
#     b1 = magpy.core.triangle_field("B", obs1, mag, ver)
#     b2 = magpy.core.triangle_field("B", obs2, mag, ver)
#     b3 = magpy.core.triangle_field("B", obs3, mag, ver)

#     for b in [b1, b2, b3]:
#         np.testing.assert_equal(b, np.array([[np.nan] * 3]))


# @pytest.mark.parametrize(
#     ("module", "class_", "missing_arg"),
#     [
#         ("magnet", "Cuboid", "dimension"),
#         ("magnet", "Cylinder", "dimension"),
#         ("magnet", "CylinderSegment", "dimension"),
#         ("magnet", "Sphere", "diameter"),
#         ("magnet", "Tetrahedron", "vertices"),
#         ("magnet", "TriangularMesh", "vertices"),
#         ("current", "Circle", "diameter"),
#         ("current", "Polyline", "vertices"),
#         ("misc", "Triangle", "vertices"),
#     ],
# )
# def test_getB_on_missing_dimensions(module, class_, missing_arg):
#     """test_getB_on_missing_dimensions"""
#     with pytest.raises(
#         MagpylibMissingInput,
#         match=rf"Parameter `{missing_arg}` of .* must be set.",
#     ):
#         getattr(getattr(magpy, module), class_)().getB([0, 0, 0])


# @pytest.mark.parametrize(
#     ("module", "class_", "missing_arg", "kwargs"),
#     [
#         ("magnet", "Cuboid", "magnetization", {"dimension": (1, 1, 1)}),
#         ("magnet", "Cylinder", "magnetization", {"dimension": (1, 1)}),
#         (
#             "magnet",
#             "CylinderSegment",
#             "magnetization",
#             {"dimension": (0, 1, 1, 45, 120)},
#         ),
#         ("magnet", "Sphere", "magnetization", {"diameter": 1}),
#         (
#             "magnet",
#             "Tetrahedron",
#             "magnetization",
#             {"vertices": [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]},
#         ),
#         (
#             "magnet",
#             "TriangularMesh",
#             "magnetization",
#             {
#                 "vertices": ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
#                 "faces": ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
#             },
#         ),
#         ("current", "Circle", "current", {"diameter": 1}),
#         ("current", "Polyline", "current", {"vertices": [[0, -1, 0], [0, 1, 0]]}),
#         (
#             "misc",
#             "Triangle",
#             "magnetization",
#             {"vertices": [(0, 0, 0), (1, 0, 0), (0, 1, 0)]},
#         ),
#         ("misc", "Dipole", "moment", {}),
#     ],
# )
# def test_getB_on_missing_excitations(module, class_, missing_arg, kwargs):
#     """test_getB_on_missing_excitations"""
#     with pytest.raises(
#         MagpylibMissingInput,
#         match=rf"Parameter `{missing_arg}` of .* must be set.",
#     ):
#         getattr(getattr(magpy, module), class_)(**kwargs).getB([0, 0, 0])

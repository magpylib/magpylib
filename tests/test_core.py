# here all core functions should be tested properly - ideally against FEM
from __future__ import annotations

import numpy as np

from magpylib.core import (
    current_circle_Hfield,
    current_polyline_Hfield,
    dipole_Hfield,
    magnet_cuboid_Bfield,
    magnet_cylinder_axial_Bfield,
    magnet_cylinder_diametral_Hfield,
    magnet_cylinder_segment_Hfield,
    magnet_sphere_Bfield,
    triangle_Bfield,
)


def test_magnet_sphere_Bfield():
    "magnet_sphere_Bfield test"
    B = magnet_sphere_Bfield(
        observers=np.array([(0, 0, 0)]),
        diameters=np.array([1]),
        polarizations=np.array([(0, 0, 1)]),
    )
    Btest = np.array([(0, 0, 2 / 3)])
    np.testing.assert_allclose(B, Btest)


def test_current_circle_Hfield():
    Htest = np.array([[0.09098208, 0.09415448], [0.0, 0.0], [0.07677892, 0.22625335]])
    H = current_circle_Hfield(
        r0=np.array([1, 2]),
        r=np.array([1, 1]),
        z=np.array([1, 2]),
        i0=np.array([1, 3]),
    )
    np.testing.assert_allclose(H, Htest)


def test_current_polyline_Hfield():
    Htest = np.array([[0.0, -2.29720373, 2.29720373], [0.0, 0.59785204, -0.59785204]])

    H = current_polyline_Hfield(
        observers=np.array([(1, 1, 1), (2, 2, 2)]),
        segments_start=np.array([(0, 0, 0), (0, 0, 0)]),
        segments_end=np.array([(1, 0, 0), (-1, 0, 0)]),
        currents=np.array([100, 200]),
    )
    np.testing.assert_allclose(H, Htest)


def test_dipole_Hfield():
    Htest = np.array(
        [
            [2.89501155e-13, 1.53146915e03, 1.53146915e03],
            [1.91433644e02, 1.91433644e02, 3.61876444e-14],
        ]
    )
    H = dipole_Hfield(
        observers=np.array([(1, 1, 1), (2, 2, 2)]),
        moments=np.array([(1e5, 0, 0), (0, 0, 1e5)]),
    )
    np.testing.assert_allclose(H, Htest)


def test_magnet_cuboid_Bfield():
    Btest = np.array(
        [
            [1.56103722e-02, 1.56103722e-02, -3.53394965e-17],
            [7.73243250e-03, 6.54431406e-03, 1.04789520e-02],
        ]
    )
    B = magnet_cuboid_Bfield(
        observers=np.array([(1, 1, 1), (2, 2, 2)]),
        dimensions=np.array([(1, 1, 1), (1, 2, 3)]),
        polarizations=np.array([(0, 0, 1), (0.5, 0.5, 0)]),
    )
    np.testing.assert_allclose(B, Btest)


def test_magnet_cylinder_axial_Bfield():
    Btest = np.array([[0.05561469, 0.04117919], [0.0, 0.0], [0.06690167, 0.01805674]])
    B = magnet_cylinder_axial_Bfield(
        z0=np.array([1, 2]),
        r=np.array([1, 2]),
        z=np.array([2, 3]),
    )
    np.testing.assert_allclose(B, Btest)


def test_magnet_cylinder_diametral_Hfield():
    Btest = np.array(
        [
            [-0.020742122169014, 0.007307203574376],
            [0.004597868528024, 0.020075245863212],
            [0.05533684822464, 0.029118084290573],
        ],
    )
    B = magnet_cylinder_diametral_Hfield(
        z0=np.array([1, 2]),
        r=np.array([1, 2]),
        z=np.array([2, 3]),
        phi=np.array([0.1, np.pi / 4]),
    )
    np.testing.assert_allclose(B, Btest)


def test_magnet_cylinder_segment_Hfield():
    Btest = np.array(
        [
            [-1948.14367497, 32319.94437208, 17616.88571231],
            [14167.64961763, 1419.94126065, 17921.6463117],
        ]
    )
    B = magnet_cylinder_segment_Hfield(
        observers=np.array([(1, 1, 2), (0, 0, 0)]),
        dimensions=np.array([(1, 2, 0.1, 0.2, -1, 1), (1, 2, 0.3, 0.9, 0, 1)]),
        magnetizations=np.array([(1e7, 0.1, 0.2), (1e6, 1.1, 2.2)]),
    )
    np.testing.assert_allclose(B, Btest)


def test_triangle_Bfield():
    Btest = np.array(
        [[7.45158965, 4.61994866, 3.13614132], [2.21345618, 2.67710148, 2.21345618]]
    )
    B = triangle_Bfield(
        observers=np.array([(2, 1, 1), (2, 2, 2)]),
        vertices=np.array(
            [[(0, 0, 0), (0, 0, 1), (1, 0, 0)], [(0, 0, 0), (0, 0, 1), (1, 0, 0)]]
        ),
        polarizations=np.array([(1, 1, 1), (1, 1, 0)]) * 1e3,
    )
    np.testing.assert_allclose(B, Btest)

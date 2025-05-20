# here all core functions should be tested properly - ideally against FEM
from __future__ import annotations

import numpy as np

from magpylib._src.fields.field_BH_circle import current_circle_Hfield
from magpylib._src.fields.field_BH_sphere import magnet_sphere_Bfield


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
    r = np.sqrt(np.array([2, 4, 6]))
    H = current_circle_Hfield(
        r0=np.array([2, 4, 6]), r=r, z=np.array([1, 2, 3]), i0=np.array([1, 1, 2]) * 1e3
    )
    np.testing.assert_allclose(H, Htest)


def test_current_polyline_Hfield(): ...
def test_dipole_Hfield(): ...
def test_magnet_cuboid_Bfield(): ...
def test_magnet_cylinder_axial_Bfield(): ...
def test_magnet_cylinder_diametral_Hfield(): ...
def test_magnet_cylinder_segment_Hfield(): ...
def test_triangle_Bfield(): ...

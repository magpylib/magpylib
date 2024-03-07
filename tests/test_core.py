# here all core functions should be tested properly - ideally against FEM
import numpy as np

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

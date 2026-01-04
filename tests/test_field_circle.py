import numpy as np
import magpylib as magpy
from magpylib._src.fields.field_BH_circle import current_circle_Hfield
import pytest


def test_current_circle_on_axis():
    """testing H field on axis of current cirlce"""
    z = np.linspace(-3, 3, 100)
    H = current_circle_Hfield(
        r0 = np.ones_like(z),
        r = np.zeros_like(z),
        z = z,
        i0 = np.ones_like(z)
    )
    Hr = H[0,:]
    Hz = H[2,:]

    Hz_expected = 1 / (2 * (1+z*z)**(3/2))
    np.testing.assert_array_almost_equal_nulp(Hz, Hz_expected, nulp=2)
    np.testing.assert_array_equal(Hr, np.zeros_like(z))

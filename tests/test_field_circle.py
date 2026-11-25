import numpy as np
import pytest

import magpylib as magpy
from magpylib._src.fields.field_BH_circle import current_circle_Hfield


def test_current_circle_on_axis():
    """testing H field on axis of current circle"""
    z = np.linspace(-3, 3, 100)
    H = current_circle_Hfield(
        r0=np.ones_like(z), r=np.zeros_like(z), z=z, i0=np.ones_like(z)
    )
    Hr = H[0, :]
    Hz = H[2, :]

    Hz_expected = 1 / (2 * (1 + z * z) ** (3 / 2))
    np.testing.assert_array_almost_equal_nulp(Hz, Hz_expected, nulp=2)
    np.testing.assert_array_equal(Hr, np.zeros_like(z))


@pytest.mark.parametrize("z", [0.0, 2e-162, 1e-170, 5e-324])
def test_current_circle_on_wire_returns_zero(z):
    """on the wire, and next to it where z**2 underflows, return 0 instead of hanging"""
    r = np.array([1.0, 1.0, 0.5])
    H = current_circle_Hfield(
        r0=np.ones(3), r=r, z=np.array([z, -z, 0.3]), i0=np.ones(3)
    )
    np.testing.assert_array_equal(H[:, :2], 0)
    H_other = current_circle_Hfield(
        r0=np.ones(1), r=r[2:], z=np.array([0.3]), i0=np.ones(1)
    )
    np.testing.assert_array_equal(H[:, 2:], H_other)


@pytest.mark.parametrize("z", [2e-162, 1e-170, 5e-324])
@pytest.mark.parametrize("n", [1, 12])
def test_circle_next_to_wire_with_underflowing_height(z, n):
    """observers at the wire radius and a tiny height, scalar and vectorized paths"""
    observers = [(1, 0, z)] * n
    H = magpy.func.circle_field("H", observers, 2, 1)
    np.testing.assert_array_equal(H, 0)
    H = magpy.current.Circle(diameter=2, current=1).getH(observers)
    np.testing.assert_array_equal(H, 0)

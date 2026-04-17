import numpy as np

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


def test_current_circle_far_away():
    """Testing H field far away from a current circle.

    This test will fail for implementations based on just ellipk/e or on cel directly;
    implementation based on cel_iter or on cel preceded by
    an analytical Bartky transformation will pass this test.
    """
    z = 0
    r = 1e8
    H = current_circle_Hfield(
        r0=np.array([1]), r=np.array([r]), z=np.array([z]), i0=np.array([1])
    )
    Hz = H[2, 0]
    Hz_expected = -1 / (4 * r**3)  # dipole approximation
    np.testing.assert_array_almost_equal_nulp(Hz, Hz_expected, nulp=2)

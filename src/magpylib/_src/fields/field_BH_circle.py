"""Circular current loop field implementation."""

import numpy as np
from scipy.constants import mu_0 as MU0

from magpylib._src.array_api_utils import (
    array_namespace,
    jax_padded,
    some,
    xp_promote,
)
from magpylib._src.fields.special_cel import _cel_iter
from magpylib._src.input_checks import check_field_input
from magpylib._src.utility import cart_to_cyl_coordinates, cyl_field_to_cart


@jax_padded(out_axis=1)
def current_circle_Hfield(
    r0: np.ndarray,
    r: np.ndarray,
    z: np.ndarray,
    i0: np.ndarray,
) -> np.ndarray:
    """H-field of i circular line-current loops in cylindrical coordinates.

    The loop lies in the ``z=0`` plane with the coordinate origin at its center.
    The output is proportional to the electrical current ``i0`` in units (A) and independent
    of the length units chosen for observers and loop radii. The returned field
    is in units (A/m).

    Parameters
    ----------
    r0 : array-like, shape (i,)
        Loop radii, should be positive (r0 > 0).
    r : array-like, shape (i,)
        Radial observer positions.
    z : array-like, shape (i,)
        Axial observer positions.
    i0 : array-like, shape (i,)
        Electrical currents in units (A).

    Returns
    -------
    ndarray, shape (3, i)
        H-field in (A/m) at observer positions in cylindrical coordinates
        (H_r, H_φ, H_z). The azimuthal component is zero for symmetry.

    Notes
    -----
    Implementation based on "Numerically stable and computationally efficient
    expression for the magnetic field of a current loop.", M. Ortner et al.,
    Magnetism 2023, 3(1), 11-31.

    Examples
    --------
    >>> import numpy as np
    >>> import magpylib as magpy
    >>> H = magpy.core.current_circle_Hfield(
    ...     r0=np.array([1, 2]),
    ...     r=np.array([1, 1]),
    ...     z=np.array([1, 2]),
    ...     i0=np.array([1, 3]),
    ... )
    >>> print(H.round(3))
    [[0.091 0.094]
     [0.    0.   ]
     [0.077 0.226]]

    """
    xp = array_namespace(r0, r, z, i0)
    r0, r, z, i0 = xp_promote(r0, r, z, i0, xp=xp)

    # express through ratios (make dimensionless, avoid large/small input values, stupid)
    r = r / r0
    z = z / r0

    # field computation from paper
    z2 = z**2
    x0 = z2 + (r + 1) ** 2
    k2 = 4 * r / x0
    q2 = (z2 + (r - 1) ** 2) / x0

    q = xp.sqrt(q2)
    p = 1 + q
    pf = i0 / (4 * xp.pi * r0 * xp.sqrt(x0) * q2)
    one = xp.ones_like(r)

    # cel* part
    cc = k2 * 4 * z / x0
    ss = 2 * cc * q / p
    Hr = pf * _cel_iter(q, p, one, cc, ss, p, q)

    # cel** part
    k4 = k2 * k2
    cc = k4 - (q2 + 1) * (4 / x0)
    ss = 2 * q * (k4 / p - (4 / x0) * p)
    Hz = -pf * _cel_iter(q, p, one, cc, ss, p, q)
    # Hz = -_cel_iter(q, p, one, pf*cc, pf*ss, p, q)

    return xp.stack((Hr, xp.zeros_like(Hr), Hz))


def _BHJM_circle(
    field: str,
    observers: np.ndarray,
    diameter: np.ndarray,
    current: np.ndarray,
) -> np.ndarray:
    """
    - translate circle core to BHJM
    - treat special cases
    """

    check_field_input(field)

    xp = array_namespace(observers, diameter, current)
    observers, diameter, current = xp_promote(observers, diameter, current, xp=xp)

    if field in "MJ":
        return xp.zeros_like(observers)

    r, phi, z = cart_to_cyl_coordinates(observers)
    r0 = xp.abs(diameter / 2)

    # Special cases:
    # case1: loop radius is 0 -> return (0, 0, 0)
    mask1 = r0 == 0
    # case2: at singularity -> return (0, 0, 0)
    mask2 = (xp.abs(r - r0) < 1e-15 * r0) & (z == 0)

    # general case
    mask3 = ~(mask1 | mask2)

    # the special cases, if there are any, are computed as the center of a
    # unit loop to stay clear of the singularities
    special = some(~mask3)
    if special:
        r0 = xp.where(mask3, r0, 1.0)
        r = xp.where(mask3, r, 0.0)
        z = xp.where(mask3, z, 0.0)
    H = current_circle_Hfield(r0=r0, r=r, z=z, i0=current)
    Hr, Hz = H[0, :], H[2, :]
    if special:
        Hr = xp.where(mask3, Hr, 0.0)
        Hz = xp.where(mask3, Hz, 0.0)

    Hx, Hy = cyl_field_to_cart(phi, Hr)
    BHJM = xp.stack((Hx, Hy, Hz), axis=-1)

    if field == "H":
        return BHJM

    if field == "B":
        return BHJM * MU0

    msg = (
        "Input output_field_type must be one of {'B', 'H', 'M', 'J'}; "
        f"instead received {field!r}."
    )
    raise ValueError(  # pragma: no cover
        msg
    )

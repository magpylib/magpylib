"""Circular current loop field implementation."""

import numpy as np
from scipy.constants import mu_0 as MU0

from magpylib._src.fields.special_cel import _cel_iter
from magpylib._src.input_checks import check_field_input
from magpylib._src.utility import cart_to_cyl_coordinates, cyl_field_to_cart


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
    n5 = len(r)

    # express through ratios (make dimensionless, avoid large/small input values, stupid)
    r = r / r0
    z = z / r0

    # field computation from paper
    z2 = z**2
    x0 = z2 + (r + 1) ** 2
    k2 = 4 * r / x0
    q2 = (z2 + (r - 1) ** 2) / x0

    q = np.sqrt(q2)
    p = 1 + q
    pf = i0 / (4 * np.pi * r0 * np.sqrt(x0) * q2)

    # cel* part
    cc = k2 * 4 * z / x0
    ss = 2 * cc * q / p
    Hr = pf * _cel_iter(q, p, np.ones(n5), cc, ss, p, q)

    # cel** part
    k4 = k2 * k2
    cc = k4 - (q2 + 1) * (4 / x0)
    ss = 2 * q * (k4 / p - (4 / x0) * p)
    Hz = -pf * _cel_iter(q, p, np.ones(n5), cc, ss, p, q)
    # Hz = -_cel_iter(q, p, np.ones(n5), pf*cc, pf*ss, p, q)

    return np.vstack((Hr, np.zeros(n5), Hz))


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

    # allocate
    BHJM = np.zeros_like(observers, dtype=float)

    check_field_input(field)
    if field in "MJ":
        return BHJM

    r, phi, z = cart_to_cyl_coordinates(observers)
    r0 = np.abs(diameter / 2)

    # Special cases:
    # case1: loop radius is 0 -> return (0, 0, 0)
    mask1 = r0 == 0
    # case2: at singularity -> return (0, 0, 0)
    mask2 = np.logical_and(abs(r - r0) < 1e-15 * r0, z == 0)

    # general case
    mask3 = ~np.logical_or(mask1, mask2)
    if np.any(mask3):
        BHJM[mask3] = current_circle_Hfield(
            r0=r0[mask3],
            r=r[mask3],
            z=z[mask3],
            i0=current[mask3],
        ).T

    BHJM[:, 0], BHJM[:, 1] = cyl_field_to_cart(phi, BHJM[:, 0])

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

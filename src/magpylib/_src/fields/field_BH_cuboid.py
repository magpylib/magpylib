"""Homogeneously magnetized cuboid field implementation."""

# pylint: disable=too-many-statements

import numpy as np
from array_api_compat import array_namespace
from scipy.constants import mu_0 as MU0

from magpylib._src.array_api_utils import xp_promote
from magpylib._src.input_checks import check_field_input


def magnet_cuboid_Bfield(
    observers: np.ndarray,
    dimensions: np.ndarray,
    polarizations: np.ndarray,
):
    """B-field of i homogeneously magnetized cuboids in Cartesian coordinates.

    Cuboid sides are parallel to the coordinate axes and the geometric center
    lies at the origin. The output is proportional to the polarization magnitude
    and independent of the length units chosen for observers and dimensions.

    Parameters
    ----------
    observers : array-like, shape (i, 3)
        Observer positions (x, y, z) in Cartesian coordinates.
    dimensions : array-like, shape (i, 3)
        Lengths of the cuboid sides (a, b, c).
    polarizations : array-like, shape (i, 3)
        Magnetic polarization vectors ``J``.

    Returns
    -------
    ndarray, shape (i, 3)
        B-field in units of input polarization at the observer positions.

    Notes
    -----
    Field computations use magnetic surface charge density; similar expressions
    were published repeatedly, e.g. Yang (Supercond. Sci. Technol. 3(12):591,
    1990), Engel-Herbert (J. Appl. Phys. 97(7):074504-4, 2005), and Camacho
    (Rev. Mex. Fís. E 59 (2013) 8-17).

    Avoiding indeterminate forms: at positions along extensions of edges (in all
    xyz-octants except the bottom negative-x/positive-y/positive-z quadrant),
    direct evaluation becomes numerically unstable. We exploit symmetry to map
    positions to their bottom-quadrant counterparts; see also Cichon, IEEE
    Sensors Journal 19(7):2509, 2019.

    Examples
    --------
    >>> import numpy as np
    >>> import magpylib as magpy
    >>> B = magpy.core.magnet_cuboid_Bfield(
    ...     observers=np.array([(1, 1, 1), (2, 2, 2)]),
    ...     dimensions=np.array([(1, 1, 1), (1, 2, 3)]),
    ...     polarizations=np.array([(0, 0, 1), (0.5, 0.5, 0.0)]),
    ... )
    >>> with np.printoptions(precision=3):
    ...     print(B)  # doctest: +ELLIPSIS
    [[ 1.561e-02  1.561e-02 ...]
     [ 7.732e-03  6.544e-03  1.048e-02]]

    """
    xp = array_namespace(observers, dimensions, polarizations)
    observers, dimensions, polarizations = xp_promote(
        observers, dimensions, polarizations, xp=xp
    )

    pol_x, pol_y, pol_z = xp.unstack(polarizations, axis=-1)
    a, b, c = xp.unstack(dimensions / 2, axis=-1)
    x, y, z = xp.unstack(observers, axis=-1)

    # avoid indeterminate forms by evaluating in bottQ4 only --------
    # basic masks
    maskx = x < 0
    masky = y > 0
    maskz = z > 0

    # change all positions to their bottQ4 counterparts
    x = xp.where(maskx, -x, x)
    y = xp.where(masky, -y, y)
    z = xp.where(maskz, -z, z)

    # sign flips for position changes: a field component changes sign with
    # every flipped axis that is either its own or that of the polarization
    sign_x = xp.where(maskx, -1.0, 1.0)
    sign_y = xp.where(masky, -1.0, 1.0)
    sign_z = xp.where(maskz, -1.0, 1.0)
    sign_xy = sign_x * sign_y
    sign_xz = sign_x * sign_z
    sign_yz = sign_y * sign_z

    # field computations --------------------------------------------
    # Note: in principle the computation for all three polarization-components can be
    #   vectorized itself using symmetries. However, tiling the three
    #   components will cost more than is gained by the vectorized evaluation

    xma, xpa = x - a, x + a
    ymb, ypb = y - b, y + b
    zmc, zpc = z - c, z + c

    xma2, xpa2 = xma**2, xpa**2
    ymb2, ypb2 = ymb**2, ypb**2
    zmc2, zpc2 = zmc**2, zpc**2

    mmm = xp.sqrt(xma2 + ymb2 + zmc2)
    pmp = xp.sqrt(xpa2 + ymb2 + zpc2)
    pmm = xp.sqrt(xpa2 + ymb2 + zmc2)
    mmp = xp.sqrt(xma2 + ymb2 + zpc2)
    mpm = xp.sqrt(xma2 + ypb2 + zmc2)
    ppp = xp.sqrt(xpa2 + ypb2 + zpc2)
    ppm = xp.sqrt(xpa2 + ypb2 + zmc2)
    mpp = xp.sqrt(xma2 + ypb2 + zpc2)

    # next to the extension of an edge an argument is 0: the result is inf
    # there, which NumPy would warn about
    with np.errstate(divide="ignore", invalid="ignore"):
        ff2x = xp.log((xma + mmm) * (xpa + ppm) * (xpa + pmp) * (xma + mpp)) - xp.log(
            (xpa + pmm) * (xma + mpm) * (xma + mmp) * (xpa + ppp)
        )

        ff2y = xp.log(
            (-ymb + mmm) * (-ypb + ppm) * (-ymb + pmp) * (-ypb + mpp)
        ) - xp.log((-ymb + pmm) * (-ypb + mpm) * (ymb - mmp) * (ypb - ppp))

        ff2z = xp.log(
            (-zmc + mmm) * (-zmc + ppm) * (-zpc + pmp) * (-zpc + mpp)
        ) - xp.log((-zmc + pmm) * (zmc - mpm) * (-zpc + mmp) * (zpc - ppp))

    ff1x = (
        xp.atan2((ymb * zmc), (xma * mmm))
        - xp.atan2((ymb * zmc), (xpa * pmm))
        - xp.atan2((ypb * zmc), (xma * mpm))
        + xp.atan2((ypb * zmc), (xpa * ppm))
        - xp.atan2((ymb * zpc), (xma * mmp))
        + xp.atan2((ymb * zpc), (xpa * pmp))
        + xp.atan2((ypb * zpc), (xma * mpp))
        - xp.atan2((ypb * zpc), (xpa * ppp))
    )

    ff1y = (
        xp.atan2((xma * zmc), (ymb * mmm))
        - xp.atan2((xpa * zmc), (ymb * pmm))
        - xp.atan2((xma * zmc), (ypb * mpm))
        + xp.atan2((xpa * zmc), (ypb * ppm))
        - xp.atan2((xma * zpc), (ymb * mmp))
        + xp.atan2((xpa * zpc), (ymb * pmp))
        + xp.atan2((xma * zpc), (ypb * mpp))
        - xp.atan2((xpa * zpc), (ypb * ppp))
    )

    ff1z = (
        xp.atan2((xma * ymb), (zmc * mmm))
        - xp.atan2((xpa * ymb), (zmc * pmm))
        - xp.atan2((xma * ypb), (zmc * mpm))
        + xp.atan2((xpa * ypb), (zmc * ppm))
        - xp.atan2((xma * ymb), (zpc * mmp))
        + xp.atan2((xpa * ymb), (zpc * pmp))
        + xp.atan2((xma * ypb), (zpc * mpp))
        - xp.atan2((xpa * ypb), (zpc * ppp))
    )

    # contributions from x-, y- and z-polarization
    #    the 'missing' third sign is hidden in ff1x, ff1y, ff1z
    bx_tot = pol_x * ff1x + pol_y * ff2z * sign_xy + pol_z * ff2y * sign_xz
    by_tot = pol_x * ff2z * sign_xy + pol_y * ff1y - pol_z * ff2x * sign_yz
    bz_tot = pol_x * ff2y * sign_xz - pol_y * ff2x * sign_yz + pol_z * ff1z

    return xp.stack((bx_tot, by_tot, bz_tot), axis=-1) / (4 * xp.pi)


def _BHJM_magnet_cuboid(
    field: str,
    observers: np.ndarray,
    dimension: np.ndarray,
    polarization: np.ndarray,
) -> np.ndarray:
    """
    - translate cuboid core field to BHJM
    - treat special cases
    """

    RTOL_SURFACE = 1e-15  # relative distance tolerance to be considered on surface

    check_field_input(field)

    xp = array_namespace(observers, dimension, polarization)
    observers, dimension, polarization = xp_promote(
        observers, dimension, polarization, xp=xp
    )

    half = xp.abs(dimension) / 2
    dist = xp.abs(observers) - half  # distance to the faces, per axis

    # SPECIAL CASE 1: polarization = (0, 0, 0)
    mask_pol_not_null = xp.any(polarization != 0, axis=-1)

    # SPECIAL CASE 2: 0 in dimension
    mask_dim_not_null = xp.all(half != 0, axis=-1)

    # SPECIAL CASE 3: observer lies on-edge/corner
    #   RTOL_SURFACE to account for numerical imprecision when e.g. rotating,
    #   relative to the side length
    # on-surface is not a special case
    mask_surf = xp.abs(dist) < RTOL_SURFACE * half
    mask_surf_x, mask_surf_y, mask_surf_z = xp.unstack(mask_surf, axis=-1)

    # inside-outside
    mask_inside_xyz = dist < RTOL_SURFACE * half
    mask_inside_x, mask_inside_y, mask_inside_z = xp.unstack(mask_inside_xyz, axis=-1)
    mask_inside = xp.all(mask_inside_xyz, axis=-1, keepdims=True)

    # on edge (requires on-surface and inside-outside)
    mask_xedge = mask_surf_y & mask_surf_z & mask_inside_x
    mask_yedge = mask_surf_x & mask_surf_z & mask_inside_y
    mask_zedge = mask_surf_x & mask_surf_y & mask_inside_z
    mask_not_edge = ~(mask_xedge | mask_yedge | mask_zedge)

    mask_gen = (mask_pol_not_null & mask_dim_not_null & mask_not_edge)[:, xp.newaxis]

    if field == "J":
        return xp.where(mask_inside, polarization, 0.0)

    if field == "M":
        return xp.where(mask_inside, polarization, 0.0) / MU0

    # return (0, 0, 0) for all special cases, which are computed as the center
    # of a unit cube to stay clear of the singularities
    BHJM = xp.where(
        mask_gen,
        magnet_cuboid_Bfield(
            observers=xp.where(mask_gen, observers, 0.0),
            dimensions=xp.where(mask_gen, dimension, 1.0),
            polarizations=polarization,
        ),
        0.0,
    )

    if field == "B":
        return BHJM

    if field == "H":
        return (BHJM - xp.where(mask_inside, polarization, 0.0)) / MU0

    msg = (
        "Input output_field_type must be one of {'B', 'H', 'M', 'J'}; "
        f"instead received {field!r}."
    )
    raise ValueError(  # pragma: no cover
        msg
    )

"""Homogeneously magnetized cylinder field implementation."""

# pylint: disable = no-name-in-module

import numpy as np
from scipy.constants import mu_0 as MU0
from scipy.special import ellipe, ellipk

from magpylib._src.fields.special_cel import _cel
from magpylib._src.input_checks import check_field_input
from magpylib._src.utility import cart_to_cyl_coordinates, cyl_field_to_cart


def magnet_cylinder_axial_Bfield(z0: np.ndarray, r: np.ndarray, z: np.ndarray) -> list:
    """B-field of i axially magnetized cylinders in cylindrical coordinates.

    The cylinder axes coincide with the z-axis of the Cylindrical CS and the
    geometric center of the cylinder lies in the origin. Length inputs are
    made dimensionless by division over the cylinder radii. Unit polarization
    is assumed.

    Parameters
    ----------
    z0 : ndarray, shape (i,)
        Ratios of half cylinder heights to cylinder radii.
    r : ndarray, shape (i,)
        Ratios of radial observer positions to cylinder radii.
    z : ndarray, shape (i,)
        Ratios of axial observer positions to cylinder radii.

    Returns
    -------
    ndarray, shape (3, i)
        B-field components (Br, Bphi, Bz) in units (T) at the
        observer positions, for unit polarization. Values scale linearly with
        the polarization magnitude.

    Notes
    -----
    Implementation based on: Derby, American Journal of Physics 78(3) (2010),
    229-235.

    Examples
    --------
    >>> import numpy as np
    >>> import magpylib as magpy
    >>> B = magpy.core.magnet_cylinder_axial_Bfield(
    ...    z0=np.array([1.0, 2.0]),
    ...    r=np.array([1.0, 2.0]),
    ...    z=np.array([2.0, 3.0]),
    ... )
    >>> with np.printoptions(precision=3):
    ...    print(B)
    [[0.056 0.041]
     [0.    0.   ]
     [0.067 0.018]]
    """
    n = len(z0)

    # some important quantities
    zph, zmh = z + z0, z - z0
    dpr, dmr = 1 + r, 1 - r

    sq0 = np.sqrt(zmh**2 + dpr**2)
    sq1 = np.sqrt(zph**2 + dpr**2)

    k1 = np.sqrt((zph**2 + dmr**2) / (zph**2 + dpr**2))
    k0 = np.sqrt((zmh**2 + dmr**2) / (zmh**2 + dpr**2))
    gamma = dmr / dpr
    one = np.ones(n)

    # radial field (unit polarization)
    Br = (_cel(k1, one, one, -one) / sq1 - _cel(k0, one, one, -one) / sq0) / np.pi

    # axial field (unit polarization)
    Bz = (
        1
        / dpr
        * (
            zph * _cel(k1, gamma**2, one, gamma) / sq1
            - zmh * _cel(k0, gamma**2, one, gamma) / sq0
        )
        / np.pi
    )

    return np.vstack((Br, np.zeros(n), Bz))


def magnet_cylinder_diametral_Hfield(
    z0: np.ndarray,
    r: np.ndarray,
    z: np.ndarray,
    phi: np.ndarray,
) -> list:
    """H-field of i diametrally magnetized cylinders in cylindrical coordinates.

    The cylinder axes coincide with the z-axis of the Cylindrical CS and the
    geometric center of the cylinder lies in the origin. Length inputs are
    made dimensionless by division over the cylinder radii. Unit magnetization
    is assumed.

    Parameters
    ----------
    z0 : ndarray, shape (i,)
        Ratios of cylinder heights to cylinder radii.
    r : ndarray, shape (i,)
        Ratios of radial observer positions to cylinder radii.
    z : ndarray, shape (i,)
        Ratios of axial observer positions to cylinder radii.
    phi : ndarray, shape (i,)
        Azimuth angles between observers and magnetization directions (radians).

    Returns
    -------
    ndarray, shape (3, i)
        H-field components (Hr, Hphi, Hz) in (A/m) at the observer
        positions, for unit magnetization. Values scale linearly with the
        magnetization magnitude.

    Notes
    -----
    Implementation partially based on Caciagli, Journal of Magnetism and
    Magnetic Materials 456 (2018): 423-432, and [Ortner, Leitner, Rauber]
    (unpublished).

    Examples
    --------
    >>> import numpy as np
    >>> import magpylib as magpy
    >>> H = magpy.core.magnet_cylinder_diametral_Hfield(
    ...    z0=np.array([1.0, 2.0]),
    ...    r=np.array([1.0, 2.0]),
    ...    z=np.array([2.0, 3.0]),
    ...    phi=np.array([0.1, np.pi/4]),
    ... )
    >>> with np.printoptions(precision=3):
    ...     print(H)
    [[-0.021  0.007]
     [ 0.005  0.02 ]
     [ 0.055  0.029]]
    """
    # pylint: disable=too-many-statements

    n = len(z0)

    # allocate to treat small r special cases
    Hr, Hphi, Hz = np.empty((3, n))

    # compute repeated quantities for all cases
    zp = z + z0
    zm = z - z0

    zp2 = zp**2
    zm2 = zm**2
    r2 = r**2

    # case small_r: numerical instability of general solution
    mask_small_r = r < 0.05
    mask_general = ~mask_small_r
    if np.any(mask_small_r):
        phiX = phi[mask_small_r]
        zpX, zmX = zp[mask_small_r], zm[mask_small_r]
        zp2X, zm2X = zp2[mask_small_r], zm2[mask_small_r]
        rX, r2X = r[mask_small_r], r2[mask_small_r]

        # taylor series for small r
        zpp = zp2X + 1
        zmm = zm2X + 1
        sqrt_p = np.sqrt(zpp)
        sqrt_m = np.sqrt(zmm)

        frac1 = zpX / sqrt_p
        frac2 = zmX / sqrt_m

        r3X = r2X * rX
        r4X = r3X * rX
        r5X = r4X * rX

        term1 = frac1 - frac2
        term2 = (frac1 / zpp**2 - frac2 / zmm**2) * r2X / 8
        term3 = (
            ((3 - 4 * zp2X) * frac1 / zpp**4 - (3 - 4 * zm2X) * frac2 / zmm**4)
            / 64
            * r4X
        )

        Hr[mask_small_r] = -np.cos(phiX) / 4 * (term1 + 9 * term2 + 25 * term3)

        Hphi[mask_small_r] = np.sin(phiX) / 4 * (term1 + 3 * term2 + 5 * term3)

        Hz[mask_small_r] = (
            -np.cos(phiX)
            / 4
            * (
                rX * (1 / zpp / sqrt_p - 1 / zmm / sqrt_m)
                + 3
                / 8
                * r3X
                * ((1 - 4 * zp2X) / zpp**3 / sqrt_p - (1 - 4 * zm2X) / zmm**3 / sqrt_m)
                + 15
                / 64
                * r5X
                * (
                    (1 - 12 * zp2X + 8 * zp2X**2) / zpp**5 / sqrt_p
                    - (1 - 12 * zm2X + 8 * zm2X**2) / zmm**5 / sqrt_m
                )
            )
        )

        # if there are small_r, select the general/case variables
        # when there are no small_r cases it is not necessary to slice with [True, True, Tue, ...]
        phi = phi[mask_general]
        n = len(phi)
        zp, zm = zp[mask_general], zm[mask_general]
        zp2, zm2 = zp2[mask_general], zm2[mask_general]
        r, r2 = r[mask_general], r2[mask_general]

    if np.any(mask_general):
        rp = r + 1
        rm = r - 1
        rp2 = rp**2
        rm2 = rm**2

        ap2 = zp2 + rm**2
        am2 = zm2 + rm**2
        ap = np.sqrt(ap2)
        am = np.sqrt(am2)

        argp = -4 * r / ap2
        argm = -4 * r / am2

        # special case r=r0 : indefinite form
        #   result is numerically stable in the vicinity of of r=r0
        #   so only the special case must be caught (not the surroundings)
        mask_special = rm == 0
        argc = np.ones(n) * 1e16  # should be np.Inf but leads to 1/0 problems in cel
        argc[~mask_special] = -4 * r[~mask_special] / rm2[~mask_special]
        # special case 1/rm
        one_over_rm = np.zeros(n)
        one_over_rm[~mask_special] = 1 / rm[~mask_special]

        elle_p = ellipe(argp)
        elle_m = ellipe(argm)
        ellk_p = ellipk(argp)
        ellk_m = ellipk(argm)
        onez = np.ones(n)
        ellpi_p = _cel(np.sqrt(1 - argp), 1 - argc, onez, onez)  # elliptic_Pi
        ellpi_m = _cel(np.sqrt(1 - argm), 1 - argc, onez, onez)  # elliptic_Pi

        # compute fields
        Hr[mask_general] = (
            -np.cos(phi)
            / (4 * np.pi * r2)
            * (
                -zm * am * elle_m
                + zp * ap * elle_p
                + zm / am * (2 + zm2) * ellk_m
                - zp / ap * (2 + zp2) * ellk_p
                + (zm / am * ellpi_m - zp / ap * ellpi_p) * rp * (r2 + 1) * one_over_rm
            )
        )

        Hphi[mask_general] = (
            np.sin(phi)
            / (4 * np.pi * r2)
            * (
                +zm * am * elle_m
                - zp * ap * elle_p
                - zm / am * (2 + zm2 + 2 * r2) * ellk_m
                + zp / ap * (2 + zp2 + 2 * r2) * ellk_p
                + zm / am * rp2 * ellpi_m
                - zp / ap * rp2 * ellpi_p
            )
        )

        Hz[mask_general] = (
            -np.cos(phi)
            / (2 * np.pi * r)
            * (
                +am * elle_m
                - ap * elle_p
                - (1 + zm2 + r2) / am * ellk_m
                + (1 + zp2 + r2) / ap * ellk_p
            )
        )

    return np.vstack((Hr, Hphi, Hz))


def _BHJM_magnet_cylinder(
    field: str,
    observers: np.ndarray,
    dimension: np.ndarray,
    polarization: np.ndarray,
) -> np.ndarray:
    """
    - Translate cylinder core fields to BHJM
    - special cases
    """

    check_field_input(field)

    # transform to Cy CS --------------------------------------------
    r, phi, z = cart_to_cyl_coordinates(observers)
    r0, z0 = dimension.T / 2

    # scale invariance (make dimensionless)
    r = r / r0
    z = z / r0
    z0 = z0 / r0

    # allocate for output
    BHJM = polarization.astype(float)

    # inside/outside
    mask_between_bases = np.abs(z) <= z0  # in-between top and bottom plane
    mask_inside_hull = r <= 1  # inside Cylinder hull plane
    mask_inside = mask_between_bases & mask_inside_hull

    if field == "J":
        BHJM[~mask_inside] = 0
        return BHJM

    if field == "M":
        BHJM[~mask_inside] = 0
        return BHJM / MU0

    # SPECIAL CASE 1: on Cylinder edge
    mask_on_hull = np.isclose(r, 1, rtol=1e-15, atol=0)  # on Cylinder hull plane
    mask_on_bases = np.isclose(abs(z), z0, rtol=1e-15, atol=0)  # on top or bottom plane
    mask_not_on_edge = ~(mask_on_hull & mask_on_bases)

    # axial/transv polarization cases
    pol_x, pol_y, pol_z = polarization.T
    mask_pol_tv = (pol_x != 0) | (pol_y != 0)
    mask_pol_ax = pol_z != 0

    # SPECIAL CASE 2: pol = 0
    mask_pol_not_null = ~((pol_x == 0) * (pol_y == 0) * (pol_z == 0))

    # general case
    mask_gen = mask_pol_not_null & mask_not_on_edge

    # general case masks
    mask_pol_tv = mask_pol_tv & mask_gen
    mask_pol_ax = mask_pol_ax & mask_gen
    mask_inside = mask_inside & mask_gen

    BHJM *= 0

    # transversal polarization contributions -----------------------
    if any(mask_pol_tv):
        pol_xy = np.sqrt(pol_x**2 + pol_y**2)[mask_pol_tv]
        tetta = np.arctan2(pol_y[mask_pol_tv], pol_x[mask_pol_tv])

        BHJM[mask_pol_tv] = (
            magnet_cylinder_diametral_Hfield(
                z0=z0[mask_pol_tv],
                r=r[mask_pol_tv],
                z=z[mask_pol_tv],
                phi=phi[mask_pol_tv] - tetta,
            )
            * pol_xy
        ).T

    # axial polarization contributions ----------------------------
    if any(mask_pol_ax):
        BHJM[mask_pol_ax] += (
            magnet_cylinder_axial_Bfield(
                z0=z0[mask_pol_ax],
                r=r[mask_pol_ax],
                z=z[mask_pol_ax],
            )
            * pol_z[mask_pol_ax]
        ).T

    BHJM[:, 0], BHJM[:, 1] = cyl_field_to_cart(phi, BHJM[:, 0], BHJM[:, 1])

    # add/subtract Mag when inside for B/H
    if field == "B":
        mask_tv_inside = mask_pol_tv * mask_inside
        if any(mask_tv_inside):  # tv computes H-field
            BHJM[mask_tv_inside, 0] += pol_x[mask_tv_inside]
            BHJM[mask_tv_inside, 1] += pol_y[mask_tv_inside]
        return BHJM

    if field == "H":
        mask_ax_inside = mask_pol_ax * mask_inside
        if any(mask_ax_inside):  # ax computes B-field
            BHJM[mask_ax_inside, 2] -= pol_z[mask_ax_inside]
        return BHJM / MU0

    msg = (
        "Input output_field_type must be one of {'B', 'H', 'M', 'J'}; "
        f"instead received {field!r}."
    )
    raise ValueError(msg)  # pragma: no cover

"""Homogeneously magnetized cylinder field implementation."""

import numpy as np
from array_api_compat import array_namespace
from scipy.constants import mu_0 as MU0

from magpylib._src.array_api_utils import xp_promote
from magpylib._src.fields.special_cel import _cel
from magpylib._src.input_checks import check_field_input
from magpylib._src.utility import cart_to_cyl_coordinates, radial_field_to_cart


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
    xp = array_namespace(z0, r, z)
    z0, r, z = xp_promote(z0, r, z, xp=xp)

    # some important quantities
    zph, zmh = z + z0, z - z0
    dpr, dmr = 1 + r, 1 - r

    sq0 = xp.sqrt(zmh**2 + dpr**2)
    sq1 = xp.sqrt(zph**2 + dpr**2)

    k1 = xp.sqrt((zph**2 + dmr**2) / (zph**2 + dpr**2))
    k0 = xp.sqrt((zmh**2 + dmr**2) / (zmh**2 + dpr**2))
    gamma = dmr / dpr
    one = xp.ones_like(z0)

    # radial field (unit polarization)
    Br = (_cel(k1, one, one, -one) / sq1 - _cel(k0, one, one, -one) / sq0) / xp.pi

    # axial field (unit polarization)
    Bz = (
        1
        / dpr
        * (
            zph * _cel(k1, gamma**2, one, gamma) / sq1
            - zmh * _cel(k0, gamma**2, one, gamma) / sq0
        )
        / xp.pi
    )

    return xp.stack((Br, xp.zeros_like(Br), Bz))


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
    xp = array_namespace(z0, r, z, phi)
    z0, r, z, phi = xp_promote(z0, r, z, phi, xp=xp)
    cos, sin = xp.cos(phi), xp.sin(phi)
    Hu, Hv, Hz = xp.unstack(_diametral_Hfield_cart(z0, r, r * cos, r * sin, z))
    return xp.stack((Hu * cos + Hv * sin, Hv * cos - Hu * sin, Hz))


def _diametral_Hfield_cart(z0, r, u, v, z):
    """H-field of ``magnet_cylinder_diametral_Hfield`` in Cartesian components
    (Hu, Hv, Hz), with the magnetization along u.

    ``u`` and ``v`` are the observer coordinates across the axis and ``r``
    their distance from it. Close to the axis the field is a polynomial of u
    and v, which is smooth on the axis, where the cylindrical components are
    not.
    """
    # pylint: disable=too-many-statements

    xp = array_namespace(z0, r, u, v, z)

    # compute repeated quantities for all cases
    zp = z + z0
    zm = z - z0

    zp2 = zp**2
    zm2 = zm**2

    # case small_r: numerical instability of general solution
    mask_small_r = r < 0.05

    # both cases are computed for all instances: with r=0 in the small_r case
    # and r=2 in the general case where they do not apply
    uX = xp.where(mask_small_r, u, 0.0)
    vX = xp.where(mask_small_r, v, 0.0)
    u2X, v2X = uX**2, vX**2
    r2X = u2X + v2X

    # taylor series for small r
    zpp = zp2 + 1
    zmm = zm2 + 1
    sqrt_p = xp.sqrt(zpp)
    sqrt_m = xp.sqrt(zmm)

    frac1 = zp / sqrt_p
    frac2 = zm / sqrt_m

    term1 = frac1 - frac2
    term2 = (frac1 / zpp**2 - frac2 / zmm**2) / 8
    term3 = ((3 - 4 * zp2) * frac1 / zpp**4 - (3 - 4 * zm2) * frac2 / zmm**4) / 64

    # Hr * cos(phi) - Hphi * sin(phi) and Hr * sin(phi) + Hphi * cos(phi) of
    # Hr = -cos(phi) / 4 * (term1 + 9 * term2 * r**2 + 25 * term3 * r**4)
    # Hphi = sin(phi) / 4 * (term1 + 3 * term2 * r**2 + 5 * term3 * r**4)
    Hu_small = (
        -(term1 + term2 * (9 * u2X + 3 * v2X) + term3 * r2X * (25 * u2X + 5 * v2X)) / 4
    )
    Hv_small = -uX * vX * (6 * term2 + 20 * term3 * r2X) / 4

    Hz_small = (
        -uX
        / 4
        * (
            (1 / zpp / sqrt_p - 1 / zmm / sqrt_m)
            + 3
            / 8
            * r2X
            * ((1 - 4 * zp2) / zpp**3 / sqrt_p - (1 - 4 * zm2) / zmm**3 / sqrt_m)
            + 15
            / 64
            * r2X**2
            * (
                (1 - 12 * zp2 + 8 * zp2**2) / zpp**5 / sqrt_p
                - (1 - 12 * zm2 + 8 * zm2**2) / zmm**5 / sqrt_m
            )
        )
    )

    # general case
    r = xp.where(mask_small_r, 2.0, r)
    r2 = r**2
    cos, sin = u / r, v / r

    rp = r + 1
    rm = r - 1
    rp2 = rp**2

    ap2 = zp2 + rm**2
    am2 = zm2 + rm**2
    ap = xp.sqrt(ap2)
    am = xp.sqrt(am2)

    argp = -4 * r / ap2
    argm = -4 * r / am2

    # special case r=r0 : indefinite form
    #   result is numerically stable in the vicinity of of r=r0
    #   so only the special case must be caught (not the surroundings)
    mask_special = rm == 0
    rm = xp.where(mask_special, 1.0, rm)  # avoid 1/0
    # should be np.inf but leads to 1/0 problems in cel
    argc = xp.where(mask_special, 1e16, -4 * r / rm**2)
    # special case 1/rm
    one_over_rm = xp.where(mask_special, 0.0, 1 / rm)

    # complete elliptic integrals E, K and Pi through cel
    onez = xp.ones_like(r)
    kcp = xp.sqrt(1 - argp)
    kcm = xp.sqrt(1 - argm)
    elle_p = _cel(kcp, onez, onez, 1 - argp)
    elle_m = _cel(kcm, onez, onez, 1 - argm)
    ellk_p = _cel(kcp, onez, onez, onez)
    ellk_m = _cel(kcm, onez, onez, onez)
    ellpi_p = _cel(kcp, 1 - argc, onez, onez)  # elliptic_Pi
    ellpi_m = _cel(kcm, 1 - argc, onez, onez)  # elliptic_Pi

    # compute fields
    Hr = (
        -cos
        / (4 * xp.pi * r2)
        * (
            -zm * am * elle_m
            + zp * ap * elle_p
            + zm / am * (2 + zm2) * ellk_m
            - zp / ap * (2 + zp2) * ellk_p
            + (zm / am * ellpi_m - zp / ap * ellpi_p) * rp * (r2 + 1) * one_over_rm
        )
    )

    Hphi = (
        sin
        / (4 * xp.pi * r2)
        * (
            +zm * am * elle_m
            - zp * ap * elle_p
            - zm / am * (2 + zm2 + 2 * r2) * ellk_m
            + zp / ap * (2 + zp2 + 2 * r2) * ellk_p
            + zm / am * rp2 * ellpi_m
            - zp / ap * rp2 * ellpi_p
        )
    )

    Hz = (
        -cos
        / (2 * xp.pi * r)
        * (
            +am * elle_m
            - ap * elle_p
            - (1 + zm2 + r2) / am * ellk_m
            + (1 + zp2 + r2) / ap * ellk_p
        )
    )

    return xp.stack(
        (
            xp.where(mask_small_r, Hu_small, Hr * cos - Hphi * sin),
            xp.where(mask_small_r, Hv_small, Hr * sin + Hphi * cos),
            xp.where(mask_small_r, Hz_small, Hz),
        )
    )


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

    xp = array_namespace(observers, dimension, polarization)
    observers, dimension, polarization = xp_promote(
        observers, dimension, polarization, xp=xp
    )

    # transform to Cy CS --------------------------------------------
    x, y, z, r = cart_to_cyl_coordinates(observers)
    r0, z0 = xp.unstack(dimension / 2, axis=-1)

    # scale invariance (make dimensionless)
    x = x / r0
    y = y / r0
    r = r / r0
    z = z / r0
    z0 = z0 / r0

    # inside/outside
    mask_between_bases = xp.abs(z) <= z0  # in-between top and bottom plane
    mask_inside_hull = r <= 1  # inside Cylinder hull plane
    mask_inside = mask_between_bases & mask_inside_hull

    if field == "J":
        return xp.where(mask_inside[:, xp.newaxis], polarization, 0.0)

    if field == "M":
        return xp.where(mask_inside[:, xp.newaxis], polarization, 0.0) / MU0

    # SPECIAL CASE 1: on Cylinder edge
    mask_on_hull = xp.abs(r - 1) <= 1e-15  # on Cylinder hull plane
    # on top or bottom plane
    mask_on_bases = xp.abs(xp.abs(z) - z0) <= 1e-15 * xp.abs(z0)
    mask_not_on_edge = ~(mask_on_hull & mask_on_bases)

    # axial/transv polarization cases
    pol_x, pol_y, pol_z = xp.unstack(polarization, axis=-1)
    mask_pol_tv = (pol_x != 0) | (pol_y != 0)
    mask_pol_ax = pol_z != 0

    # SPECIAL CASE 2: pol = 0
    mask_pol_not_null = mask_pol_tv | mask_pol_ax

    # general case
    mask_gen = mask_pol_not_null & mask_not_on_edge

    # general case masks
    mask_pol_tv = mask_pol_tv & mask_gen
    mask_pol_ax = mask_pol_ax & mask_gen
    mask_inside = mask_inside & mask_gen

    # the core functions are computed for all instances: in the center of a
    # cylinder with z0=1 where they do not apply, to stay clear of the
    # singularities

    # transversal polarization contributions -----------------------
    # in the coordinates u along the transversal polarization and v across
    pol_xy = xp.sqrt(pol_x**2 + pol_y**2)
    tetta = xp.atan2(
        xp.where(mask_pol_tv, pol_y, 0.0), xp.where(mask_pol_tv, pol_x, 1.0)
    )
    cos, sin = xp.cos(tetta), xp.sin(tetta)
    Hu, Hv, Hz_tv = xp.unstack(
        xp.where(
            mask_pol_tv,
            _diametral_Hfield_cart(
                z0=xp.where(mask_pol_tv, z0, 1.0),
                r=xp.where(mask_pol_tv, r, 0.0),
                u=xp.where(mask_pol_tv, x * cos + y * sin, 0.0),
                v=xp.where(mask_pol_tv, y * cos - x * sin, 0.0),
                z=xp.where(mask_pol_tv, z, 0.0),
            )
            * pol_xy,
            0.0,
        )
    )

    # axial polarization contributions ----------------------------
    z0_ax = xp.where(mask_pol_ax, z0, 1.0)
    z_ax = xp.where(mask_pol_ax, z, 0.0)
    Br, _, Bz_ax = xp.unstack(
        xp.where(
            mask_pol_ax,
            magnet_cylinder_axial_Bfield(
                z0=z0_ax, r=xp.where(mask_pol_ax, r, 0.0), z=z_ax
            )
            * pol_z,
            0.0,
        )
    )
    # on the axis Br / r is -dBz/dz / 2, as the field has no divergence. Br
    # is a difference with an absolute error of eps, and Br / r is off from
    # its limit by r**2: next to the axis the limit is the better value
    slope = ((z_ax - z0_ax) ** 2 + 1) ** -1.5 - ((z_ax + z0_ax) ** 2 + 1) ** -1.5
    Bx_ax, By_ax = radial_field_to_cart(
        x,
        y,
        r,
        Br,
        xp.where(mask_pol_ax, slope * pol_z / 4, 0.0),
        r_min=xp.finfo(r.dtype).eps ** (1 / 3),
    )

    Bx = Hu * cos - Hv * sin + Bx_ax
    By = Hu * sin + Hv * cos + By_ax
    Bz = Hz_tv + Bz_ax

    # add/subtract Mag when inside for B/H
    if field == "B":
        mask_tv_inside = mask_pol_tv & mask_inside  # tv computes H-field
        Bx = xp.where(mask_tv_inside, Bx + pol_x, Bx)
        By = xp.where(mask_tv_inside, By + pol_y, By)
        return xp.stack((Bx, By, Bz), axis=-1)

    if field == "H":
        mask_ax_inside = mask_pol_ax & mask_inside  # ax computes B-field
        Bz = xp.where(mask_ax_inside, Bz - pol_z, Bz)
        return xp.stack((Bx, By, Bz), axis=-1) / MU0

    msg = (
        "Input output_field_type must be one of {'B', 'H', 'M', 'J'}; "
        f"instead received {field!r}."
    )
    raise ValueError(msg)  # pragma: no cover

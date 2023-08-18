"""
Implementations of analytical expressions for the magnetic field of
homogeneously magnetized Cylinders. Computation details in function docstrings.
"""
# pylint: disable = no-name-in-module
import numpy as np
from scipy.special import ellipe
from scipy.special import ellipk

from magpylib._src.fields.special_cel import cel
from magpylib._src.input_checks import check_field_input
from magpylib._src.utility import cart_to_cyl_coordinates
from magpylib._src.utility import cyl_field_to_cart


def fieldB_cylinder_axial(z0: np.ndarray, r: np.ndarray, z: np.ndarray) -> list:
    """
    B-field in Cylindrical CS of Cylinder magnet with homogenous axial unit
    magnetization. The Cylinder axis coincides with the z-axis of the
    CS. The geometric center of the Cylinder is in the origin.

    Implementation from [Derby2009].

    Parameters
    ----------
    dim: ndarray, shape (n,2)
        dimension of cylinder (d, h), diameter and height, in units of mm
    pos_obs: ndarray, shape (n,2)
        position of observer (r,z) in cylindrical coordinates in units of mm

    Returns
    -------
    B-field: ndarray
        B-field array of shape (n,2) in cylindrical coordinates (Br,Bz) in units of mT.
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

    # radial field (unit magnetization)
    Br = (cel(k1, one, one, -one) / sq1 - cel(k0, one, one, -one) / sq0) / np.pi

    # axial field (unit magnetization)
    Bz = (
        1
        / dpr
        * (
            zph * cel(k1, gamma**2, one, gamma) / sq1
            - zmh * cel(k0, gamma**2, one, gamma) / sq0
        )
        / np.pi
    )

    return Br, Bz


def fieldH_cylinder_diametral(
    z0: np.ndarray,
    r: np.ndarray,
    phi: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """
    H-field in Cylindrical CS of Cylinder magnet with homogenous
    diametral unit magnetization. The Cylinder axis coincides with the z-axis of the
    CS. The geometric center of the Cylinder is in the origin.

    Implementation from [Rauber2021].

    H-Field computed analytically via the magnetic scalar potential. Final integration
    reduced to complete elliptic integrals.

    Numerical Instabilities: See discussion on GitHub.

    Parameters
    ----------
    dim: ndarray, shape (n,2)
        dimension of cylinder (d, h), diameter and height, in units of mm
    tetta: ndarray, shape (n,)
        angle between magnetization vector and x-axis in [rad]. M = (cos(tetta), sin(tetta), 0)
    obs_pos: ndarray, shape (n,3)
        position of observer (r,phi,z) in cylindrical coordinates in units of mm and rad

    Returns
    -------
    H-field: ndarray
        H-field array of shape (n,3) in cylindrical coordinates (Hr, Hphi, Hz) in units of kA/m.
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
                * (
                    (1 - 4 * zp2X) / zpp**3 / sqrt_p
                    - (1 - 4 * zm2X) / zmm**3 / sqrt_m
                )
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
        # when there are no small_r cases it is not necessary to slice with [True, True, Tue,...]
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
        ellpi_p = cel(np.sqrt(1 - argp), 1 - argc, onez, onez)  # elliptic_Pi
        ellpi_m = cel(np.sqrt(1 - argm), 1 - argc, onez, onez)  # elliptic_Pi

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

    return Hr, Hphi, Hz


# CORE
def magnet_cylinder_field(
    field: str,
    observers: np.ndarray,
    magnetization: np.ndarray,
    dimension: np.ndarray,
) -> np.ndarray:
    """Magnetic field of a homogeneously magnetized cylinder.

    The cylinder axis coincides with the z-axis and the geometric center of the
    cylinder lies in the origin.

    Parameters
    ----------
    field: str, default=`'B'`
        If `field='B'` return B-field in units of mT, if `field='H'` return H-field
        in units of kA/m.

    observers: ndarray, shape (n,3)
        Observer positions (x,y,z) in Cartesian coordinates in units of mm.

    magnetization: ndarray, shape (n,3)
        Homogeneous magnetization vector in units of mT.

    dimension: ndarray, shape (n,2)
        Cylinder dimension (d,h) with diameter d and height h in units of mm.

    observer: ndarray, shape (n,3)
        Observer positions (x,y,z) in Cartesian coordinates in units of mm.

    field: str, default=`'B'`
        If `field='B'` return B-field in units of mT, if `field='H'` return H-field
        in units of kA/m.

    Returns
    -------
    B-field or H-field: ndarray, shape (n,3)
        B/H-field of magnet in Cartesian coordinates (Bx, By, Bz) in units of mT/(kA/m).

    Examples
    --------
    Compute the B-field of two different cylinder magnets at position (1,2,3).

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> mag = np.array([(0,0,1000), (100,0,100)])
    >>> dim = np.array([(1,1), (2,3)])
    >>> obs = np.array([(1,2,3), (1,2,3)])
    >>> B = magpy.core.magnet_cylinder_field('B', obs, mag, dim)
    >>> print(B)
    [[ 0.77141782  1.54283565  1.10384481]
     [-0.15185713  2.90352915  2.23601722]]

    Notes
    -----

    Axial implementation based on

    Derby: American Journal of Physics 78.3 (2010): 229-235.

    Diametral implementation based on

    Caciagli: Journal of Magnetism and Magnetic Materials 456 (2018): 423-432.

    Leitner/Rauber/Orter: WIP
    """

    bh = check_field_input(field, "magnet_cylinder_field()")

    # transform to Cy CS --------------------------------------------
    r, phi, z = cart_to_cyl_coordinates(observers)
    r0, z0 = dimension.T / 2

    # scale invariance (make dimensionless)
    r = np.copy(r / r0)
    z = np.copy(z / r0)
    z0 = np.copy(z0 / r0)

    # allocate field vectors ----------------------------------------
    Br, Bphi, Bz = np.zeros((3, len(r)))

    # create masks to distinguish between cases ---------------------
    m0 = np.isclose(r, 1, rtol=1e-15, atol=0)  # on Cylinder hull plane
    m1 = np.isclose(abs(z), z0, rtol=1e-15, atol=0)  # on top or bottom plane
    m2 = np.abs(z) <= z0  # in-between top and bottom plane
    m3 = r <= 1  # inside Cylinder hull plane

    # special case: mag = 0
    mask0 = np.linalg.norm(magnetization, axis=1) == 0

    # special case: on Cylinder edge
    mask_edge = m0 & m1

    # general case
    mask_gen = ~mask0 & ~mask_edge

    # axial/transv magnetization cases
    magx, magy, magz = magnetization.T
    mask_tv = (magx != 0) | (magy != 0)
    mask_ax = magz != 0

    # inside/outside
    mask_inside = m2 & m3

    # general case masks
    mask_tv = mask_tv & mask_gen
    mask_ax = mask_ax & mask_gen
    mask_inside = mask_inside & mask_gen

    # transversal magnetization contributions -----------------------
    if any(mask_tv):
        magxy = np.sqrt(magx**2 + magy**2)[mask_tv]
        tetta = np.arctan2(magy[mask_tv], magx[mask_tv])
        br_tv, bphi_tv, bz_tv = fieldH_cylinder_diametral(
            z0[mask_tv], r[mask_tv], phi[mask_tv] - tetta, z[mask_tv]
        )

        # add to H-field (inside magxy is missing for B !!!)
        Br[mask_tv] += magxy * br_tv
        Bphi[mask_tv] += magxy * bphi_tv
        Bz[mask_tv] += magxy * bz_tv

    # axial magnetization contributions -----------------------------
    if any(mask_ax):
        br_ax, bz_ax = fieldB_cylinder_axial(z0[mask_ax], r[mask_ax], z[mask_ax])
        Br[mask_ax] += magz[mask_ax] * br_ax
        Bz[mask_ax] += magz[mask_ax] * bz_ax

    # transform field to cartesian CS -------------------------------
    Bx, By = cyl_field_to_cart(phi, Br, Bphi)

    # add/subtract Mag when inside for B/H --------------------------
    if bh:
        if any(mask_tv):  # tv computes H-field
            Bx[mask_tv * mask_inside] += magx[mask_tv * mask_inside]
            By[mask_tv * mask_inside] += magy[mask_tv * mask_inside]
        return np.concatenate(((Bx,), (By,), (Bz,)), axis=0).T

    if any(mask_ax):  # ax computes B-field
        Bz[mask_tv * mask_inside] -= magz[mask_tv * mask_inside]
    return np.concatenate(((Bx,), (By,), (Bz,)), axis=0).T * 10 / 4 / np.pi


# old iterative solution by Furlani

# def magnet_cyl_dia_H_Furlani1994(
#         tetta: np.ndarray,
#         dim: np.ndarray,
#         pos_obs: np.ndarray,
#         niter: int,
#         ) -> np.ndarray:
#     """
#     H-field in Cylindrical CS of Cylinder magnet with homogenous
#     diametral unit magnetization. The Cylinder axis coincides with the z-axis of the
#     CS. The geometric center of the Cylinder is in the origin.

#     Implementation from [Furlani1994].

#     Parameters
#     ----------
#     dim: ndarray, shape (n,2)
#         dimension of cylinder (d, h), diameter and height, in units of mm
#     tetta: ndarray, shape (n,)
#         angle between magnetization vector and x-axis in [rad]. M = (cos(tetta), sin(tetta), 0)
#     obs_pos: ndarray, shape (n,3)
#         position of observer (r,phi,z) in cylindrical coordinates in units of mm and rad
#     niter: int
#         Iterations for Simpsons approximation of the final integral

#     Returns
#     -------
#     H-field: ndarray
#         H-field array of shape (n,3) in cylindrical coordinates (Hr, Hphi Hz) in units of kA/m.

#     Examples
#     --------
#     Compute field at three instances.

#     >>> import numpy as np
#     >>> import magpylib as magpy
#     >>> tetta = np.zeros(3)
#     >>> dim = np.array([(2,2), (2,3), (3,4)])
#     >>> obs = np.array([(.1,0,2), (2,0.12,3), (4,0.2,1)])
#     >>> B = magpy.core.magnet_cyl_dia_H_Furlani1994(tetta, dim, obs, 1000)
#     >>> print(B)
#     [[-5.99240321e-02  1.41132875e-19  8.02440419e-03]
#      [ 1.93282782e-03  2.19048077e-03  2.60408201e-02]
#      [ 5.27008607e-02  6.06112282e-03  1.54692676e-02]]

#     Notes
#     -----
#     H-Field computed from the charge picture, Simpsons approximation used
#     to approximate the integral.
#     """

#     r0, z0 = dim.T/2
#     r, phi, z = pos_obs.T
#     n = len(r0)

#     # phi is now relative between mag and pos_obs
#     phi = phi-tetta

#     #implementation of Furlani1993
#     # generating the iterative summand basics for simpsons approximation
#     phi0 = 2*np.pi/niter       # discretization
#     sphi = np.arange(niter+1)
#     sphi[sphi%2==0] = 2.
#     sphi[sphi%2==1] = 4.
#     sphi[0] = 1.
#     sphi[-1] = 1.

#     sphiex = np.outer(sphi, np.ones(n))
#     phi0ex = np.outer(np.arange(niter+1), np.ones(n))*phi0
#     zex    = np.outer(np.ones(niter+1), z)
#     hex   = np.outer(np.ones(niter+1), z0)       # pylint: disable=redefined-builtin
#     phiex  = np.outer(np.ones(niter+1), phi)
#     dr2ex  = np.outer(np.ones(niter+1), 2*r0*r)
#     r2d2ex = np.outer(np.ones(niter+1), r**2+r0**2)

#     # repetitives
#     cos_phi0ex = np.cos(phi0ex)
#     cos_phi = np.cos(phiex-phi0ex)

#     # compute r-phi components
#     mask = (r2d2ex-dr2ex*cos_phi == 0) # special case r = d/2 and cos_phi=1
#     unite  = np.ones([niter+1,n])
#     unite[mask] = - (1/2)/(zex[mask]+hex[mask])**2 + (1/2)/(zex[mask]-hex[mask])**2

#     rrc = r2d2ex[~mask] - dr2ex[~mask]*cos_phi[~mask]
#     g_m = 1/np.sqrt(rrc + (zex[~mask] + hex[~mask])**2)
#     g_p = 1/np.sqrt(rrc + (zex[~mask] - hex[~mask])**2)
#     unite[~mask] = ((zex+hex)[~mask]*g_m - (zex-hex)[~mask]*g_p)/rrc

#     summand = sphiex/3*cos_phi0ex*unite

#     Br   = r0/2/niter*np.sum(summand*(r-r0*cos_phi), axis=0)
#     Bphi = r0**2/2/niter*np.sum(summand*np.sin(phiex-phi0ex), axis=0)

#     # compute z-component
#     gz_m = 1/np.sqrt(r**2 + r0**2 - 2*r0*r*cos_phi + (zex+z0)**2)
#     gz_p = 1/np.sqrt(r**2 + r0**2 - 2*r0*r*cos_phi + (zex-z0)**2)
#     summandz = sphiex/3*cos_phi0ex*(gz_p - gz_m)
#     Bz = r0/2/niter*np.sum(summandz, axis=0)

#     return np.array([Br, Bphi, Bz]).T

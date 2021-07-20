"""
Implementations of analytical expressions for the magnetic field of
homogeneously magnetized Cylinders. Computation details in function docstrings.
"""
# pylint: disable = no-name-in-module

import numpy as np
from scipy.special import ellipk, ellipe
from magpylib._lib.fields.special_cel import cel
from magpylib._lib.utility import close


def field_Bcy_axial(dim: np.ndarray, pos_obs: np.ndarray) -> list:
    """ Compute B-field of Cylinder magnet with homogenous unit axial
            magnetization in Cylindrical CS.

    ### Args:
    - dim  (ndarray Nx2): dimension of Cylinder (d x h) in units of mm
    - obs_pos (ndarray Nx3): position of observer (r,phi,z) in mm/rad/mm

    ### Returns:
    - list with [Br, Bz] in units of mT

    ### init_state:
    A Cylinder with diameter d  and height h. The Cylinder axis coincides
    with the z-axis of a Cartesian CS. The geometric center of the Cylinder
    is in the origin of the CS.

    ### Computation info:
    Field computed from the current picture (perfect solenoid)
    - Derby: "Cylindrical Magnets and Ideal Solenoids" (2009)

    ### Numerical instabilities:
        When approaching the edges numerical instabilities appear
        at 1e-15. Default wrapper returns 0 when approaching edges.
    """

    d,h = dim.T / 2       # d/h are now radius and h/2
    r,_,z = pos_obs.T
    n = len(d)

    # some important quantitites
    zph, zmh = z+h, z-h
    dpr, dmr = d+r, d-r

    sq0 = np.sqrt(zmh**2+dpr**2)
    sq1 = np.sqrt(zph**2+dpr**2)

    k1 = np.sqrt((zph**2+dmr**2)/(zph**2+dpr**2))
    k0 = np.sqrt((zmh**2+dmr**2)/(zmh**2+dpr**2))
    gamma = dmr/dpr
    one = np.ones(n)

    # radial field (unit magnetization)
    Br = d*(cel(k1, one, one, -one)/sq1 - cel(k0, one, one, -one)/sq0)/np.pi

    # axial field (unit magnetization)
    Bz = d/dpr*(zph*cel(k1, gamma**2, one, gamma)/sq1
              - zmh*cel(k0, gamma**2, one, gamma)/sq0)/np.pi

    return [Br, Bz]  # contribution from axial magnetization


def field_Hcy_transv(
        tetta: np.ndarray,
        dim: np.ndarray,
        pos_obs: np.ndarray,
        ) -> list:
    """ Compute H-field of Cylinder magnet with homogenous unit diametral
            magnetization in Cylindrical CS.

    ### Args:
    - tetta (ndarray N): angle between xymag and x-axis
    - dim (ndarray Nx2): dimension of Cylinder (d x h) in units of mm
    - obs_pos (ndarray Nx3): position of observer (r,phi,z) in mm/rad/mm

    ### Returns:
    - list with [Hr, Hphi Hz] in units of [mT]

    ### init_state:
    A Cylinder with diameter d  and height h. The Cylinder axis coincides
    with the z-axis of a Cartesian CS. The geometric center of the Cylinder
    is in the origin of the CS.

    ### Computation info (Rauber):
    H-Field computed analytically via the magnetic scalar potential. Integration
    reduced to complete elliptic integrals.

    ### Computation info (Furlani, old):
    H-Field computed from the charge picture, Simpsons approximation used
        to approximate the intergral
    - Furlani: "A three dimensional field solution for bipolar cylinders" (1994)

    ### Numerical instabilities
        When approaching the edges numerical instabilities appear
        at 1e-15. Default wrapper returns 0 when approaching edges.
        SOLUTION NEEDS TESTING
    """

    r0, z0 = dim.T/2
    r, phi, z = pos_obs.T
    n = len(r0)

    # phi is now relative between mag and pos_obs
    phi = phi-tetta

    # implementation of Rauber2021
    # compute repeated quantities
    zp = z+z0
    zm = z-z0
    rp = r+r0
    rm = r-r0

    zp2 = zp**2
    zm2 = zm**2
    rp2 = rp**2
    rm2 = rm**2
    r2 = r**2
    r02 = r0**2
    rr0 = r*r0

    ap2 = zp2+rm**2
    am2 = zm2+rm**2
    ap = np.sqrt(ap2)
    am = np.sqrt(am2)

    argp = -4*rr0/ap2
    argm = -4*rr0/am2

    # special case r=r0 : indefinite form
    #   result is numerically stable in the vicinity of of r=r0
    #   so only the special case must be caught (not the surroundings)
    mask_special = rm==0
    argc = np.zeros(n)
    argc[~mask_special] = -4*rr0[~mask_special]/rm2[~mask_special]

    elle_p = ellipe(argp)
    elle_m = ellipe(argm)
    ellk_p = ellipk(argp)
    ellk_m = ellipk(argm)
    onez = np.ones(n)
    ellpi_p = cel(np.sqrt(1-argp), 1-argc, onez, onez) # elliptic_Pi
    ellpi_m = cel(np.sqrt(1-argm), 1-argc, onez, onez) # elliptic_Pi

    # compute fields
    Hphi = np.sin(phi)/(4*np.pi*r2)*(
        + zm*am                  * elle_m   -  zp*ap                  * elle_p
        - zm/am*(2*r02+zm2+2*r2) * ellk_m   +  zp/ap*(2*r02+zp2+2*r2) * ellk_p
        + zm/am*rp2              * ellpi_m  -  zp/ap*rp2              * ellpi_p
        )

    Hz = - np.cos(phi)/(2*np.pi*r)*(
        + am              * elle_m  -  ap              * elle_p
        - (r02+zm2+r2)/am * ellk_m  +  (r02+zp2+r2)/ap * ellk_p
        )

    # special case 1/rm
    one_over_rm = np.zeros(n)
    one_over_rm[~mask_special] = 1/rm[~mask_special]

    Hr = - np.cos(phi)/(4*np.pi*r2)*(
        - zm*am             * elle_m   +  zp*ap             * elle_p
        + zm/am*(2*r02+zm2) * ellk_m   -  zp/ap*(2*r02+zp2) * ellk_p
        + (zm/am* ellpi_m  -  zp/ap * ellpi_p) * rp*(r2+r02) * one_over_rm)
    # prefactor        
    #Hr = - np.cos(phi)/(4*np.pi*r2)*Hr

    # implementation of Furlani1993
    # # generating the iterative summand basics for simpsons approximation
    # phi0 = 2*np.pi/niter       # discretization
    # sphi = np.arange(niter+1)
    # sphi[sphi%2==0] = 2.
    # sphi[sphi%2==1] = 4.
    # sphi[0] = 1.
    # sphi[-1] = 1.

    # sphiex = np.outer(sphi, np.ones(n))
    # phi0ex = np.outer(np.arange(niter+1), np.ones(n))*phi0
    # zex    = np.outer(np.ones(niter+1), z)
    # hex   = np.outer(np.ones(niter+1), h)       # pylint: disable=redefined-builtin
    # phiex  = np.outer(np.ones(niter+1), phi)
    # dr2ex  = np.outer(np.ones(niter+1), 2*d*r)
    # r2d2ex = np.outer(np.ones(niter+1), r**2+d**2)

    # # repetitives
    # cos_phi0ex = np.cos(phi0ex)
    # cos_phi = np.cos(phiex-phi0ex)

    # # compute r-phi components
    # mask = (r2d2ex-dr2ex*cos_phi == 0) # special case r = d/2 and cos_phi=1
    # unite  = np.ones([niter+1,n])
    # unite[mask] = - (1/2)/(zex[mask]+hex[mask])**2 + (1/2)/(zex[mask]-hex[mask])**2

    # rrc = r2d2ex[~mask] - dr2ex[~mask]*cos_phi[~mask]
    # g_m = 1/np.sqrt(rrc + (zex[~mask] + hex[~mask])**2)
    # g_p = 1/np.sqrt(rrc + (zex[~mask] - hex[~mask])**2)
    # unite[~mask] = ((zex+hex)[~mask]*g_m - (zex-hex)[~mask]*g_p)/rrc

    # summand = sphiex/3*cos_phi0ex*unite

    # Br   = d/2/niter*np.sum(summand*(r-d*cos_phi), axis=0)
    # Bphi = d**2/2/niter*np.sum(summand*np.sin(phiex-phi0ex), axis=0)

    # # compute z-component
    # gz_m = 1/np.sqrt(r**2 + d**2 - 2*d*r*cos_phi + (zex+h)**2)
    # gz_p = 1/np.sqrt(r**2 + d**2 - 2*d*r*cos_phi + (zex-h)**2)
    # summandz = sphiex/3*cos_phi0ex*(gz_p - gz_m)
    # Bz = d/2/niter*np.sum(summandz, axis=0)

    return [Hr, Hphi, Hz]


def field_BH_cylinder(
        bh: bool,
        mag: np.ndarray,
        dim: np.ndarray,
        pos_obs: np.ndarray
        ) -> np.ndarray:
    """ setting up the Cylinder field computation
    - transform to Cylindrical CS
    - separate mag=0 cases (returning 0)
    - separate edge/corner cases (returning 0)
    - separate magz and magxy
    - call field computation for general cases
    - select B or H
    - transform B<-->H (inside check)

    ### Args:
    - bh (boolean): True=B, False=H
    - mag (ndarray Nx3): homogeneous magnetization vector in units of mT
    - dim (ndarray Nx2): dimension of Cylinder DxH in units of mm
    - pos_obs (ndarray Nx3): position of observer in units of mm

    ### Returns:
    - B/H-field (ndarray Nx3): magnetic field vectors at pos_obs in units of mT / kA/m
    """

    # transform to Cy CS --------------------------------------------
    x, y, z = pos_obs.T
    r, phi = np.sqrt(x**2+y**2), np.arctan2(y, x)
    pos_obs_cy = np.concatenate(((r,),(phi,),(z,)),axis=0).T

    # allocate field vectors ----------------------------------------
    Br, Bphi, Bz = np.zeros((3,len(x)))

    # create masks to distinguish between cases ---------------------

    r0,z0 = dim.T/2
    m0 = close(r, r0)        # on Cylinder plane
    m1 = close(abs(z), z0)   # on top or bottom plane
    m2 = np.abs(z)<=z0       # inside Cylinder plane
    m3 = r<=r0               # in-between top and bottom plane

    # special case: mag = 0
    mask0 = (np.linalg.norm(mag,axis=1)==0)

    # special case: on Cylinder surface
    mask_edge = (m0 & m2) | (m1 & m3)

    # general case
    mask_gen = ~mask0 & ~mask_edge

    # axial/transv magnetization cases
    magx, magy, magz = mag.T
    mask_tv = (magx != 0) | (magy != 0)
    mask_ax = (magz != 0)

    # inside/outside
    mask_inside = m2 & m3

    # general case masks
    mask_tv = mask_tv & mask_gen
    mask_ax = mask_ax & mask_gen
    mask_inside = mask_inside & mask_gen

    # transversal magnetization contributions -----------------------
    if any(mask_tv):
        # select non-zero tv parts
        magxy = np.sqrt(magx**2 + magy**2)[mask_tv]
        tetta = np.arctan2(magy[mask_tv], magx[mask_tv])
        pos_obs_tv = pos_obs_cy[mask_tv]
        dim_tv = dim[mask_tv]
        # compute H-field (in mT)
        br_tv, bphi_tv, bz_tv = field_Hcy_transv(tetta, dim_tv, pos_obs_tv)
        # add to H-field (inside magxy is missing for B)
        Br[mask_tv]   += magxy*br_tv
        Bphi[mask_tv] += magxy*bphi_tv
        Bz[mask_tv]   += magxy*bz_tv

    # axial magnetization contributions -----------------------------
    if any(mask_ax):
        # select non-zero ax parts
        pos_obs_ax = pos_obs_cy[mask_ax]
        magz_ax = magz[mask_ax]
        dim_ax = dim[mask_ax]
        # compute B-field
        br_ax, bz_ax = field_Bcy_axial(dim_ax, pos_obs_ax)
        # add to B-field
        Br[mask_ax] += magz_ax*br_ax
        Bz[mask_ax] += magz_ax*bz_ax

    # transform field to cartesian CS -------------------------------
    Bx = Br*np.cos(phi) - Bphi*np.sin(phi)
    By = Br*np.sin(phi) + Bphi*np.cos(phi)

    # add/subtract Mag when inside for B/H --------------------------
    if bh:
        if any(mask_tv): # tv computes H-field
            Bx[mask_tv*mask_inside] += magx[mask_tv*mask_inside]
            By[mask_tv*mask_inside] += magy[mask_tv*mask_inside]
        return np.concatenate(((Bx,),(By,),(Bz,)),axis=0).T

    if any(mask_ax): # ax computes B-field
        Bz[mask_tv*mask_inside] -= magz[mask_tv*mask_inside]
    return np.concatenate(((Bx,),(By,),(Bz,)),axis=0).T*10/4/np.pi

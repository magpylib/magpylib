"""
Implementations of analytical expressions for the magnetic field of homogeneously magnetized Cylinders.
Computation details in function docstrings.
"""

import numpy as np
from magpylib3._lib.math_utility.special_functions import celv
from magpylib3._lib.config import config


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
    br = d*(celv(k1, one, one, -one)/sq1 - celv(k0, one, one, -one)/sq0)/np.pi

    # axial field (unit magnetization)
    bz = d/dpr*(zph*celv(k1, gamma**2, one, gamma)/sq1 -
                    zmh*celv(k0, gamma**2, one, gamma)/sq0)/np.pi

    return [br, bz]  # contribution from axial magnetization


def field_Hcy_transv(tetta: np.ndarray, dim: np.ndarray, pos_obs: np.ndarray, niter: int) -> list:
    """ Compute H-field of Cylinder magnet with homogenous unit diametral 
            magnetization in Cylindrical CS.

    ### Args:
    - tetta (ndarray N): angle between xymag and x-axis
    - dim (ndarray Nx2): dimension of Cylinder (d x h) in units of mm
    - obs_pos (ndarray Nx3): position of observer (r,phi,z) in mm/rad/mm

    ### Returns:
    - list with [Hr, Hphi Hz] in units of !!! mT !!!

    ### init_state:
    A Cylinder with diameter d  and height h. The Cylinder axis coincides
    with the z-axis of a Cartesian CS. The geometric center of the Cylinder
    is in the origin of the CS.

    ### Computation info:
    Field computed from the charge picture, Simpsons approximation used
        to approximate the intergral
    - Furlani: "A three dimensional field solution for bipolar cylinders" (1994)

    ### Numerical instabilities
        When approaching the edges numerical instabilities appear
        at 1e-15. Default wrapper returns 0 when approaching edges.
        SOLUTION NEEDS TESTING
        
    """
    d,h = dim.T / 2       # d/h are now radius and (h/2)
    r,phi,z = pos_obs.T
    n = len(d)

    phi = phi-tetta        # phi is now relative between mag and pos_obs
    
    # generating the iterative summand basics for simpsons approximation
    phi0 = 2*np.pi/niter       # discretization
    sphi = np.arange(niter+1)
    sphi[sphi%2==0] = 2.
    sphi[sphi%2==1] = 4.
    sphi[0] = 1.
    sphi[-1] = 1.
    
    sphie = np.outer(sphi, np.ones(n))
    phi0e = np.outer(np.arange(niter+1), np.ones(n))*phi0
    ze    = np.outer(np.ones(niter+1), z)
    he    = np.outer(np.ones(niter+1), h)
    phie  = np.outer(np.ones(niter+1), phi)
    dr2e  = np.outer(np.ones(niter+1), 2*d*r)
    r2d2e = np.outer(np.ones(niter+1), r**2+d**2)
    
    # repetitives
    cos_phi0e = np.cos(phi0e)
    cos_phi = np.cos(phie-phi0e)
    
    # compute r-phi components
    ma = (r2d2e-dr2e*cos_phi == 0) # special case r = d/2 and cos_phi=1
    I1xE  = np.ones([niter+1,n])
    I1xE[ma] = - (1/2)/(ze[ma]+he[ma])**2 + (1/2)/(ze[ma]-he[ma])**2

    nma = np.logical_not(ma)
    rrc = r2d2e[nma] - dr2e[nma]*cos_phi[nma]
    Gm = 1/np.sqrt(rrc + (ze[nma] + he[nma])**2)
    Gp = 1/np.sqrt(rrc + (ze[nma] - he[nma])**2)
    I1xE[nma] = ((ze+he)[nma]*Gm - (ze-he)[nma]*Gp)/rrc

    Summand = sphie/3*cos_phi0e*I1xE

    br   = d/2/niter*np.sum(Summand*(r-d*cos_phi), axis=0)
    bphi = d**2/2/niter*np.sum(Summand*np.sin(phie-phi0e), axis=0)

    # compute z-component
    Gzm = 1/np.sqrt(r**2 + d**2 - 2*d*r*cos_phi + (ze+h)**2)
    Gzp = 1/np.sqrt(r**2 + d**2 - 2*d*r*cos_phi + (ze-h)**2)
    SummandZ = sphie/3*cos_phi0e*(Gzp-Gzm)
    bz = d/2/niter*np.sum(SummandZ, axis=0)

    return [br,bphi,bz]


def field_BH_cylinder(bh: bool, mag: np.ndarray, dim: np.ndarray, pos_obs: np.ndarray, niter: int) -> np.ndarray:
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
    
    EDGESIZE = config.EDGESIZE
    
    # transform to Cy CS --------------------------------------------
    x, y, z = pos_obs.T
    r, phi = np.sqrt(x**2+y**2), np.arctan2(y, x)
    pos_obs_cy = np.c_[r,phi,z]

    # allocate field vectors ----------------------------------------
    Br, Bphi, Bz = np.zeros((3,len(x)))

    # create masks to distinguish between cases ---------------------
    # special case mag = 0
    mask0 = (np.linalg.norm(mag,axis=1)==0)
    # special case on/off edge
    d,h = dim.T
    mask_edge = (abs(r-d/2) < EDGESIZE) & (abs(abs(z)-h/2) < EDGESIZE)
    # not special case
    mask_gen = ~mask0 & ~mask_edge

    # ax/transv
    magx, magy, magz = mag.T
    mask_tv = (magx != 0) | (magy != 0)
    mask_ax = (magz != 0)
    # inside/outside
    mask_inside = (r<dim[:,0]/2) * (abs(z)<dim[:,1]/2)
    
    # general cases
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
        # compute h-field (in mT)
        br_tv, bphi_tv, bz_tv = field_Hcy_transv(tetta, dim_tv, pos_obs_tv, niter)
        # add to B-field (inside magxy is missing for B)
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
        return np.c_[Bx,By,Bz]
    else:
        if any(mask_ax): # ax computes B-field
            Bz[mask_tv*mask_inside] -= magz[mask_tv*mask_inside]
        return np.c_[Bx,By,Bz]*10/4/np.pi
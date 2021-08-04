"""
Implementations of analytical expressions for the magnetic field of homogeneously
magnetized Cuboids. Computation details in function docstrings.
"""

import numpy as np
from magpylib._lib.config import Config


def field_BH_cuboid(
        bh: bool,
        mag: np.ndarray,
        dim: np.ndarray,
        pos_obs: np.ndarray
        ) -> np.ndarray:
    """ setting up the Cuboid field computation
    - separate mag=0 cases (returning 0)
    - separate edge/corner cases (returning 0)
    - call field computation for general cases
    - select B or H
    - transform B->H (inside check)

    ### Args:
    - bh (boolean): True=B, False=H
    - mag (ndarray Nx3): homogeneous magnetization vector in units of mT
    - dim (ndarray Nx3): dimension of Cuboid side lengths in units of mm
    - pos_obs (ndarray Nx3): position of observer in units of mm

    ### Returns:
    - B/H-field (ndarray Nx3): magnetic field vectors at pos_obs in units of mT / kA/m
    """

    edgesize = Config.EDGESIZE

    # allocate field vectors ----------------------
    B = np.zeros((len(mag),3))

    # special case mag = 0 ------------------------
    mask0 = (mag[:,0]==0) * (mag[:,1]==0) * (mag[:,2]==0)

    # special cases for edge/corner fields --------
    x, y, z = np.copy(pos_obs.T)
    a, b, c = dim.T/2

    mx1 = (abs(abs(x)-a) < edgesize)
    my1 = (abs(abs(y)-b) < edgesize)
    mz1 = (abs(abs(z)-c) < edgesize)

    mx2 = (abs(x)-a < edgesize) # within actual edge
    my2 = (abs(y)-b < edgesize)
    mz2 = (abs(z)-c < edgesize)

    mask_xedge = my1 & mz1 & mx2
    mask_yedge = mx1 & mz1 & my2
    mask_zedge = mx1 & my1 & mz2
    mask_edge = mask_xedge | mask_yedge | mask_zedge

    # not a special case --------------------------
    mask_gen = ~mask_edge & ~mask0

    # compute field -------------------------------
    if np.any(mask_gen):
        B[mask_gen] = magnet_cuboid_B_Yang1999(mag[mask_gen], dim[mask_gen], pos_obs[mask_gen])

    # return B or compute and retun H -------------
    if bh:
        return B

    # if inside magnet subtract magnetization vector
    poso_abs = np.abs(pos_obs)
    case1 = poso_abs[:,0]<=dim[:,0]/2-edgesize
    case2 = poso_abs[:,1]<=dim[:,1]/2-edgesize
    case3 = poso_abs[:,2]<=dim[:,2]/2-edgesize
    mask_inside = case1*case2*case3
    B[mask_inside] -= mag[mask_inside]
    # transform units mT -> kA/m
    H = B*10/4/np.pi
    return H


# ON INTERFACE
def magnet_cuboid_B_Yang1999(
    mag: np.ndarray,
    dim: np.ndarray,
    pos_obs: np.ndarray) -> np.ndarray:
    """
    B-field in Cartesian CS of Cuboid magnet with homogenous magnetization.
    The Cuboid sides are parallel to the CS axes.
    The geometric center of the Cuboid lies in the origin.

    Implementation from [Yang1999], [Engel-Herbert2005], [Camacho2013], [Cichon2019].

    Parameters
    ----------
    mag: ndarray, shape (n,3)
        Homogeneous magnetization vector in units of [mT]

    dim: ndarray, shape (n,3)
        Cuboid side lengths in units of [mm]

    pos_obs: ndarray, (n,3)
        position of observer in units of [mm]

    Returns
    -------
    B-field: ndarray, shape (n,3)
        B-field of Cuboid (Bx, By, Bz) in units of [mT].

    Info
    ----
    Field computations via magnetic surface charge density. See e.g.
    - Camacho: Revista Mexicana de F´ısica E 59 (2013) 8–17
    - Engel-Herbert: Journal of Applied Physics 97(7):074504 - 074504-4 (2005)
    - Yang: Superconductor Science and Technology 3(12):591 (1999)

    On magnet corners/edges (0,0,0) is returned

    Avoiding indeterminate forms:

    In the above implementations there are several indeterminate forms
    where the limit must be taken. These forms appear at positions
    that are extensions of the edges in all xyz-octants except bottQ4.
    In the vicinity of these indeterminat forms the formula becomes
    numerically instable.

    Chosen solution: use symmetries of the problem to change all
    positions to their bottQ4 counterparts. see also

    Cichon: IEEE SENSORS JOURNAL, VOL. 19, NO. 7, APRIL 1, 2019, p.2509

    On the magnet surface either inside or outside field is returned.

    Indeterminate forms at cuboid edges and corners remain. The numerical
    evaluation is instable in the vicinity. Avoid field computation near
    (1e-6*sidelength) Cuboid edges. FIX THIS PROBLEM !!!
    """
    # pylint: disable=too-many-statements

    magx, magy, magz = mag.T
    x, y, z = pos_obs.T
    a, b, c = dim.T/2
    n = len(magx)

    # avoid indeterminate forms by evaluating in bottQ4 only --------
    # basic masks
    maskx = x<0
    masky = y>0
    maskz = z>0

    # change all positions to their bottQ4 counterparts
    x[maskx] = x[maskx]*-1
    y[masky] = y[masky]*-1
    z[maskz] = z[maskz]*-1

    # create sign flips for position changes
    qsigns = np.ones((n,3,3))
    qs_flipx = np.array([[ 1,-1,-1],[-1, 1, 1],[-1, 1, 1]])
    qs_flipy = np.array([[ 1,-1, 1],[-1, 1,-1],[ 1,-1, 1]])
    qs_flipz = np.array([[ 1, 1,-1],[ 1, 1,-1],[-1,-1, 1]])
    # signs flips can be applied subsequently
    qsigns[maskx] = qsigns[maskx]*qs_flipx
    qsigns[masky] = qsigns[masky]*qs_flipy
    qsigns[maskz] = qsigns[maskz]*qs_flipz

    # field computations --------------------------------------------
    # Note: in principle the computation for all three mag-components can be
    #   vectorized itself using symmetries. However, tiling the three
    #   components will cost more than is gained by the vectorized evaluation

    # Note: making the following computation steps is not necessary
    #   as mkl will cache such small computations
    xma, xpa = x-a, x+a
    ymb, ypb = y-b, y+b
    zmc, zpc = z-c, z+c

    xma2, xpa2 = xma**2, xpa**2
    ymb2, ypb2 = ymb**2, ypb**2
    zmc2, zpc2 = zmc**2, zpc**2

    mmm = np.sqrt(xma2 + ymb2 + zmc2)
    pmp = np.sqrt(xpa2 + ymb2 + zpc2)
    pmm = np.sqrt(xpa2 + ymb2 + zmc2)
    mmp = np.sqrt(xma2 + ymb2 + zpc2)
    mpm = np.sqrt(xma2 + ypb2 + zmc2)
    ppp = np.sqrt(xpa2 + ypb2 + zpc2)
    ppm = np.sqrt(xpa2 + ypb2 + zmc2)
    mpp = np.sqrt(xma2 + ypb2 + zpc2)

    ff2x = np.log((xma+mmm) * (xpa+ppm) * (xpa+pmp) * (xma+mpp))     \
            -np.log((xpa+pmm) * (xma+mpm) * (xma+mmp) * (xpa+ppp))

    ff2y = np.log((-ymb+mmm) * (-ypb+ppm) * (-ymb+pmp) * (-ypb+mpp)) \
           -np.log((-ymb+pmm) * (-ypb+mpm) * ( ymb-mmp) * ( ypb-ppp))

    ff2z = np.log((-zmc+mmm) * (-zmc+ppm) * (-zpc+pmp) * (-zpc+mpp)) \
           -np.log((-zmc+pmm) * ( zmc-mpm) * (-zpc+mmp) * ( zpc-ppp))

    ff1x = (np.arctan2((ymb*zmc),(xma*mmm)) - np.arctan2((ymb*zmc),(xpa*pmm))
            - np.arctan2((ypb*zmc),(xma*mpm)) + np.arctan2((ypb*zmc),(xpa*ppm))
            - np.arctan2((ymb*zpc),(xma*mmp)) + np.arctan2((ymb*zpc),(xpa*pmp))
            + np.arctan2((ypb*zpc),(xma*mpp)) - np.arctan2((ypb*zpc),(xpa*ppp)))

    ff1y = (np.arctan2((xma*zmc),(ymb*mmm)) - np.arctan2((xpa*zmc),(ymb*pmm))
            - np.arctan2((xma*zmc),(ypb*mpm)) + np.arctan2((xpa*zmc),(ypb*ppm))
            - np.arctan2((xma*zpc),(ymb*mmp)) + np.arctan2((xpa*zpc),(ymb*pmp))
            + np.arctan2((xma*zpc),(ypb*mpp)) - np.arctan2((xpa*zpc),(ypb*ppp)))

    ff1z = (np.arctan2((xma*ymb),(zmc*mmm)) - np.arctan2((xpa*ymb),(zmc*pmm))
            - np.arctan2((xma*ypb),(zmc*mpm)) + np.arctan2((xpa*ypb),(zmc*ppm))
            - np.arctan2((xma*ymb),(zpc*mmp)) + np.arctan2((xpa*ymb),(zpc*pmp))
            + np.arctan2((xma*ypb),(zpc*mpp)) - np.arctan2((xpa*ypb),(zpc*ppp)))

    # contributions from x-magnetization
    bx_magx = magx * ff1x * qsigns[:,0,0]  # the 'missing' third sign is hidden in ff1x
    by_magx = magx * ff2z * qsigns[:,0,1]
    bz_magx = magx * ff2y * qsigns[:,0,2]
    # contributions from y-magnetization
    bx_magy =  magy * ff2z * qsigns[:,1,0]
    by_magy =  magy * ff1y * qsigns[:,1,1]
    bz_magy = -magy * ff2x * qsigns[:,1,2]
    # contributions from z-magnetization
    bx_magz =  magz * ff2y * qsigns[:,2,0]
    by_magz = -magz * ff2x * qsigns[:,2,1]
    bz_magz =  magz * ff1z * qsigns[:,2,2]

    # summing all contributions
    bx_tot = bx_magx + bx_magy + bx_magz
    by_tot = by_magx + by_magy + by_magz
    bz_tot = bz_magx + bz_magy + bz_magz

    # combine with special edge/corner cases
    # B = np.c_[bx_tot, by_tot, bz_tot]      # faster for 10^5 and more evaluations
    B = np.concatenate(((bx_tot,),(by_tot,),(bz_tot,)), axis=0).T

    return B / (4*np.pi)

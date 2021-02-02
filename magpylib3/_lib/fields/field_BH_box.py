"""
Implementations of analytical expressions for the magnetic field of homogeneously magnetized Cuboids.
Computation details in function docstrings.
"""

import numpy as np


def field_BH_box(bh: bool, mag: np.ndarray, dim: np.ndarray, pos_obs: np.ndarray,) -> np.ndarray:
    """ Wrapper function to select box B- or H-field, which are treated equally
    at higher levels

    ### Args:
    - bh (boolean): True=B, False=H
    - mag (ndarray Nx3): homogeneous magnetization vector in units of mT
    - dim (ndarray Nx3): dimension of Cuboid side lengths in units of mm
    - pos_obs (ndarray Nx3): position of observer in units of mm

    ### Returns:
    - B/H-field (ndarray Nx3): magnetic field vectors at pos_obs in units of mT / kA/m
    """
    if bh:
        return field_B_box(mag, dim, pos_obs)
    else:
        return field_H_box(mag, dim, pos_obs)


def field_B_box(mag: np.ndarray, dim: np.ndarray, pos_obs: np.ndarray) -> np.ndarray:
    """ Compute B-field of Cuboid magnet with homogenous magnetization.

    ### Args:
    - mag (ndarray Nx3): homogeneous magnetization vector in units of mT
    - dim (ndarray Nx3): dimension of Cuboid side lengths in units of mm
    - pos_obs (ndarray Nx3): position of observer in units of mm

    ### Returns:
    - B-field (ndarray Nx3): B-field vectors at pos_obs in units of mT

    ### init_state:
    A Cuboid with side lengths a,b,c. The sides are parallel to the 
    axes x,y,z of a Cartesian CS. The geometric center of the Cuboid 
    is in the origin of the CS.

    ### Computation info:
    Field computations via magnetic surface charge density. See e.g.
    - Camacho: Revista Mexicana de F´ısica E 59 (2013) 8–17
    - Engel-Herbert: Journal of Applied Physics 97(7):074504 - 074504-4
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

    Indeterminate forms at box edges and corners remain. The numerical 
    evaluation is instable in the vicinity. Avoid field computation near
    (1e-6*sidelength) Cuboid edges. FIX THIS PROBLEM !!!
    """
    
    B = np.empty((len(mag),3))   # store fields here eventually

    # special cases for edge/corner fields --------------------------
    x, y, z = np.copy(pos_obs.T)
    a, b, c = dim.T/2

    mx1 = ( x == a) | (x == -a) # on edgeline
    my1 = ( y == b) | (y == -b)
    mz1 = ( z == c) | (z == -c)
    mx2 = (-a <= x) & (x <=  a) # within actual edge
    my2 = (-b <= y) & (y <=  b)
    mz2 = (-c <= z) & (z <=  c)

    mask_xedge = my1 * mz1 * mx2 
    mask_yedge = mx1 * mz1 * my2
    mask_zedge = mx1 * my1 * mz2
    
    B[mask_xedge] = 0
    B[mask_yedge] = 0
    B[mask_zedge] = 0

    # create inputs excluding edge/corner cases----------------------
    mask_not_edge = np.invert(mask_xedge | mask_yedge | mask_zedge)
    magx, magy, magz = mag[mask_not_edge].T
    x, y, z = pos_obs[mask_not_edge].T
    a, b, c = dim[mask_not_edge].T/2
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
    Bx_magx = magx * ff1x * qsigns[:,0,0]  # the 'missing' third sign is hidden in ff1x
    By_magx = magx * ff2z * qsigns[:,0,1]
    Bz_magx = magx * ff2y * qsigns[:,0,2]
    # contributions from y-magnetization
    Bx_magy =  magy * ff2z * qsigns[:,1,0]
    By_magy =  magy * ff1y * qsigns[:,1,1]
    Bz_magy = -magy * ff2x * qsigns[:,1,2]
    # contributions from z-magnetization
    Bx_magz =  magz * ff2y * qsigns[:,2,0]
    By_magz = -magz * ff2x * qsigns[:,2,1]
    Bz_magz =  magz * ff1z * qsigns[:,2,2]

    # summing all contributions
    Bxtot = Bx_magx + Bx_magy + Bx_magz
    Bytot = By_magx + By_magy + By_magz
    Bztot = Bz_magx + Bz_magy + Bz_magz

    # combine with special edge/corner cases 
    B[mask_not_edge] = np.array([Bxtot, Bytot, Bztot]).T

    return B / (4*np.pi)


def field_H_box(mag:np.ndarray, dim:np.ndarray, pos_obs:np.ndarray) -> np.ndarray:
    """ Compute H-field of Cuboid magnet with homogenous magnetization.

    ### Args:
    - mag (ndarray Nx3): homogeneous magnetization vector in units of mT
    - dim (ndarray Nx3): dimension of Cuboid side lengths in units of mm
    - pos_obs (ndarray Nx3): position of observer in units of mm

    ### Returns:
    - H-field (ndarray Nx3): H-field vectors at pos_obs in units of kA/m
    
    ### Computation info:
    Computation based on B-field computation.
    """

    B = field_B_box(mag, pos_obs, dim)
    
    # if inside magnet subtract magnetization vector
    pa = np.abs(pos_obs)
    c1 = pa[:,0]<=dim[:,0]/2
    c2 = pa[:,1]<=dim[:,1]/2
    c3 = pa[:,2]<=dim[:,2]/2
    mask = c1*c2*c3
    B[mask] -= mag[mask]

    # transform units mT -> kA/m
    mu0 = 4*np.pi*1e-7
    H = B/mu0/1e6

    return H
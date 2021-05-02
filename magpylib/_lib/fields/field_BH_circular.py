"""
Implementations of analytical expressions for the magnetic field of
a circular current loop. Computation details in function docstrings.
"""

import numpy as np
from magpylib._lib.fields.special_functions import celv


def field_BH_circular(
    bh: bool,
    current: np.ndarray,
    dim: np.ndarray,
    pos_obs: np.ndarray
    ) -> list:
    """ Compute B-field of circular current loop carrying a unit current
    in the cylindrical CS

    ### Args:
    - dim  (ndarray N): diameter of current loop in units of [mm]
    - obs_pos (ndarray Nx3): position of observer (x,y,z) in [mm]

    ### Returns:
    - Bfield, ndarray (N,3), in cartesian CS in units of [mT]

    ### init_state:
    A circular current loop with diameter d lies in the x-y plane. The geometric
    center of the current loop is the origin of the CS.

    ### Computation info:
    This field can be ontained by direct application of the Biot-Savardt law.
    Several sources in the literature provides these formulas.
    - Smythe, "Static and dynamic electricity" McGraw-Hill New York, 1950, vol. 3.
    - Simpson, "Simple analytic expressions for the magnetic field of a circular
        current loop," 2001.
    - Ortner, "Feedback of Eddy Currents in Layered Materials for Magnetic Speed Sensing",
        IEEE Transactions on Magnetics ( Volume: 53, Issue: 8, Aug. 2017)

    ### Numerical instabilities:
        - singularity at r=r0 & z=0 (pos_obs = wire pos). Set to zero there
    """

    # inputs   -----------------------------------------------------------
    x, y, z = pos_obs.T
    r0 = dim/2
    n = len(x)

    # cylindrical coordinates ----------------------------------------------
    r, phi = np.sqrt(x**2+y**2), np.arctan2(y, x)
    #pos_obs_cy = np.concatenate(((r,),(phi,),(z,)),axis=0).T
    #pos_obs_cy = np.array([r,phi,z]).T

    # allocate fields as zeros ----------------------------------------------
    #    singularities will become zero, other will be overwritten
    Br_all = np.zeros(n)
    Bz_all = np.zeros(n)

    # determine singularity points (pos_obs on wire) ------------------------
    mask0 = (r==r0)*(z==0)

    # forward only non-singularity observer positions for computation--------
    r0 = r0[~mask0]
    r = r[~mask0]
    z = z[~mask0]
    n = np.sum(~mask0)
    current = current[~mask0]

    # pre compute small quantities that might not get stored in the cache otherwise
    r2 = r**2
    r02 = r0**2
    z2 = z**2
    brack = (z2 + (r0+r)**2)
    k2 = 4*r*r0/brack
    k_over_sq_rr0 = 2/np.sqrt(brack)

    # evaluate complete elliptic integrals ------------------------------------
    one = np.ones(n)
    ellipk = celv(np.sqrt(1-k2), one, one, one)
    ellipe = celv(np.sqrt(1-k2), one, one, 1-k2)

    # compute fields from formulas (paper Ortner) -----------------------------
    mask1 = r==0
    z_over_r = np.zeros(n)
    z_over_r[~mask1] = z[~mask1]/r[~mask1] # will be zero when r=0 -> Br=0

    prefactor = 1/10*current
    Br = prefactor/2*k_over_sq_rr0 * z_over_r * ((2-k2)/(1-k2)*ellipe - 2*ellipk)
    Bz = prefactor*k_over_sq_rr0 * (ellipk - (r2-r02+z2)/((r0-r)**2+z2)*ellipe)

    # insert non-singular computations into total vectors----------------------
    Br_all[~mask0] = Br
    Bz_all[~mask0] = Bz

    # transform field to cartesian CS -----------------------------------------
    Bx = Br_all*np.cos(phi)
    By = Br_all*np.sin(phi)
    B_cart = np.concatenate(((Bx,),(By,),(Bz_all,)),axis=0).T # ugly but fast

    if bh:
        return B_cart

    # transform to H-field
    return B_cart*10/4/np.pi

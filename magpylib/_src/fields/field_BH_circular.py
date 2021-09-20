"""
Implementations of analytical expressions for the magnetic field of
a circular current loop. Computation details in function docstrings.
"""
# pylint: disable=no-name-in-module

import numpy as np
from scipy.special import ellipe, ellipk


def field_BH_circular(
    bh: bool,
    current: np.ndarray,
    dia: np.ndarray,
    pos_obs: np.ndarray
    ) -> list:
    """
    Field of circular current loop.
    - wraps fundamental implementation Smythe1950
    - selects B or H
    - sets singularity at wire to 0
    - Cylinder CS <-> Cartesian CS
    """
    x, y, z = pos_obs.T
    r0 = dia/2
    n = len(x)

    # cylindrical coordinates ----------------------------------------------
    r, phi = np.sqrt(x**2+y**2), np.arctan2(y, x)
    pos_obs_cy = np.concatenate(((r,),(z,)),axis=0).T

    # allocate fields as zeros ----------------------------------------------
    #    singularities will become zero, other will be overwritten
    Br_all = np.zeros(n)
    Bz_all = np.zeros(n)

    # determine singularity positions (pos_obs on wire) ------------------------
    mask0 = (r==r0)*(z==0)

    # forward only non-singularity observer positions for computation--------
    Br, Bz = current_loop_B_Smythe1950(r0[~mask0], pos_obs_cy[~mask0]).T

    # insert non-singular computations into total vectors----------------------
    Br_all[~mask0] = Br
    Bz_all[~mask0] = Bz

    # transform field to cartesian CS -----------------------------------------
    Bx = Br_all*np.cos(phi)
    By = Br_all*np.sin(phi)
    B_cart = np.concatenate(((Bx,),(By,),(Bz_all,)),axis=0) # ugly but fast

    B_cart *= current/10

    # B or H field
    if bh:
        return B_cart.T
    return B_cart.T*10/4/np.pi


# ON INTERFACE
def current_loop_B_Smythe1950(
    radius: np.ndarray,
    pos_obs: np.ndarray
    ) -> np.ndarray:
    """
    B-field in cylindrical CS of circular line-current loop.
    The current loop lies in the z=0 plane with the origin at its center.

    Implementation from [Smythe1950], [Simpson2001], [Ortner2017].

    Parameters
    ----------
    radius: ndarray, shape (n,)
        radius of current loop in units of [mm].

    pos_obs: ndarray, shape (n,2)
        position of observer in cylindrical coordinates (r, z)
        in units of [mm].

    Returns
    -------
    B-field: ndarray
        B-field of current loop in cylindrical coordinates (Br, Bz),
        shape (n,2) in units of [mT].

    Examples
    --------
    Compute the field of three different loops at three different positions.

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> rad = np.array([1,2,3])
    >>> obs = np.array([(1,1), (2,2), (3,3)])
    >>> B = magpy.lib.current_loop_B_Smythe1950(rad, obs)
    >>> print(B)
    [[1.14331448 0.96483239]
     [0.57165724 0.48241619]
     [0.38110483 0.3216108 ]]

    Notes
    -----
    This field can be obtained by direct application of the Biot-Savardt law.
    Several sources in the literature provides these formulas:

    Smythe, "Static and dynamic electricity" McGraw-Hill New York, 1950, vol. 3.

    Simpson, "Simple analytic expressions for the magnetic field of a circular current
    loop," 2001.

    Ortner, "Feedback of Eddy Currents in Layered Materials for Magnetic Speed Sensing",
    IEEE Transactions on Magnetics ( Volume: 53, Issue: 8, Aug. 2017)

    Numerical instabilities: see discussion on GitHub
    """

    # inputs   -----------------------------------------------------------
    r0 = radius
    r, z = pos_obs.T
    n = len(r0)

    # pre compute small quantities that might not get stored in the cache otherwise
    r2 = r**2
    r02 = r0**2
    z2 = z**2
    brack = (z2 + (r0+r)**2)
    k2 = 4*r*r0/brack
    k_over_sq_rr0 = 2/np.sqrt(brack)

    # evaluate complete elliptic integrals ------------------------------------
    ellip_e = ellipe(k2)
    ellip_k = ellipk(k2)

    # compute fields from formulas [Ortner2017] -----------------------------
    mask1 = r==0
    z_over_r = np.zeros(n)
    z_over_r[~mask1] = z[~mask1]/r[~mask1] # will be zero when r=0 -> Br=0

    Br = .5*k_over_sq_rr0 * z_over_r * ((2-k2)/(1-k2)*ellip_e - 2*ellip_k)
    Bz = k_over_sq_rr0 * (ellip_k - (r2-r02+z2)/((r0-r)**2+z2)*ellip_e)

    return np.array([Br, Bz]).T

"""
Implementations of analytical expressions for the magnetic field of
a circular current loop. Computation details in function docstrings.
"""
# pylint: disable=no-name-in-module

import numpy as np
from scipy.special import ellipe
from magpylib._src.fields.special_cel import cel_loop_stable


def field_BH_loop(
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
    Br, Bz = current_loop_B_Leitner2021(r0[~mask0], pos_obs_cy[~mask0]).T

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
def current_loop_B_Leitner2021(
    radius: np.ndarray,
    pos_obs: np.ndarray
    ) -> np.ndarray:
    """
    B-field in cylindrical CS of circular line-current loop. The
    current loop lies in the z=0 plane with the origin at its center.
    The field is computed with 12-14 digits precision. Implementation
    from [Leitner2021].

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

    Leitner, "work in progress"
    """

    # inputs   -----------------------------------------------------------
    r0 = radius
    r, z = pos_obs.T
    n = len(r0)

    # make dimensionless
    rb = r/r0
    zb = z/r0

    # pre-compute small quantities that mighjt not be cached
    z2 = zb**2
    brack = (z2+(rb+1)**2)
    k2 = 4*rb/brack
    pf = 1/np.sqrt(brack)/(1-k2)
    xi = cel_loop_stable(k2)

    # rb=0 requires special treatment
    mask1 = rb==0
    z_over_r = np.zeros(n)
    k2_over_rb = np.ones(n)*4/(z2+1)
    z_over_r[~mask1] = z[~mask1]/rb[~mask1]    # will be zero when r=0
    k2_over_rb[~mask1] = k2[~mask1]/rb[~mask1] # will be zero when r=0

    # field components
    Br = pf * z_over_r * xi
    Bz = pf * (k2_over_rb*ellipe(k2) - xi)

    return np.array([Br, Bz]).T

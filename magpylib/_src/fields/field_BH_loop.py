"""
Implementations of analytical expressions for the magnetic field of
a circular current loop. Computation details in function docstrings.
"""
# pylint: disable=no-name-in-module # reason scipy.special.ellipe import

import numpy as np
from scipy.special import ellipe
from magpylib._src.fields.special_cel import cel_loop_stable
from magpylib._src.utility import cart_to_cyl_coordinates, cyl_field_to_cart


def field_BH_loop(
    bh: bool,
    current: np.ndarray,
    dia: np.ndarray,
    pos_obs: np.ndarray
    ) -> list:
    """
    Field of circular current loop.

    - wraps fundamental implementation
    - selects B or H
    - Cylinder CS <-> Cartesian CS

    Parameters:
    ----------
    - bh: boolean, True=B, False=H
    - current: ndarray shape (n,), current in units of [A]
    - dia: ndarray shape (n,), diameter in units of [mm]
    - pos_obs: ndarray shape (n,3), position of observer in units of mm

    Returns:
    --------
    B/H-field (ndarray Nx3): magnetic field vectors at pos_obs in units of mT / kA/m
    """

    r0 = dia/2
    r, phi, z = cart_to_cyl_coordinates(pos_obs)

    # compute field
    pos_obs_cy = np.concatenate(((r,), (z,)),axis=0).T
    Br, Bz = current_loop_Bfield(current, r0, pos_obs_cy).T

    # transform field to cartesian CS
    Bx, By = cyl_field_to_cart(phi, Br)
    B_cart = np.concatenate(((Bx,),(By,),(Bz,)),axis=0) # ugly but fast

    # B or H field
    if bh:
        return B_cart.T
    return B_cart.T*10/4/np.pi


# ON INTERFACE
def current_loop_Bfield(
    current: np.ndarray,
    radius: np.ndarray,
    observer: np.ndarray
    ) -> np.ndarray:
    """
    B-field in cylindrical CS of circular line-current loop. The
    current loop lies in the z=0 plane with the origin at its center.

    The field is computed with 12-14 digits precision. On the loop the
    result is set 0 (instead of nan). Implementation from Ortner/Leitner wip.

    Parameters
    ----------
    current: ndarray, shape (n,)
        Electrical current in loop in units of [A].

    radius: ndarray, shape (n,)
        Radius of current loop in units of [mm].

    observer: ndarray, shape (n,2)
        position of observer in cylindrical coordinates (r, z)
        in units of [mm].

    Returns
    -------
    B-field: ndarray, shape (n,2)
        B-field of current loop in cylindrical coordinates (Br, Bz),
        shape (n,2) in units of [mT].

    Examples
    --------
    Compute the field of three different loops at three different positions.

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> cur = np.array([1,1,2])
    >>> rad = np.array([1,2,3])
    >>> obs = np.array([(1,1), (2,2), (3,3)])
    >>> B = magpy.lib.current_loop_Bfield(cur, rad, obs)
    >>> print(B)
    [[0.11433145 0.09648324]
     [0.05716572 0.04824162]
     [0.07622097 0.06432216]]

    Notes
    -----
    This field can be obtained by direct application of the Biot-Savardt law.
    Several sources in the literature provides these formulas:

    Smythe, "Static and dynamic electricity" McGraw-Hill New York, 1950, vol. 3.

    Simpson, "Simple analytic expressions for the magnetic field of a circular current
    loop," 2001.

    Ortner, "Feedback of Eddy Currents in Layered Materials for Magnetic Speed Sensing",
    IEEE Transactions on Magnetics ( Volume: 53, Issue: 8, Aug. 2017)

    Leitner/Ortner, "work in progress"
    """

    rad = np.abs(radius)
    n = len(rad)
    r, z = observer.T

    # allocate with zeros to deal with special cases ON_LOOP and RADIUS=0
    B_total = np.zeros((n,2))
    mask_radius0 = rad==0
    # rel pos deviation by 1e-15 to account for num errors (e.g. when rotating)
    mask_on_loop = np.logical_and(abs(r-rad)<1e-15*rad, z==0)
    mask_general = ~np.logical_or(mask_radius0, mask_on_loop)

    # collect general case inputs
    r = r[mask_general]
    z = z[mask_general]
    rad = rad[mask_general]
    current = current[mask_general]
    n = len(rad)

    # express through ratios (make dimensionless, avoid large/small input values)
    rb = r/rad
    zb = z/rad

    # pre-compute small quantities that might not be cached
    z2 = zb**2
    brack = (z2+(rb+1)**2)
    k2 = 4*rb/brack
    xi = cel_loop_stable(k2)

    # rb=0 (on z-axis) requires special treatment because ellipe(x)/x and
    # cel(x)/x for x->0 both appear in the expressions, both are finite at x=0
    # but evaluation gives 0/0=nan
    # To avoid treating x=0 as a special case (with masks), one would have to find
    # designated algorithms for these expressions (which require additional evaluation
    # and computation time because xi must be evaluated in any case).

    mask1 = rb==0
    z_over_r = np.zeros(n)
    k2_over_rb = np.ones(n)*4/(z2+1)
    z_over_r[~mask1] = zb[~mask1]/rb[~mask1]    # will be zero when r=0
    k2_over_rb[~mask1] = k2[~mask1]/rb[~mask1]  # will be zero when r=0

    # field components
    pf = 1/np.sqrt(brack)/(1-k2)/rad
    Br = pf * z_over_r * xi
    Bz = pf * (k2_over_rb*ellipe(k2) - xi)

    # current and [mT] unit
    B_total[mask_general] = (np.array([Br, Bz])*current).T/10

    return B_total

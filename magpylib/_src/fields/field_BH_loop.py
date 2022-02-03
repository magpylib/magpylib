"""
Implementations of analytical expressions for the magnetic field of
a circular current loop. Computation details in function docstrings.
"""
# pylint: disable=no-name-in-module # reason scipy.special.ellipe import

import numpy as np
from scipy.special import ellipe
from magpylib._src.fields.special_cel import cel_loop_stable
from magpylib._src.utility import cart_to_cyl_coordinates, cyl_field_to_cart


# ON INTERFACE
def current_loop_field(
    current: np.ndarray,
    diameter: np.ndarray,
    observer: np.ndarray,
    Bfield=True
    ) -> np.ndarray:
    """
    B-field in in Cartesian CS of a circular line-current loop. The
    current loop lies in the z=0 plane with the origin at its center.

    The field is computed with >12 digits precision. On the loop the
    result is set to (0,0,0). Implementation from Ortner/Leitner wip.

    Parameters
    ----------
    current: ndarray, shape (n,)
        Electrical current in loop in units of [A].

    diameter: ndarray, shape (n,)
        Diameter of current loop in units of [mm].

    observer: ndarray, shape (n,3)
        Position of observer in Cartesian coordinates (x,y,z)
        in units of [mm].

    Bfield: bool, default=True
        If True return B-field in units of [mT], else return H-field in units of [kA/m].

    Returns
    -------
    B-field or H-field: ndarray, shape (n,3)
        B/H-field of current in Cartesian coordinates (Bx, By, Bz) in units of [mT]/[kA/m].

    Examples
    --------
    Compute the field of three different loops at three different positions.

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> cur = np.array([1,1,2])
    >>> dia = np.array([2,4,6])
    >>> obs = np.array([(1,1), (2,2), (3,3)])
    >>> B = magpy.lib.current_loop_field(cur, dia, obs)
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

    r, phi, z = cart_to_cyl_coordinates(observer)
    rad = np.abs(diameter/2)
    n = len(rad)

    # define masks for special cases
    mask_radius0 = rad==0
    # rel pos deviation by 1e-15 to account for num errors (e.g. when rotating)
    mask_on_loop = np.logical_and(abs(r-rad)<1e-15*rad, z==0)
    mask_general = ~np.logical_or(mask_radius0, mask_on_loop)

    # collect general case inputs
    r = r[mask_general]
    z = z[mask_general]
    rad = rad[mask_general]
    nX = len(rad)

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
    z_over_r = np.zeros(nX)
    k2_over_rb = np.ones(nX)*4/(z2+1)
    z_over_r[~mask1] = zb[~mask1]/rb[~mask1]    # will be zero when r=0
    k2_over_rb[~mask1] = k2[~mask1]/rb[~mask1]  # will be zero when r=0

    # field components
    pf = 1/np.sqrt(brack)/(1-k2)/rad
    Br = pf * z_over_r * xi
    Bz = pf * (k2_over_rb*ellipe(k2) - xi)

    # current and [mT] unit
    Br_tot, Bz_tot = np.zeros((2,n))
    Br_tot[mask_general] = Br
    Bz_tot[mask_general] = Bz


    # transform field to cartesian CS
    Bx_tot, By_tot = cyl_field_to_cart(phi, Br_tot)
    B_cart = (np.concatenate(((Bx_tot,),(By_tot,),(Bz_tot,)), axis=0)*current).T # ugly but fast

    # B or H field
    if Bfield:
        return B_cart/10
    return B_cart/4/np.pi

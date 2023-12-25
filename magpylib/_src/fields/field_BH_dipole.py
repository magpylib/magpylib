"""
Core implementation of dipole field
"""
import numpy as np

from magpylib._src.input_checks import check_field_input
from magpylib._src.utility import MU0


# CORE
def dipole_field(
    *,
    field: str,
    observers: np.ndarray,
    moment: np.ndarray,
) -> np.ndarray:
    """Magnetic field of a dipole moments.

    The dipole moment lies in the origin of the coordinate system.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    field: str, default=`'B'`
        If `field='B'` return B-field in units of T, if `field='H'` return H-field
        in units of A/m.

    observers: ndarray, shape (n,3)
        Observer positions (x,y,z) in Cartesian coordinates in units of m.

    moment: ndarray, shape (n,3)
        Dipole moment vector in units of A*m^2.

    Returns
    -------
    B-field or H-field: ndarray, shape (n,3)
        B- or H-field of source in Cartesian coordinates in units of T or A/m.

    Examples
    --------
    Compute the B-field of two different dipole-observer instances.

    >>> import magpylib as magpy
    >>> import numpy as np
    >>> B = magpy.core.dipole_field(
    ...     field="B",
    ...     observers=np.array([(1,2,3), (-1,-2,-3)]),
    ...     moment=np.array([(0,0,1e6), (1e5,0,1e7)])
    ... )
    >>> print(B)
    [[0.00122722 0.00245444 0.00177265]
     [0.01212221 0.02462621 0.01784923]]

    Notes
    -----
    Advanced unit use: The input unit of magnetization and polarization
    gives the output unit of H and B. All results are independent of the
    length input units. One must be careful, however, to use consistently
    the same length unit throughout a script.

    The moment of a magnet is given by its volume*magnetization.
    """
    bh = check_field_input(field, "dipole_field()")

    x, y, z = observers.T
    r = np.sqrt(x**2 + y**2 + z**2)  # faster than np.linalg.norm
    with np.errstate(divide="ignore", invalid="ignore"):
        # 0/0 produces invalid warn and results in np.nan
        # x/0 produces divide warn and results in np.inf
        B = (
            3 * np.sum(moment * observers, axis=1) * observers.T / r**5
            - moment.T / r**3
        ).T * 1e-7

    # when r=0 return np.inf in all non-zero moment directions
    mask1 = r == 0
    if np.any(mask1):
        with np.errstate(divide="ignore", invalid="ignore"):
            B[mask1] = moment[mask1] / 0.0
            np.nan_to_num(B, copy=False, posinf=np.inf, neginf=np.NINF)

    if bh:
        return B

    H = B / MU0
    return H

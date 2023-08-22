"""
Dipole implementation
"""
import numpy as np

from magpylib._src.input_checks import check_field_input


# CORE
def dipole_field(
    field: str,
    observers: np.ndarray,
    moment: np.ndarray,
) -> np.ndarray:
    """Magnetic field of a dipole moment.

    The dipole moment lies in the origin of the coordinate system.

    Parameters
    ----------
    field: str, default=`'B'`
        If `field='B'` return B-field in units of mT, if `field='H'` return H-field
        in units of kA/m.

    observers: ndarray, shape (n,3)
        Observer positions (x,y,z) in Cartesian coordinates in units of mm.

    moment: ndarray, shape (n,3)
        Dipole moment vector in units of mT*mm^3.

    Returns
    -------
    B-field or H-field: ndarray, shape (n,3)
        B/H-field of dipole in Cartesian coordinates (Bx, By, Bz) in units of mT/(kA/m).

    Examples
    --------
    Compute the B-field of two different dipole-observer instances.

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> mom = np.array([(1,2,3), (0,0,1)])
    >>> obs = np.array([(1,1,1), (0,0,2)])
    >>> B = magpy.core.dipole_field('B', obs, mom)
    >>> print(B)
    [[0.07657346 0.06125877 0.04594407]
     [0.         0.         0.01989437]]

    Notes
    -----
    The field is similar to the outside-field of a spherical magnet with Volume = 1 mm^3.
    """
    bh = check_field_input(field, "dipole_field()")

    x, y, z = observers.T
    r = np.sqrt(x**2 + y**2 + z**2)  # faster than np.linalg.norm
    with np.errstate(divide="ignore", invalid="ignore"):
        # 0/0 produces invalid warn and results in np.nan
        # x/0 produces divide warn and results in np.inf
        B = (
            (
                3 * np.sum(moment * observers, axis=1) * observers.T / r**5
                - moment.T / r**3
            ).T
            / 4
            / np.pi
        )

    # when r=0 return np.inf in all non-zero moment directions
    mask1 = r == 0
    if np.any(mask1):
        with np.errstate(divide="ignore", invalid="ignore"):
            B[mask1] = moment[mask1] / 0.0
            np.nan_to_num(B, copy=False, posinf=np.inf, neginf=np.NINF)

    # return B or H
    if bh:
        return B

    H = B * 10 / 4 / np.pi
    return H

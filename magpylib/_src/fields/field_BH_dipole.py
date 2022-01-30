"""
Dipole implementation
"""

import numpy as np


def field_BH_dipole(
        bh: bool,
        moment: np.ndarray,
        observer: np.ndarray
        ) -> np.ndarray:
    """
    Dipole field
    - wraps fundamental dipole formula
    - selects B or H

    Parameters:
    ----------
    - bh (boolean): True=B, False=H
    - moment (ndarray Nx3): dipole moment vector in units of mT*mm^3
    - observer (ndarray Nx3): position of observer in units of mm

    Returns:
    --------
    B/H-field (ndarray Nx3): magnetic field vectors at pos_obs in units of mT / kA/m
    """
    B = dipole_Bfield(moment, observer)

    if bh:
        return B

    # adjust and return H
    H = B*10/4/np.pi
    return H


def dipole_Bfield(
    moment: np.ndarray,
    observer: np.ndarray
    ) -> np.ndarray:
    """
    The B-field of a magnetic dipole moment. The field is similar to the outside-field
    of a spherical magnet with Volume = 1 [mm^3].

    Parameters:
    ----------
    - moment: ndarray, shape (n,3)
        Dipole moment vector in units of [mT*mm^3].
    - observer: ndarray, shape (n,3)
        Position of observer in units of [mm].

    Returns:
    --------
    B-field: ndarray, shape (n,3)
        B-field field vectors at observer positions in units of [mT].

    Examples:
    ---------
    >>> import numpy as np
    >>> import magpylib as magpy
    >>> mom = np.array([(1,2,3), (0,0,1)])
    >>> obs = np.array([(1,1,1), (0,0,2)])
    >>> B = magpy.lib.dipole_Bfield(mom, obs)
    >>> print(B)
    [[0.07657346 0.06125877 0.04594407]
     [0.         0.         0.01989437]]
    """

    x, y, z = observer.T
    r = np.sqrt(x**2+y**2+z**2)   # faster than np.linalg.norm
    with np.errstate(divide='ignore', invalid='ignore'):
        # 0/0 produces invalid warn and results in np.nan
        # x/0 produces divide warn and results in np.inf
        B = (3*np.sum(moment*observer,axis=1)*observer.T/r**5 - moment.T/r**3).T/4/np.pi

    # when r=0 return np.inf in all non-zero moment directions
    mask1 = r==0
    if np.any(mask1):
        with np.errstate(divide='ignore', invalid='ignore'):
            B[mask1] = moment[mask1]/0.
            np.nan_to_num(B, copy=False, posinf=np.inf, neginf=np.NINF)

    return B

"""
Dipole implementation
"""

import numpy as np


def field_BH_dipole(
        bh: bool,
        moment: np.ndarray,
        pos_obs: np.ndarray
        ) -> np.ndarray:
    """
    The B-field of a magnetic dipole moment. The solution is set up so that
    the moment is given in mT*mm^3, i.e. the field is similar to the outside-field
    of a spherical magnet with Volume = 1 mm^3.

    Paramters:
    ----------
    - bh (boolean): True=B, False=H
    - moment (ndarray Nx3): dipole moment vector in units of mT*mm^3
    - pos_obs (ndarray Nx3): position of observer in units of mm

    Returns:
    --------
    B/H-field (ndarray Nx3): magnetic field vectors at pos_obs in units of mT / kA/m
    """

    x, y, z = pos_obs.T
    r = np.sqrt(x**2+y**2+z**2)   # faster than np.linalg.norm
    B = (3*(np.sum(moment*pos_obs,axis=1)*pos_obs.T)/r**5 - moment.T/r**3).T/4/np.pi

    if bh:
        return B

    # adjust and return H
    H = B*10/4/np.pi
    return H

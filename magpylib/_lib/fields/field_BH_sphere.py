"""
Implementations of analytical expressions for the magnetic field of homogeneously
magnetized Spheres. Computation details in function docstrings.
"""

import numpy as np


def field_BH_sphere(
        bh: bool,
        mag: np.ndarray,
        dim: np.ndarray,
        pos_obs: np.ndarray
        ) -> np.ndarray:
    """
    The B-field of a homogeneously magnetized spherical magnet corresponds to a dipole
    field on the outside and is 2/3*mag in the inside (see e.g. "Theoretical Physics, Bertelmann")

    - separate mag=0 cases (returning 0)
    - select B or H

    Paramters:
    ----------
    - bh (boolean): True=B, False=H
    - mag (ndarray Nx3): homogeneous magnetization vector in units of mT
    - dim (ndarray Nx3): Sphere diameter
    - pos_obs (ndarray Nx3): position of observer in units of mm

    Returns:
    --------
    B/H-field (ndarray Nx3): magnetic field vectors at pos_obs in units of mT / kA/m
    """

    x, y, z = np.copy(pos_obs.T)
    r = np.sqrt(x**2+y**2+z**2)   # faster than np.linalg.norm
    radius = dim/2

    # inside field & allocate
    B = mag*2/3

    # overwrite outside field
    # special case mag=0 automatically reflected in field formulas
    mask_out = r>=radius

    mag1 = mag[mask_out]
    poso1 = pos_obs[mask_out]
    r1 = r[mask_out]
    dim1 = dim[mask_out]
    val = (3*(np.sum(mag1*poso1,axis=1)*poso1.T)/r1**5 - mag1.T/r1**3)*dim1**3/24
    B[mask_out] = val.T

    # return B
    if bh:
        return B

    # adjust and return H
    mask_in = ~mask_out
    B[mask_in] = -mag[mask_in]/3
    H = B*10/4/np.pi
    return H

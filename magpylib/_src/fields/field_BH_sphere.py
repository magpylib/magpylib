"""
Implementations of analytical expressions for the magnetic field of homogeneously
magnetized Spheres. Computation details in function docstrings.
"""

import numpy as np


def field_BH_sphere(
        bh: bool,
        mag: np.ndarray,
        dia: np.ndarray,
        pos_obs: np.ndarray
        ) -> np.ndarray:
    """
    wraps fundamental sphere field computation

    - select B or H

    Paramters:
    ----------
    - bh: boolean, True=B, False=H
    - mag: ndarray shape (n,3), magnetization vector in units of mT
    - dia: ndarray shape (n,), diameter in units of [mm]
    - pos_obs: ndarray shape (n,3), position of observer in units of mm

    Returns:
    --------
    B/H-field (ndarray Nx3): magnetic field vectors at pos_obs in units of mT / kA/m
    """

    B = magnet_sphere_Bfield(mag, dia, pos_obs)
    if bh:
        return B

    # adjust and return H
    # as a result of bad code layout the inside-outside check is repeated here.
    # this should be done in the function magnet_sphere_Bfield() which is kept
    # simple because it lies at the lib interface.

    x, y, z = np.copy(pos_obs.T)
    r = np.sqrt(x**2+y**2+z**2)   # faster than np.linalg.norm
    mask_in = (r<dia/2)
    B[mask_in] = -mag[mask_in]/3
    H = B*10/4/np.pi
    return H


def magnet_sphere_Bfield(
    magnetization: np.ndarray,
    diameter: np.ndarray,
    observer: np.ndarray,
    )->np.ndarray:
    """
    The B-field of a homogeneously magnetized spherical magnet corresponds to a dipole
    field on the outside and is 2/3*mag in the inside (see e.g. "Theoretical Physics, Bertelmann")

    Parameters:
    -----------
    magnetization: ndarray, shape (n,3)
        Homogeneous magnetization vector in units of [mT].

    diameter: ndarray, shape (n,3)
        Sphere diameter in units of [mm].

    observer: ndarray, shape (n,3)
        Position of observers in units of [mm].

    Returns
    -------
    B-field: ndarray, shape (n,3)
        B-field of magnet in cartesian coordinates (Bx, By, Bz) in units of [mT].

    Examples
    --------
    Compute the field of two different spherical magnets at position (1,1,1),
    inside and outside.

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> dia = np.array([1,5])
    >>> obs = np.array([(1,1,1), (1,1,1)])
    >>> mag = np.array([(1,2,3), (0,0,3)])
    >>> B = magpy.lib.magnet_sphere_Bfield(mag, dia, obs)
    >>> print(B)
    [[0.04009377 0.03207501 0.02405626]
     [0.         0.         2.        ]]
    """
    x, y, z = np.copy(observer.T)
    r = np.sqrt(x**2+y**2+z**2)   # faster than np.linalg.norm

    # inside field & allocate
    B = magnetization*2/3

    # overwrite outside field entries
    mask_out = (r>=diameter/2)

    mag1 = magnetization[mask_out]
    obs1 = observer[mask_out]
    r1 = r[mask_out]
    dim1 = diameter[mask_out]

    field_out = (3*(np.sum(mag1*obs1,axis=1)*obs1.T)/r1**5 - mag1.T/r1**3)*dim1**3/24
    B[mask_out] = field_out.T

    return B

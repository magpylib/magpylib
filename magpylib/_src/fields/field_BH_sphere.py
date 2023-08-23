"""
Implementations of analytical expressions for the magnetic field of homogeneously
magnetized Spheres. Computation details in function docstrings.
"""
import numpy as np

from magpylib._src.input_checks import check_field_input


# CORE
def magnet_sphere_field(
    field: str,
    observers: np.ndarray,
    magnetization: np.ndarray,
    diameter: np.ndarray,
) -> np.ndarray:
    """Magnetic field of a homogeneously magnetized sphere.

    The center of the sphere lies in the origin of the coordinate system.

    Parameters
    ----------
    field: str, default=`'B'`
        If `field='B'` return B-field in units of mT, if `field='H'` return H-field
        in units of kA/m.

    observers: ndarray, shape (n,3)
        Observer positions (x,y,z) in Cartesian coordinates in units of mm.

    magnetization: ndarray, shape (n,3)
        Homogeneous magnetization vector in units of mT.

    diameter: ndarray, shape (n,3)
        Sphere diameter in units of mm.

    Returns
    -------
    B-field or H-field: ndarray, shape (n,3)
        B/H-field of magnet in Cartesian coordinates (Bx, By, Bz) in units of mT/(kA/m).

    Examples
    --------
    Compute the field of two different spherical magnets at position (1,1,1),
    inside and outside.

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> dia = np.array([1,5])
    >>> obs = np.array([(1,1,1), (1,1,1)])
    >>> mag = np.array([(1,2,3), (0,0,3)])
    >>> B = magpy.core.magnet_sphere_field('B', obs, mag, dia)
    >>> print(B)
    [[0.04009377 0.03207501 0.02405626]
     [0.         0.         2.        ]]

    Notes
    -----
    The field corresponds to a dipole field on the outside and is 2/3*mag
    in the inside (see e.g. "Theoretical Physics, Bertelmann").
    """

    bh = check_field_input(field, "magnet_sphere_field()")

    # all special cases r0=0 and mag=0 automatically covered

    x, y, z = np.copy(observers.T)
    r = np.sqrt(x**2 + y**2 + z**2)  # faster than np.linalg.norm
    r0 = abs(diameter) / 2

    # inside field & allocate
    B = magnetization * 2 / 3

    # overwrite outside field entries
    mask_out = r >= r0

    mag1 = magnetization[mask_out]
    obs1 = observers[mask_out]
    r1 = r[mask_out]
    r01 = r0[mask_out]

    field_out = (
        (3 * (np.sum(mag1 * obs1, axis=1) * obs1.T) / r1**5 - mag1.T / r1**3)
        * r01**3
        / 3
    )
    B[mask_out] = field_out.T

    if bh:
        return B

    # adjust and return H
    B[~mask_out] = -magnetization[~mask_out] / 3
    H = B * 10 / 4 / np.pi
    return H

"""
Implementations of analytical expressions for the magnetic field of homogeneously
magnetized Spheres. Computation details in function docstrings.
"""
import numpy as np

from magpylib._src.input_checks import check_field_input
from magpylib._src.utility import MU0


# CORE
def magnet_sphere_field(
    field: str,
    observers: np.ndarray,
    diameters: np.ndarray,
    polarizations: np.ndarray,
) -> np.ndarray:
    """Magnetic field of a homogeneously magnetized sphere.

    The center of the sphere lies in the origin of the coordinate system.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    field: str, default=`'B'`
        If `field='B'` return B-field in units of T, if `field='H'` return H-field
        in units of A/m.

    observers: ndarray, shape (n,3)
        Observer positions (x,y,z) in Cartesian coordinates in units of m.

    diameters: ndarray, shape (n,)
        Sphere diameters in units of m.

    polarizations: ndarray, shape (n,3)
        Magnetic polarization vectors in units of T.

    Returns
    -------
    B-field or H-field: ndarray, shape (n,3)
        B- or H-field of magnet in Cartesian coordinates in units of T or A/m.

    Examples
    --------
    Compute the field of two different spherical magnets at position (1,1,1),
    inside and outside.

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> B = magpy.core.magnet_sphere_field(
    >>>     field='B',
    >>>     observers=np.array([(1,1,1), (1,1,1)]),
    >>>     diameters=np.array([1,5]),
    >>>     polarizations=np.array([(1,2,3), (0,0,3)]),
    >>> )
    >>> print(B)
    [[0.04009377 0.03207501 0.02405626]
     [0.         0.         2.        ]]

    Notes
    -----
    Advanced unit use: The input unit of magnetization and polarization
    gives the output unit of H and B. All results are independent of the
    length input units. One must be careful, however, to use consistently
    the same length unit throughout a script.

    The field corresponds to a dipole field on the outside and is 2/3*mag
    in the inside (see e.g. "Theoretical Physics, Bertelmann").
    """

    bh = check_field_input(field, "magnet_sphere_field()")

    # all special cases r0=0 and mag=0 automatically covered

    x, y, z = np.copy(observers.T)
    r = np.sqrt(x**2 + y**2 + z**2)  # faster than np.linalg.norm
    r0 = abs(diameters) / 2

    # inside field & allocate
    B = polarizations * 2 / 3

    # overwrite outside field entries
    mask_out = r >= r0

    mag1 = polarizations[mask_out]
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
    B[~mask_out] = -polarizations[~mask_out] / 3
    H = B / MU0
    return H

"""
Implementations of analytical expressions for the magnetic field of homogeneously
magnetized Spheres. Computation details in function docstrings.
"""
import numpy as np

from magpylib._src.input_checks import check_field_input
from magpylib._src.utility import MU0


def magnet_sphere_Bfield(
    *,
    observers: np.ndarray,
    diameters: np.ndarray,
    polarizations: np.ndarray,
) -> np.ndarray:
    """Magnetic field of homogeneously magnetized spheres.

    The center of the sphere lies in the origin of the coordinate system.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    field: str, default=`'B'`
        If `field='B'` return B-field in units of T, if `field='H'` return H-field
        in units of A/m.

    observers: ndarray, shape (n,3)
        Observer positions (x,y,z) in Cartesian coordinates in units of m.

    diameter: ndarray, shape (n,)
        Sphere diameter in units of m.

    polarization: ndarray, shape (n,3)
        Magnetic polarization vectors in units of T.

    in_out: {'auto', 'inside', 'outside'}
        Specify the location of the observers relative to the magnet body, affecting the calculation
        of the magnetic field. The options are:
        - 'auto': The location (inside or outside the cuboid) is determined automatically for each
          observer.
        - 'inside': All observers are considered to be inside the cuboid; use this for performance
          optimization if applicable.
        - 'outside': All observers are considered to be outside the cuboid; use this for performance
          optimization if applicable.
        Choosing 'auto' is fail-safe but may be computationally intensive if the mix of observer
        locations is unknown.

    Returns
    -------
    B-field or H-field: ndarray, shape (n,3)
        B- or H-field of source in Cartesian coordinates in units of T or A/m.

    Examples
    --------
    Compute the field of two different spherical magnets at position (1,1,1),
    inside and outside.

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> B = magpy.core.magnet_sphere_field(
    ...     field='B',
    ...     observers=np.array([(1,1,1), (1,1,1)]),
    ...     diameter=np.array([1,5]),
    ...     polarization=np.array([(1,2,3), (0,0,3)]),
    ... )
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
    return BHJM_magnet_sphere(
        field="B",
        observers=observers,
        diameter=diameters,
        polarization=polarizations,
    )


def BHJM_magnet_sphere(
    *,
    field: str,
    observers: np.ndarray,
    diameter: np.ndarray,
    polarization: np.ndarray,
    in_out="auto",
) -> np.ndarray:
    """
    magnet sphere field, cannot be moved to core function, because
    core computation requires inside-outside check, but BHJM translation also.
    Would require 2 checks, or forwarding the masks ... both not ideal
    """
    check_field_input(field)

    x, y, z = np.copy(observers.T)
    r = np.sqrt(x**2 + y**2 + z**2)  # faster than np.linalg.norm
    r_sphere = abs(diameter) / 2

    # inside field & allocate
    BHJM = polarization.astype(float) * 2 / 3
    mask_outside = r > r_sphere

    if field == "J":
        BHJM[mask_outside] = 0.0
        return BHJM

    if field == "M":
        BHJM[mask_outside] = 0.0
        return BHJM / MU0

    BHJM[mask_outside] = (
        (
            3
            * (
                np.sum(polarization[mask_outside] * observers[mask_outside], axis=1)
                * observers[mask_outside].T
            )
            / r[mask_outside] ** 5
            - polarization[mask_outside].T / r[mask_outside] ** 3
        )
        * observers[mask_outside] ** 3
        / 3
    ).T

    if field == "B":
        return BHJM

    if field == "H":
        BHJM[~mask_outside] -= polarization[~mask_outside]
        return BHJM / MU0

    raise ValueError(  # pragma: no cover
        "`output_field_type` must be one of ('B', 'H', 'M', 'J'), " f"got {field!r}"
    )

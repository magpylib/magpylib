"""Homogeneously magnetized sphere field implementation."""

import numpy as np
from scipy.constants import mu_0 as MU0

from magpylib._src.input_checks import check_field_input


def magnet_sphere_Bfield(
    observers: np.ndarray,
    diameters: np.ndarray,
    polarizations: np.ndarray,
) -> np.ndarray:
    """B-field of i homogeneously magnetized spheres in Cartesian coordinates.

    The center of the spheres lie in the origin of the coordinate system. The output
    is proportional to the polarization magnitudes, and independent of the length
    units chosen for observers and diameters.

    Parameters
    ----------
    observers : ndarray, shape (i, 3)
        Observer positions (x, y, z) in Cartesian coordinates.
    diameters : ndarray, shape (i,)
        Sphere diameters.
    polarizations : ndarray, shape (i, 3)
        Magnetic polarization vectors.

    Returns
    -------
    ndarray, shape (i, 3)
        B-field in units of ``polarization`` input at the observer positions.

    Notes
    -----
    Outside the sphere, the field corresponds to a dipole field. Inside, it is
    2/3 of the polarization (see e.g., "Theoretical Physics", Bertelmann).

    Examples
    --------
    >>> import numpy as np
    >>> import magpylib as magpy
    >>> B = magpy.core.magnet_sphere_Bfield(
    ...     observers=np.array([(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]),
    ...     diameters=np.array([1.0, 2.0]),
    ...     polarizations=np.array([(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]),
    ... )
    >>> with np.printoptions(precision=3):
    ...     print(B)
    [[1.187e-18 8.019e-03 8.019e-03]
     [8.019e-03 8.019e-03 1.604e-02]]
    """
    return _BHJM_magnet_sphere(
        field="B",
        observers=observers,
        diameter=diameters,
        polarization=polarizations,
    )


def _BHJM_magnet_sphere(
    field: str,
    observers: np.ndarray,
    diameter: np.ndarray,
    polarization: np.ndarray,
) -> np.ndarray:
    """
    - compute sphere field and translate to BHJM
    - magnet sphere field, cannot be moved to a core function, because
    core computation requires inside-outside check, but BHJM translation also.
    Would require 2 checks, or forwarding the masks ... both not ideal
    """
    check_field_input(field)

    x, y, z = np.copy(observers.T)
    r = np.sqrt(x**2 + y**2 + z**2)  # faster than np.linalg.norm
    r_sphere = abs(diameter) / 2

    # inside field & allocate
    BHJM = polarization.astype(float)
    out = r > r_sphere

    if field == "J":
        BHJM[out] = 0.0
        return BHJM

    if field == "M":
        BHJM[out] = 0.0
        return BHJM / MU0

    BHJM *= 2 / 3

    BHJM[out] = (
        (
            3 * np.sum(polarization[out] * observers[out], axis=1) * observers[out].T
            - polarization[out].T * r[out] ** 2
        )
        / r[out] ** 5
        * r_sphere[out] ** 3
        / 3
    ).T

    if field == "B":
        return BHJM

    if field == "H":
        BHJM[~out] -= polarization[~out]
        return BHJM / MU0

    msg = (
        "Input output_field_type must be one of {'B', 'H', 'M', 'J'}; "
        f"instead received {field!r}."
    )
    raise ValueError(msg)  # pragma: no cover

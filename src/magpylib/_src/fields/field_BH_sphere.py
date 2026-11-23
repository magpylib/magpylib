"""Homogeneously magnetized sphere field implementation."""

import numpy as np
from array_api_compat import array_namespace
from scipy.constants import mu_0 as MU0

from magpylib._src.array_api_utils import jax_padded, xp_promote
from magpylib._src.input_checks import check_field_input


@jax_padded()
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

    xp = array_namespace(observers, diameter, polarization)
    observers, diameter, polarization = xp_promote(
        observers, diameter, polarization, xp=xp
    )

    r = xp.linalg.vector_norm(observers, axis=-1, keepdims=True)
    r_sphere = xp.abs(diameter)[:, xp.newaxis] / 2
    mask_out = r > r_sphere

    if field == "J":
        return xp.where(mask_out, 0.0, polarization)

    if field == "M":
        return xp.where(mask_out, 0.0, polarization) / MU0

    # dipole field outside, 2/3 of the polarization inside
    r = xp.where(mask_out, r, 1.0)  # avoid 0/0 in the center
    dotprod = xp.sum(polarization * observers, axis=-1, keepdims=True)
    B_out = (3 * dotprod * observers - polarization * r**2) / r**5 * r_sphere**3 / 3
    BHJM = xp.where(mask_out, B_out, polarization * (2 / 3))

    if field == "B":
        return BHJM

    if field == "H":
        return (BHJM - xp.where(mask_out, 0.0, polarization)) / MU0

    msg = (
        "Input output_field_type must be one of {'B', 'H', 'M', 'J'}; "
        f"instead received {field!r}."
    )
    raise ValueError(msg)  # pragma: no cover

"""Dipole field implementation."""

import numpy as np
from scipy.constants import mu_0 as MU0

from magpylib._src.array_api_utils import (
    array_namespace,
    dot,
    jax_padded,
    some,
    xp_promote,
)
from magpylib._src.input_checks import check_field_input


@jax_padded()
def dipole_Hfield(
    observers: np.ndarray,
    moments: np.ndarray,
) -> np.ndarray:
    """Magnetic field of i dipole moments in Cartesian coordinates.

    The dipole moment lies in the origin of the coordinate system.
    The output is proportional to the moment input, and is independent
    of length units used for observers (and moment) input considering
    that the moment is proportional to [L]**2.
    Returns np.inf for all non-zero moment components in the origin.

    Parameters
    ----------
    observers : ndarray, shape (i, 3)
        Observer positions ``(x, y, z)`` in Cartesian coordinates.
    moments : ndarray, shape (i, 3)
        Dipole moment vectors at the origin.

    Returns
    -------
    ndarray, shape (i, 3)
        H-field in units of ``moments`` input in Cartesian coordinates.

    Notes
    -----
    - The moment of a magnet is given by its volume times magnetization.
    - At ``r = 0``, non-zero moment components yield ``np.inf`` in the
      corresponding field components.

    Examples
    --------
    >>> import numpy as np
    >>> import magpylib as magpy
    >>> H = magpy.core.dipole_Hfield(
    ...    observers=np.array([(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]),
    ...    moments=np.array([(1e5, 0.0, 0.0), (0.0, 0.0, 1e5)]),
    ... )
    >>> with np.printoptions(precision=3):
    ...     print(H)  # doctest: +ELLIPSIS
    [[... 1.531e+03 1.531e+03]
     [1.914e+02 1.914e+02 ...]]
    """

    xp = array_namespace(observers, moments)
    observers, moments = xp_promote(observers, moments, xp=xp)

    r = xp.linalg.vector_norm(observers, axis=-1, keepdims=True)

    # with the unit vector to the observer, as a power of r above the third
    # underflows in float32 already at nanometres
    mask_origin = r == 0
    at_origin = some(mask_origin)
    if at_origin:
        r = xp.where(mask_origin, 1.0, r)  # avoid 0/0
    unit = observers / r
    dotprod = dot(moments, unit)[:, xp.newaxis]
    H = (3 * dotprod * unit - moments) / (r * r * r) / (4 * xp.pi)
    if not at_origin:
        return H

    # at r=0 return inf in the direction of all non-zero moment components
    zero = xp.zeros_like(moments)  # where of the array API needs an array
    H_origin = xp.where(moments > 0, xp.inf, xp.where(moments < 0, -xp.inf, zero))
    return xp.where(mask_origin, H_origin, H)


def _BHJM_dipole(
    field: str,
    observers: np.ndarray,
    moment: np.ndarray,
) -> np.ndarray:
    """
    - translate dipole field to BHJM
    """
    check_field_input(field)

    if field in "MJ":
        xp = array_namespace(observers, moment)
        return xp.zeros_like(xp_promote(observers, xp=xp))

    BHJM = dipole_Hfield(
        observers=observers,
        moments=moment,
    )

    if field == "H":
        return BHJM

    if field == "B":
        return BHJM * MU0

    msg = (
        "Input output_field_type must be one of ('B', 'H', 'M', 'J'); "
        f"instead received {field!r}."
    )
    raise ValueError(msg)  # pragma: no cover

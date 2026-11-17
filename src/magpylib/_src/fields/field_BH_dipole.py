"""Dipole field implementation."""

import numpy as np
from array_api_compat import array_namespace
from scipy.constants import mu_0 as MU0

from magpylib._src.array_api_utils import xp_promote
from magpylib._src.input_checks import check_field_input


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
    ...     print(H)
    [[2.895e-13 1.531e+03 1.531e+03]
     [1.914e+02 1.914e+02 3.619e-14]]
    """

    xp = array_namespace(observers, moments)
    observers, moments = xp_promote(observers, moments, xp=xp)

    r = xp.linalg.vector_norm(observers, axis=-1, keepdims=True)

    # at r=0 return inf in the direction of all non-zero moment components
    mask_origin = r == 0
    H_origin = xp.where(moments > 0, xp.inf, xp.where(moments < 0, -xp.inf, 0.0))

    r = xp.where(mask_origin, 1.0, r)  # avoid 0/0
    dotprod = xp.sum(moments * observers, axis=-1, keepdims=True)
    H = (3 * dotprod * observers / r**5 - moments / r**3) / 4 / xp.pi

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

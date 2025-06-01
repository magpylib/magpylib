"""
Core implementation of dipole field
"""

from __future__ import annotations

import array_api_extra as xpx
import numpy as np
from array_api_compat import array_namespace
from scipy.constants import mu_0 as MU0

from magpylib._src.array_api_utils import xp_promote
from magpylib._src.input_checks import check_field_input


# CORE
def dipole_Hfield(
    observers: np.ndarray,
    moments: np.ndarray,
) -> np.ndarray:
    """Magnetic field of a dipole moments.

    The dipole moment lies in the origin of the coordinate system.
    The output is proportional to the moment input, and is independent
    of length units used for observers (and moment) input considering
    that the moment is proportional to [L]**2.
    Returns np.inf for all non-zero moment components in the origin.

    Parameters
    ----------
    observers: ndarray, shape (n,3)
        Observer positions (x,y,z) in Cartesian coordinates.

    moments: ndarray, shape (n,3)
        Dipole moment vector.

    Returns
    -------
    H-field: ndarray, shape (n,3)
        H-field of Dipole in Cartesian coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> import magpylib as magpy
    >>> H = magpy.core.dipole_Hfield(
    ...    observers=np.array([(1,1,1), (2,2,2)]),
    ...    moments=np.array([(1e5,0,0), (0,0,1e5)]),
    ... )
    >>> with np.printoptions(precision=3):
    ...     print(H)
    [[2.895e-13 1.531e+03 1.531e+03]
     [1.914e+02 1.914e+02 3.619e-14]]

    Notes
    -----
    The moment of a magnet is given by its volume*magnetization.
    """
    xp = array_namespace(observers, moments)
    observers, moments = xp_promote(observers, moments, force_floating=True, xp=xp)
    r = xp.linalg.vector_norm(observers, axis=-1, keepdims=True)

    # 0/0 produces invalid warn and results in np.nan
    # x/0 produces divide warn and results in np.inf
    mask_r = r == 0.0
    r = xpx.at(r)[mask_r].set(1.0)
    dotprod = xp.vecdot(moments, observers)[:, xp.newaxis]

    def B(dotprod, observers, moments, r):
        A = xp.divide(3 * dotprod * observers, r**5)
        B = xp.divide(moments, r**3)
        return xp.divide((A - B), 4.0 * xp.pi)

    H = xpx.apply_where(~mask_r, (dotprod, observers, moments, r), B, fill_value=xp.inf)
    # H = xpx.at(H)[xp.broadcast_to(mask_r, H.shape)].set(xp.inf)

    return H


def BHJM_dipole(
    field: str,
    observers: np.ndarray,
    moment: np.ndarray,
) -> np.ndarray:
    """
    - translate dipole field to BHJM
    """
    check_field_input(field)

    if field in "MJ":
        return np.zeros_like(observers, dtype=float)

    BHJM = dipole_Hfield(
        observers=observers,
        moments=moment,
    )

    if field == "H":
        return BHJM

    if field == "B":
        return BHJM * MU0

    msg = f"`output_field_type` must be one of ('B', 'H', 'M', 'J'), got {field!r}"
    raise ValueError(msg)  # pragma: no cover

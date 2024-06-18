"""
Core implementation of dipole field
"""

import numpy as np
from scipy.constants import mu_0 as MU0

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

    Notes
    -----
    The moment of a magnet is given by its volume*magnetization.
    """

    x, y, z = observers.T
    r = np.sqrt(x**2 + y**2 + z**2)  # faster than np.linalg.norm
    with np.errstate(divide="ignore", invalid="ignore"):
        # 0/0 produces invalid warn and results in np.nan
        # x/0 produces divide warn and results in np.inf
        H = (
            (
                3 * np.sum(moments * observers, axis=1) * observers.T / r**5
                - moments.T / r**3
            ).T
            / 4
            / np.pi
        )

    # when r=0 return np.inf in all non-zero moments directions
    mask1 = r == 0
    if np.any(mask1):
        with np.errstate(divide="ignore", invalid="ignore"):
            H[mask1] = moments[mask1] / 0.0
            np.nan_to_num(H, copy=False, posinf=np.inf, neginf=-np.inf)

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

    raise ValueError(  # pragma: no cover
        "`output_field_type` must be one of ('B', 'H', 'M', 'J'), " f"got {field!r}"
    )

"""
Core implementation of dipole field
"""
import numpy as np

from magpylib._src.input_checks import check_field_input
from magpylib._src.utility import MU0


# CORE
def dipole_Hfield(
    *,
    observers: np.ndarray,
    moments: np.ndarray,
) -> np.ndarray:
    """Magnetic field of a dipole moments.

    The dipole moment lies in the origin of the coordinate system. SI units are used
    for all inputs and outputs. Returns np.inf for all non-zero moment components
    in the origin.

    Parameters
    ----------
    observers: ndarray, shape (n,3)
        Observer positions (x,y,z) in Cartesian coordinates in units of m.

    moments: ndarray, shape (n,3)
        Dipole moment vector in units of A·m².

    Returns
    -------
    H-field: ndarray, shape (n,3)
        H-field of dipole in Cartesian coordinates in units of A/m.

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
            np.nan_to_num(H, copy=False, posinf=np.inf, neginf=np.NINF)

    return H


def BHJM_dipole(
    *,
    field: str,
    observers: np.ndarray,
    moment: np.ndarray,
) -> np.ndarray:
    check_field_input(field)

    if field in "MJ":
        return np.zeros_like(observers, dtype=float)

    H = dipole_Hfield(
        observers=observers,
        moments=moment,
    )

    if field == "H":
        return H

    if field == "B":
        return H * MU0

    raise ValueError(  # pragma: no cover
        "`output_field_type` must be one of ('B', 'H', 'M', 'J'), " f"got {field!r}"
    )

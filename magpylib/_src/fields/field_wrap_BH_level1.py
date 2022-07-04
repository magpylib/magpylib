from typing import Callable

import numpy as np


def getBH_level1(
    *,
    field_func: Callable,
    field: str,
    position: np.ndarray,
    orientation: np.ndarray,
    observers: np.ndarray,
    **kwargs: dict,
) -> np.ndarray:
    """Vectorized field computation

    - applies spatial transformations global CS <-> source CS
    - selects the correct Bfield_XXX function from input

    Args
    ----
    kwargs: dict of shape (N,x) input vectors that describes the computation.

    Returns
    -------
    field: ndarray, shape (N,3)

    """

    # transform obs_pos into source CS
    pos_rel_rot = orientation.apply(observers - position, inverse=True)

    # compute field
    BH = field_func(field=field, observers=pos_rel_rot, **kwargs)

    # transform field back into global CS
    BH = orientation.apply(BH)

    return BH

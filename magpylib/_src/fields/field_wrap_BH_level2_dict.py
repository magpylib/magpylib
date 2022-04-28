""" getBHv wrapper codes"""
import numpy as np
from scipy.spatial.transform import Rotation as R

from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib._src.fields.field_wrap_BH_level1 import getBH_level1
from magpylib._src.utility import LIBRARY_BH_DICT_SOURCE_STRINGS


PARAM_TILE_DIMS = {
    "observers": 2,
    "position": 2,
    "orientation": 2,
    "magnetization": 2,
    "current": 1,
    "moment": 2,
    "dimension": 2,
    "diameter": 1,
    "segment_start": 2,
    "segment_end": 2,
}


def getBH_dict_level2(
    source_type,
    observers,
    *,
    field: str,
    position=(0, 0, 0),
    orientation=R.identity(),
    squeeze=True,
    **kwargs: dict,
) -> np.ndarray:
    """Direct interface access to vectorized computation

    Parameters
    ----------
    kwargs: dict that describes the computation.

    Returns
    -------
    field: ndarray, shape (N,3), field at obs_pos in [mT] or [kA/m]

    Info
    ----
    - check inputs

    - secures input types (list/tuple -> ndarray)
    - test if mandatory inputs are there
    - sets default input variables (e.g. pos, rot) if missing
    - tiles 1D inputs vectors to correct dimension
    """
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements

    # generate dict of secured inputs for auto-tiling ---------------
    #  entries in this dict will be tested for input length, and then
    #  be automatically tiled up and stored back into kwargs for calling
    #  getBH_level1().
    #  To allow different input dimensions, the tdim argument is also given
    #  which tells the program which dimension it should tile up.

    if source_type not in LIBRARY_BH_DICT_SOURCE_STRINGS:
        raise MagpylibBadUserInput(
            f"Input parameter `sources` must be one of {LIBRARY_BH_DICT_SOURCE_STRINGS}"
            " when using the direct interface."
        )

    kwargs["observers"] = observers
    kwargs["position"] = position

    # change orientation to Rotation numpy array for tiling
    kwargs["orientation"] = orientation.as_quat()

    # evaluation vector lengths
    vec_lengths = []
    for key, val in kwargs.items():
        try:
            val = np.array(val, dtype=float)
        except TypeError as err:
            raise MagpylibBadUserInput(
                f"{key} input must be array-like.\n" f"Instead received {val}"
            ) from err
        tdim = PARAM_TILE_DIMS.get(key, 1)
        if val.ndim == tdim:
            vec_lengths.append(len(val))
        kwargs[key] = val

    if len(set(vec_lengths)) > 1:
        raise MagpylibBadUserInput(
            "Input array lengths must be 1 or of a similar length.\n"
            f"Instead received {set(vec_lengths)}"
        )
    vec_len = max(vec_lengths, default=1)

    # tile 1D inputs and replace original values in kwargs
    for key, val in kwargs.items():
        tdim = PARAM_TILE_DIMS.get(key, 1)
        if val.ndim < tdim:
            if tdim == 2:
                kwargs[key] = np.tile(val, (vec_len, 1))
            elif tdim == 1:
                kwargs[key] = np.array([val] * vec_len)
        else:
            kwargs[key] = val

    # change orientation back to Rotation object
    kwargs["orientation"] = R.from_quat(kwargs["orientation"])

    # compute and return B
    B = getBH_level1(source_type=source_type, field=field, **kwargs)

    if squeeze:
        return np.squeeze(B)
    return B

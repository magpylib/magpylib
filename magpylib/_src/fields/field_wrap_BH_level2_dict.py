""" getBHv wrapper codes"""
import numpy as np
from scipy.spatial.transform import Rotation as R

from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib._src.fields.field_wrap_BH_level1 import getBH_level1
from magpylib._src.utility import LIBRARY_BH_DICT_SOURCE_STRINGS


def getBH_dict_level2(**kwargs: dict) -> np.ndarray:
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
    tile_params = {}

    # mandatory general inputs ------------------
    try:
        src_type = kwargs["source_type"]
        if src_type not in LIBRARY_BH_DICT_SOURCE_STRINGS:
            raise MagpylibBadUserInput(
                f"Input parameter `sources` must be one of {LIBRARY_BH_DICT_SOURCE_STRINGS}"
                " when using the direct interface."
            )

        poso = np.array(kwargs["observers"], dtype=float)
        tile_params["observers"] = (poso, 2)  # <-- (input,tdim)

        # optional general inputs -------------------
        # if no input set pos=0
        pos = np.array(kwargs.get("position", (0, 0, 0)), dtype=float)
        tile_params["position"] = (pos, 2)
        # if no input set rot=unit
        rot = kwargs.get("orientation", R.from_quat((0, 0, 0, 1)))
        tile_params["orientation"] = (rot.as_quat(), 2)
        # if no input set squeeze=True
        squeeze = kwargs.get("squeeze", True)

        # mandatory class specific inputs -----------
        if src_type == "Cuboid":
            mag = np.array(kwargs["magnetization"], dtype=float)
            tile_params["magnetization"] = (mag, 2)
            dim = np.array(kwargs["dimension"], dtype=float)
            tile_params["dimension"] = (dim, 2)

        elif src_type == "Cylinder":
            mag = np.array(kwargs["magnetization"], dtype=float)
            tile_params["magnetization"] = (mag, 2)
            dim = np.array(kwargs["dimension"], dtype=float)
            tile_params["dimension"] = (dim, 2)

        elif src_type == "CylinderSegment":
            mag = np.array(kwargs["magnetization"], dtype=float)
            tile_params["magnetization"] = (mag, 2)
            dim = np.array(kwargs["dimension"], dtype=float)
            tile_params["dimension"] = (dim, 2)

        elif src_type == "Sphere":
            mag = np.array(kwargs["magnetization"], dtype=float)
            tile_params["magnetization"] = (mag, 2)
            dia = np.array(kwargs["diameter"], dtype=float)
            tile_params["diameter"] = (dia, 1)

        elif src_type == "Dipole":
            moment = np.array(kwargs["moment"], dtype=float)
            tile_params["moment"] = (moment, 2)

        elif src_type == "Loop":
            current = np.array(kwargs["current"], dtype=float)
            tile_params["current"] = (current, 1)
            dia = np.array(kwargs["diameter"], dtype=float)
            tile_params["diameter"] = (dia, 1)

        elif src_type == "Line":
            current = np.array(kwargs["current"], dtype=float)
            tile_params["current"] = (current, 1)
            pos_start = np.array(kwargs["segment_start"], dtype=float)
            tile_params["segment_start"] = (pos_start, 2)
            pos_end = np.array(kwargs["segment_end"], dtype=float)
            tile_params["segment_end"] = (pos_end, 2)

    except KeyError as kerr:
        raise MagpylibBadUserInput(f"Missing input keys: {str(kerr)}") from kerr
    except TypeError as terr:
        raise MagpylibBadUserInput(
            "Bad user input type. When sources argument is a string,"
            " all other inputs must be scalar or array_like."
        ) from terr

    # auto tile 1D parameters ---------------------------------------

    # evaluation vector length
    ns = [len(val) for val, tdim in tile_params.values() if val.ndim == tdim]
    if len(set(ns)) > 1:
        raise MagpylibBadUserInput(
            "Input array lengths must be 1 or of a similarlength.\n"
            f"Instead received {str(set(ns))}"
        )
    n = max(ns, default=1)

    # tile 1D inputs and replace original values in kwargs
    for key, (val, tdim) in tile_params.items():
        if val.ndim < tdim:
            if tdim == 2:
                kwargs[key] = np.tile(val, (n, 1))
            elif tdim == 1:
                kwargs[key] = np.array([val] * n)
        else:
            kwargs[key] = val

    # change rot to Rotation object
    kwargs["orientation"] = R.from_quat(kwargs["orientation"])

    # compute and return B
    B = getBH_level1(**kwargs)

    if squeeze:
        return np.squeeze(B)
    return B

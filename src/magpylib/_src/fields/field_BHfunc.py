"""Magnetic field computation functional interface"""

import numbers
from collections.abc import Callable
from magpylib._src.utility import get_registered_sources
from magpylib._src.utility import has_parameter
from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib._src.fields.field_BH_circle import _BHJM_circle
from magpylib._src.fields.field_BH_polyline import _BHJM_current_polyline

import numpy as np
from scipy.spatial.transform import Rotation as R

def current_circle_field(
       field,
       observers,
       diameters,
       currents,
       positions=(0,0,0),
       orientations=None,
       squeeze=True,
):
    """Return B- or H-field of circular current loop for ``i`` given instances.

    When ``position=(0,0,0)`` and ``orientation=None`` the loop lies in the
    ``z=0`` plane with the coordinate origin at its center. A positive current
    flows in mathematically positive direction (counter-clockwise).

    Parameters
    ----------
    field : {'B', 'H'}
        Select field that should be returned.
    observers : array-like, shape (3,) or (i, 3)
        Points where the field is evaluated in units (m).
    positions : array-like, shape (3,) or (i, 3), default (0, 0, 0)
        Positions of cuboid centroids in units (m).
    orientations : None | Rotation, default None
        Orientation of the sources. If None, unit rotation is applied.
    diameters : float | array-like, shape (i,)
        Current loop diameters in units (m).
    currents : float | array-like, shape (i,)
        Electric currents in units (A).
    squeeze : bool, default True
        If ``True`` squeeze singleton axes (when ``i=1``).
    
    Returns
    -------
    ndarray, shape (3,) or (i, 3)
        Magnetic field values at the observer locations.
    """
    params = {
        "observers": observers,
        "diameter": diameters,
        "current": currents,
        "position": positions,
        "orientation": orientations,
    }
    return _getBH_func(_BHJM_circle, field, params, squeeze)


def current_polyline_field(
       field,
       observers,
       segments_start,
       segments_end,
       currents,
       positions=(0,0,0),
       orientations=None,
       squeeze=True,
):
    """Return B- or H-field of straight current segments for ``i`` instances.

    When ``position=(0,0,0)`` and ``orientation=None`` local and
    global coordinates conincide. The current flows from vertex
    start to end positions. The field is set to (0,0,0) on a
    line segment.    

    Parameters
    ----------
    field : {'B', 'H'}
        Select field that should be returned.
    observers : array-like, shape (3,) or (i, 3)
        Points where the field is evaluated in units (m).
    positions : array-like, shape (3,) or (i, 3), default (0, 0, 0)
        Positions of cuboid centroids in units (m).
    orientations : None | Rotation, default None
        Orientation of the sources. If None, unit rotation is applied.
    diameters : float | array-like, shape (i,)
        Current loop diameters in units (m).
    currents : float | array-like, shape (i,)
        Electric currents in units (A).
    squeeze : bool, default True
        If ``True`` squeeze singleton axes (when ``i=1``).
    
    Returns
    -------
    ndarray, shape (3,) or (i, 3)
        Magnetic field values at the observer locations.
    """
    params = {
        "observers": observers,
        "segment_start": segments_start,
        "segment_end": segments_end,
        "current": currents,
        "position": positions,
        "orientation": orientations,
    }
    return _getBH_func(_BHJM_current_polyline, field, params, squeeze)


DIM = {
    "position": 2,
    "orientation": 2,
    "observers": 2,
    "diameter": 1,
    "current": 1,
    "segment_start": 2,
    "segment_end": 2
}
SHAPE = {
    "position": (3,),
    "orientation": (4,),
    "observers": (3,),
    "diameter": (),
    "current": (),
    "segment_start": (3,),
    "segment_end": (3,)
}

def _getBH_func(field_func, field, params, squeeze):
    """Functional interface for magnetic field computation
    
    field_func: callable
    field: {"B", "H"}
    params: dict
        contains all kwargs required for field_func() call
    """
    # remove squeeze from params

    # Check orientation input
    if params['orientation'] is None:
        params['orientation'] = R.identity()
    if not isinstance(params['orientation'], R):
        msg = "Input orientation must be a scipy Rotation instance or None."
        raise TypeError(msg)

    # Transform Rotation to Quat
    params['orientation'] = params['orientation'].as_quat()

    # Transform all inputs to ndarray
    for key, val in params.items():
        try:
            if not isinstance(val, np.ndarray):
                params[key] = np.array(val, dtype=float)
        except ValueError as err:
            msg = f"{key} input must be array-like.\nInstead received {val}."
            raise ValueError(msg) from err

    # Tile missing ndims, Find maxlength
    nmax = 1
    for key, val in params.items():
        if val.ndim < DIM[key]:
            params[key] = np.expand_dims(val, axis=0)
        if val.ndim > DIM[key]:
            msg = f"{key} input has too many dimensions ({val.ndim})."
            raise ValueError(msg)
        # store maxlength
        n = params[key].shape[0]
        nmax = n if n > nmax else nmax

    # Check if shapes are correct, Tile to maxlength
    for key, val in params.items():
        if val.shape[1:] != SHAPE[key]:
            msg = (
                f"{key} input has incorrect shape {val.shape[1:]} in ndim>0."
                f" Expected {SHAPE[key]}."
            )
            raise ValueError(msg)
        if val.shape[0] not in (1, nmax):
            msg = (
                f"{key} input has incorrect number of instances {val.shape[0]}."
                f" Must be 1 (will be tiled up) or {nmax} (given max)."
            )
            raise ValueError(msg)
        # tile up to nmax if only 1 instance is given
        if nmax>1 and val.shape[0]==1:
            params[key] = np.tile(val, [nmax] + [1]*(val.ndim-1))

    # Transform Quat to Rotation object
    params['orientation'] = R.from_quat(params['orientation'])

    # Call to level1, squeeze, return
    field = _getBH_level1(field_func=field_func, field=field, **params)

    if squeeze:
        return np.squeeze(field)
    return field


# REMOVE IN FUTURE VERSIONS ############################################
# REMOVE IN FUTURE VERSIONS ############################################
# REMOVE IN FUTURE VERSIONS ############################################

def _getBH_dict_level2(
    source_type,
    observers,
    *,
    field: str,
    position=(0, 0, 0),
    orientation=None,
    squeeze=True,
    in_out="auto",
    **kwargs: dict,
) -> np.ndarray:
    """Functional interface access to vectorized computation

    Parameters
    ----------
    kwargs: dict that describes the computation.

    Returns
    -------
    field: ndarray, shape (N,3), field at obs_pos in tesla or A/m

    Info
    ----
    - check inputs

    - secures input types (list/tuple -> ndarray)
    - test if mandatory inputs are there
    - sets default input variables (e.g. pos, rot) if missing
    - tiles 1D inputs vectors to correct dimension
    """
    # pylint: disable=protected-access
    # pylint: disable=too-many-branches

    # generate dict of secured inputs for auto-tiling ---------------
    #  entries in this dict will be tested for input length, and then
    #  be automatically tiled up and stored back into kwargs for calling
    #  _getBH_level1().
    #  To allow different input dimensions, the ndim argument is also given
    #  which tells the program which dimension it should tile up.

    # pylint: disable=import-outside-toplevel
    if orientation is None:
        orientation = R.identity()
    try:
        source_classes = get_registered_sources()
        field_func = source_classes[source_type]._field_func
        field_func_kwargs_ndim = {"position": 2, "orientation": 2, "observers": 2}
        field_func_kwargs_ndim.update(
            source_classes[source_type]._field_func_kwargs_ndim
        )
    except KeyError as err:
        msg = (
            f"Input parameter `sources` must be one of {list(source_classes)}"
            " when using the functional interface."
        )
        raise MagpylibBadUserInput(msg) from err

    kwargs["observers"] = observers
    kwargs["position"] = position

    # change orientation to Rotation numpy array for tiling
    kwargs["orientation"] = orientation.as_quat()

    # evaluation vector lengths
    vec_lengths = {}
    ragged_seq = {}
    for key, val_item in kwargs.items():
        val = val_item
        try:
            if (
                not isinstance(val, numbers.Number)
                and not isinstance(val[0], numbers.Number)
                and any(len(o) != len(val[0]) for o in val)
            ):
                ragged_seq[key] = True
                val = np.array([np.array(v, dtype=float) for v in val], dtype="object")
            else:
                ragged_seq[key] = False
                val = np.array(val, dtype=float)
        except TypeError as err:
            msg = f"{key} input must be array-like.\nInstead received {val}"
            raise MagpylibBadUserInput(msg) from err
        expected_dim = field_func_kwargs_ndim.get(key, 1)
        if val.ndim == expected_dim or ragged_seq[key]:
            if len(val) == 1:
                val = np.squeeze(val)
            else:
                vec_lengths[key] = len(val)

        kwargs[key] = val

    if len(set(vec_lengths.values())) > 1:
        msg = (
            "Input array lengths must be 1 or of a similar length.\n"
            f"Instead received lengths {vec_lengths}"
        )
        raise MagpylibBadUserInput(msg)
    vec_len = max(vec_lengths.values(), default=1)
    # tile 1D inputs and replace original values in kwargs
    for key, val in kwargs.items():
        expected_dim = field_func_kwargs_ndim.get(key, 1)
        if val.ndim < expected_dim and not ragged_seq[key]:
            kwargs[key] = np.tile(val, (vec_len, *[1] * (expected_dim - 1)))

    # change orientation back to Rotation object
    kwargs["orientation"] = R.from_quat(kwargs["orientation"])

    # compute and return B
    B = _getBH_level1(field=field, field_func=field_func, in_out=in_out, **kwargs)

    if B is not None and squeeze:
        return np.squeeze(B)
    return B


def _getBH_level1(
    *,
    field_func: Callable,
    field: str,
    position: np.ndarray,
    orientation: np.ndarray,
    observers: np.ndarray,
    **kwargs: dict,
) -> np.ndarray:
    """
    COPY FROM field_BH TO AVOID CIRCULAR IMPORT
    
    Vectorized field computation

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

    # filter arguments
    if not has_parameter(field_func, "in_out"):  # in_out passed only to magnets
        kwargs.pop("in_out", None)

    # compute field
    BH = field_func(field=field, observers=pos_rel_rot, **kwargs)

    # transform field back into global CS
    if BH is not None:  # catch non-implemented field_func a level above
        BH = orientation.apply(BH)

    return BH

"""Magnetic field computation functional interface"""

import numbers
from collections.abc import Callable
from magpylib._src.utility import get_registered_sources
from magpylib._src.utility import has_parameter
from magpylib._src.exceptions import MagpylibBadUserInput

import numpy as np
from scipy.spatial.transform import Rotation as R


def getB_cuboid():
    """Return cuboid field for ``n`` given instances.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    observers : array-like, shape (3,) or (i, 3)
        Points where the field is evaluated in units (m).
    positions : array-like, shape (3,) or (i, 3), default (0, 0, 0)
        Positions of cuboid centroids in units (m).
    orientations : Rotation, default None
        Orientation of the sources. If None, unit rotation is applied
        and cuboids axes are parallel to coordinate axes.
    dimensions : array-like, shape (3,) or (i, 3)
        Dimensions of the cuboids in units (m).
    magnetizations : array-like, shape (3,) or (i, 3)
        Magnetization of the cuboids in units (A/m).
    polarizations : array-like, shape (3,) or (i, 3)
        Polarization of the cuboids in units (T).
    squeeze : bool, default True
        If ``True`` squeeze singleton axes (when ``i=1``).
    in_out : {'auto', 'inside', 'outside'}, default 'auto'
        Assumption about observer locations relative to magnet bodies. ``'auto'`` detects per
        observer (safest, slower). ``'inside'`` treats all inside (faster). ``'outside'`` treats
        all outside (faster).

    Examples
    --------
    Through the functional interface we can compute the same fields for the loop as:

    >>> obs = [(.01,.01,.01), (.01,.01,-.01)]
    >>> B = magpy.getB('Circle', obs, current=100, diameter=.002)
    >>> with np.printoptions(precision=3):
    ...     print(B)
    [[ 6.054e-06  6.054e-06  2.357e-08]
     [-6.054e-06 -6.054e-06  2.357e-08]]

    But also for a set of four completely different instances:

    >>> B = magpy.getB(
    ...     'Circle',
    ...     observers=((.01,.01,.01), (.01,.01,-.01), (.01,.02,.03), (.02,.02,.02)),
    ...     current=(11, 22, 33, 44),
    ...     diameter=(.001, .002, .003, .004),
    ...     position=((0,0,0), (0,0,.01), (0,0,.02), (0,0,.03)),
    ... )
    >>> with np.printoptions(precision=3):
    ...     print(B)
    [[ 1.663e-07  1.663e-07  1.617e-10]
     [-4.695e-07 -4.695e-07  4.707e-07]
     [ 7.970e-07  1.594e-06 -7.913e-07]
     [-1.374e-06 -1.374e-06 -1.366e-06]]
    """
    B = _getBH_dict_level2()
    return 0


def _getBH_func(
    source_type,
    field_func,
    observers,
    *,
    positions=(0,0,0),
    orientations=None,
    squeeze=True,
    in_out="auto",
):
    """Magnetic field computation functional interface"""
    if orientation is None:
        orientation = R.identity()
    field_func_kwargs_ndim = {"position": 2, "orientation": 2, "observers": 2}
    kwargs = {}







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

# REMOVE IN FUTURE VERSIONS ############################################
# REMOVE IN FUTURE VERSIONS ############################################
# REMOVE IN FUTURE VERSIONS ############################################
# This is here only to avoid circular imports with field_BH.
# Once _getBH_dict_level2 is not called from field_BH, this can be removed
# and can be called directly from field_BH.

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

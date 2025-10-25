"""input checks code"""

# pylint: disable=import-outside-toplevel
# pylint: disable=cyclic-import
# pylint: disable=too-many-positional-arguments
# pylint: disable=no-member

import inspect
import numbers
from importlib.util import find_spec

import numpy as np
from scipy.spatial.transform import Rotation

from magpylib import _src
from magpylib._src.defaults.defaults_classes import default_settings
from magpylib._src.defaults.defaults_utility import SUPPORTED_PLOTTING_BACKENDS
from magpylib._src.exceptions import (
    MagpylibBadUserInput,
    MagpylibInternalError,
    MagpylibMissingInput,
)
from magpylib._src.utility import format_obj_input, wrong_obj_msg

#################################################################
#################################################################
# FUNDAMENTAL CHECKS


def all_same(lst: list) -> bool:
    """test if all list entries are the same"""
    return lst[1:] == lst[:-1]


def is_array_like(inp, msg: str):
    """test if inp is array-like: type list, tuple or ndarray
    inp: test object
    msg: str, error msg
    """
    if not isinstance(inp, list | tuple | np.ndarray):
        raise MagpylibBadUserInput(msg)


def make_float_array(inp, msg: str):
    """transform inp to array with dtype=float, throw error with bad input
    inp: test object
    msg: str, error msg
    """
    try:
        inp_array = np.array(inp, dtype=float)
    except Exception as err:
        raise MagpylibBadUserInput(msg + f"{err}") from err
    return inp_array


def check_array_shape(inp: np.ndarray, dims: tuple, shape_m1: int, length=None, msg=""):
    """check if inp shape is allowed
    inp: test object
    dims: list, list of allowed dims
    shape_m1: shape of lowest level, if 'any' allow any shape
    msg: str, error msg
    """
    if inp.ndim in dims:
        if length is None:
            if inp.shape[-1] == shape_m1:
                return
            if shape_m1 == "any":
                return
        elif len(inp) == length:
            return
    raise MagpylibBadUserInput(msg)


def check_input_zoom(inp):
    """check show zoom input"""
    if not (isinstance(inp, numbers.Number) and inp >= 0):
        msg = f"Input zoom must be a positive number or zero; instead received {inp!r}."
        raise MagpylibBadUserInput(msg)


def check_input_animation(inp):
    """check show animation input"""
    msg = (
        "Input animation must be boolean or a positive number; "
        f"instead received {inp!r}."
    )
    if not isinstance(inp, numbers.Number):
        raise MagpylibBadUserInput(msg)
    if inp < 0:
        raise MagpylibBadUserInput(msg)


#################################################################
#################################################################
# SIMPLE CHECKS


def check_start_type(inp):
    """start input must be int or str"""
    if not (
        isinstance(inp, int | np.integer) or (isinstance(inp, str) and inp == "auto")
    ):
        msg = f"Input start must be an integer or 'auto'; instead received {inp!r}."
        raise MagpylibBadUserInput(msg)


def check_degree_type(inp):
    """degrees input must be bool"""
    if not isinstance(inp, bool):
        msg = f"Input degrees must be boolean; instead received {inp!r}."
        raise MagpylibBadUserInput(msg)


def check_field_input(inp):
    """check field input"""
    allowed = tuple("BHMJ")
    if not (isinstance(inp, str) and inp in allowed):
        opts = {"B", "H", "M", "J"}
        msg = f"Input field must be one of {opts}; instead received {inp!r}."
        raise MagpylibBadUserInput(msg)


def validate_field_func(val):
    """test if field function for custom source is valid
    - needs to be a callable
    - input and output shape must match
    """
    if val is None:
        return

    if not callable(val):
        msg = (
            "Input field_func must be a callable; "
            f"instead received {type(val).__name__}."
        )
        raise MagpylibBadUserInput(msg)

    fn_args = inspect.getfullargspec(val).args
    if fn_args[:2] != ["field", "observers"]:
        msg = (
            "Input field_func must have two positional arguments called field and observers; "
            f"instead received a callable where the first two arguments are: {fn_args[:2]!r}."
        )
        raise MagpylibBadUserInput(msg)

    for field in ["B", "H"]:
        out = val(field, np.array([[1, 2, 3], [4, 5, 6]]))
        if out is not None:
            if not isinstance(out, np.ndarray):
                msg = (
                    "Input field_func must be a callable that returns B- and H-field as a NumPy ndarray; "
                    f"instead it returns type {type(out).__name__} for {field}-field."
                )
                raise MagpylibBadUserInput(msg)
            if out.shape != (2, 3):
                msg = (
                    "Input field_func must be a callable that returns B- and H-field as a NumPy ndarray with shape (o, 3) "
                    "when the observers input has shape (o, 3); "
                    f"instead it returns shape {out.shape} for {field}-field for input shape (2, 3)."
                )
                raise MagpylibBadUserInput(msg)

    return


#################################################################
#################################################################
# CHECK - FORMAT


def match_shape(shape, pattern):
    """
    Return True if `shape` (tuple/list) matches `pattern` where pattern
    elements are ints (exact), None or 'any' (wildcard matching any value),
    and Ellipsis (...) matches any number (>=0) of elements.
    """
    shape = tuple(shape)
    pattern = tuple(pattern)

    def elem_matches(p, s):
        if p is None or p == "any":
            return True
        return p == s

    def helper(pi, si):
        # advance while both have elements and current pattern is not Ellipsis
        while pi < len(pattern) and si < len(shape) and pattern[pi] is not Ellipsis:
            if not elem_matches(pattern[pi], shape[si]):
                return False
            pi += 1
            si += 1

        # if we've consumed the whole pattern, shape must also be consumed
        if pi == len(pattern):
            return si == len(shape)

        # pattern[pi] is Ellipsis
        # if ellipsis is the last pattern element it matches the rest (including empty)
        if pi == len(pattern) - 1:
            return True

        # otherwise try all possible allocations of elements to the ellipsis (including zero)
        next_pi = pi + 1
        # try to align pattern[next_pi] with shape at positions si..len(shape)
        for k in range(si, len(shape) + 1):
            # quick check: ensure enough remaining shape elements for remaining non-ellipsis pattern items
            remaining_non_ellipsis = [p for p in pattern[next_pi:] if p is not Ellipsis]
            if len(shape) - k < len(remaining_non_ellipsis):
                break
            if helper(next_pi, k):
                return True
        return False

    return helper(0, 0)


def check_condition(
    inp,
    cond,
    threshold,
    name="input",
    mode="all",
):
    """Check that inp satisfies condition cond with threshold.
    cond can be one of {'eq','ne','lt','le','gt','ge'} or a callable(arr, threshold)->bool/array.
    mode: 'all' (default) requires all elements to satisfy, 'any' requires at least one.
    Returns inp unchanged on success, raises MagpylibBadUserInput on failure.
    """

    # build name for error messages
    msg_name = "Input" + (f" {name}" if name is not None else "")
    ops = {
        "eq": np.equal,
        "ne": np.not_equal,
        "lt": np.less,
        "le": np.less_equal,
        "gt": np.greater,
        "ge": np.greater_equal,
    }

    if isinstance(cond, str):
        try:
            func = ops[cond]
        except KeyError as err:
            msg = f"Unknown condition string {cond!r}."
            raise MagpylibInternalError(msg) from err
    elif callable(cond):
        func = cond
    else:
        msg = (
            f"Condition must be a string or callable; instead received {type(cond)!r}."
        )
        raise MagpylibInternalError(msg)

    try:
        arr = np.array(inp)
        res = func(arr, threshold)
    except Exception as err:
        msg = f"Failed to evaluate condition {cond!r} on input {inp!r} with threshold {threshold!r}: {err}"
        raise MagpylibInternalError(msg) from err

    if not (isinstance(res, (bool, np.bool_))):
        if mode == "all":
            ok = bool(np.all(res))
        elif mode == "any":
            ok = bool(np.any(res))
        else:
            msg = f"Mode must be 'all' or 'any'; instead received {mode!r}."
            raise MagpylibInternalError(msg)
    else:
        ok = bool(res)

    if not ok:
        msg = (
            f"Input {name} must satisfy condition {cond!r} with threshold"
            " {threshold!r} (mode={mode!r}); instead received {inp!r}."
        )
        raise MagpylibBadUserInput(msg)
    return inp


def check_format_input_numeric(
    inp,
    dtype,
    shapes=None,
    name=None,
    allow_None=False,
    reshape=None,
    value_conditions=None,
):
    """Validate numeric input and return normalized form."""
    if allow_None and inp is None:
        return None

    def check_conditions(array):
        if value_conditions is not None:
            for cond, threshold, mode in value_conditions:
                check_condition(
                    array,
                    cond,
                    threshold,
                    name=name,
                    mode=mode,
                )
        return array

    # build name for error messages
    msg_name = "Input" + (f" {name}" if name is not None else "")

    dims = []
    if shapes is None:
        shapes = (None,)
    shapes_clean = []
    for shape in shapes:
        shape_clean = None
        if shape is None:
            dims.append(0)
        elif isinstance(shape, int):
            dims.append(1)
            shape_clean = (shape,)
        elif isinstance(shape, (list, tuple)):
            shape_clean = shape
            assert all(
                (isinstance(s, int) and s >= 0) or s is None or s is Ellipsis
                for s in shape
            )
            if Ellipsis in shape:
                dims.append(None)
            else:
                dims.append(len(shape))
        else:
            # internal check
            msg = "shapes must be either None for scalar or a tuple for arrays"
            raise AssertionError(msg)
        if shape_clean is not None:
            shapes_clean.append(shape_clean)
    dims = tuple(dict.fromkeys(dims))
    shapes = tuple(shapes_clean)

    is_an_array = isinstance(inp, (list, tuple, np.ndarray))
    is_a_number = isinstance(inp, numbers.Number)

    # scalar case
    if dims == (0,) and not is_a_number:
        msg = (
            f"{msg_name} must be a scalar of type {dtype};"
            " instead received type {type(inp)}."
        )
        raise MagpylibBadUserInput(msg)

    if 0 in dims and is_a_number:
        inp = dtype(inp)
        return check_conditions(dtype(inp))

    if 0 not in dims and is_a_number:
        dims_str = " or ".join(str(d) for d in dims)
        msg = (
            f"{msg_name} must be an array of dimension {dims_str};"
            " instead received type {type(inp)}."
        )
        raise MagpylibBadUserInput(msg)

    # array-like case
    if not is_an_array:
        msg_scalar = "scalar or " if 0 in dims else ""
        msg = (
            f"{msg_name} must be {msg_scalar}array-like of type {dtype};"
            " instead received type {type(inp)!r}."
        )
        raise MagpylibBadUserInput(msg)

    try:
        array = np.array(inp, dtype=dtype)
    except Exception as err:
        msg = f"{msg_name} cannot be transformed into a numpy array. {err}"
        raise MagpylibBadUserInput(msg) from err

    if None not in dims and array.ndim not in dims:
        msg_scalar = "scalar or " if 0 in dims else ""
        dims_str = " or ".join(str(d) for d in dims if d != 0)
        msg = (
            f"{msg_name} must be {msg_scalar}array-like of dimension {dims_str};"
            " instead received an input of dimension {array.ndim}."
        )
        raise MagpylibBadUserInput(msg)

    if shapes == (None,):
        return check_conditions(array)

    shape_match = False
    for shape in shapes:
        if match_shape(array.shape, shape):
            shape_match = True
            break

    if not shape_match:
        shapes_str = " or ".join(
            str(d).replace("Ellipsis", "...").replace("None", "any") for d in shapes
        )
        msg = (
            f"{msg_name} must be of shape {shapes_str};"
            " instead received shape {array.shape}."
        )
        raise MagpylibBadUserInput(msg)

    if reshape is not None:
        assert isinstance(reshape, tuple), "reshape must be a tuple"
        array = np.reshape(inp, reshape)

    return check_conditions(array)


def check_format_input_orientation(inp, init_format=False):
    """checks orientation input returns in formatted form
    - inp must be None or Rotation object
    - transform None to unit rotation as quat (0, 0, 0, 1)
    if init_format: (for move method)
        return inp and inpQ
    else: (for init and setter)
        return inpQ in shape (-1, 4)

    This function is used for setter and init only -> shape (1, 4) and (4,) input
    creates same behavior.
    """
    # check type
    if not isinstance(inp, Rotation | type(None)):
        msg = (
            "Input orientation must be None or a SciPy Rotation object; "
            f"instead received type {type(inp).__name__}."
        )
        raise MagpylibBadUserInput(msg)
    # handle None input and compute inpQ
    if inp is None:
        inpQ = np.array((0, 0, 0, 1))
        inp = Rotation.from_quat(inpQ)
    else:
        inpQ = inp.as_quat()
    # return
    if init_format:
        return np.reshape(inpQ, (-1, 4))
    return inp, inpQ


def check_format_input_anchor(inp):
    """checks rotate anchor input and return in formatted form
    - input must be array-like or None or 0
    """
    if isinstance(inp, numbers.Number) and inp == 0:
        return np.array((0.0, 0.0, 0.0))

    return check_format_input_vector(
        inp,
        dims=(1, 2),
        shape_m1=3,
        sig_name="anchor",
        sig_type="None or 0 or array-like (list, tuple, ndarray) with shape (3,)",
        allow_None=True,
    )


def check_format_input_axis(inp):
    """check rotate_from_angax axis input and return in formatted form
    - input must be array-like or str
    - if string 'x'->(1, 0, 0), 'y'->(0, 1, 0), 'z'->(0, 0, 1)
    - convert inp to ndarray with dtype float
    - inp shape must be (3,)
    - axis must not be (0, 0, 0)
    - return as ndarray shape (3,)
    """
    if isinstance(inp, str):
        if inp == "x":
            return np.array((1, 0, 0))
        if inp == "y":
            return np.array((0, 1, 0))
        if inp == "z":
            return np.array((0, 0, 1))
        msg = (
            "Input axis must be array-like with shape (3,) or one of {'x', 'y', 'z'}; "
            f"instead received string {inp!r}."
        )
        raise MagpylibBadUserInput(msg)

    inp = check_format_input_vector(
        inp,
        dims=(1,),
        shape_m1=3,
        sig_name="axis",
        sig_type="array-like (list, tuple, ndarray) with shape (3,) or one of {'x', 'y', 'z'}",
    )

    if np.all(inp == 0):
        msg = "Input axis must be a non-zero vector; instead received (0, 0, 0)."
        raise MagpylibBadUserInput(msg)
    return inp


def check_format_input_angle(inp):
    """check rotate_from_angax angle input and return in formatted form
    - must be scalar (int/float) or array-like
    - if scalar
        - return float
    - if array-like
        - convert inp to ndarray with dtype float
        - inp shape must be (n,)
        - return as ndarray
    """
    if isinstance(inp, numbers.Number):
        return float(inp)

    return check_format_input_vector(
        inp,
        dims=(1,),
        shape_m1="any",
        sig_name="angle",
        sig_type="int, float, or array-like (list, tuple, ndarray) with shape (n,)",
    )


def check_format_input_scalar(
    inp, sig_name, sig_type, allow_None=False, forbid_negative=False
):
    """check scalar input and return in formatted form
    - must be scalar or None (if allowed)
    - must be float compatible
    - transform into float
    """
    if allow_None and inp is None:
        return None

    ERR_MSG = f"Input {sig_name} must be {sig_type}; instead received {inp!r}."

    if not isinstance(inp, numbers.Number):
        raise MagpylibBadUserInput(ERR_MSG)

    inp = float(inp)

    if forbid_negative and inp < 0:
        raise MagpylibBadUserInput(ERR_MSG)
    return inp


def check_format_input_vector(
    inp,
    dims,
    shape_m1,
    sig_name,
    sig_type,
    length=None,
    reshape=False,
    allow_None=False,
    forbid_negative0=False,
):
    """checks vector input and returns in formatted form
    - inp must be array-like
    - convert inp to ndarray with dtype float
    - inp shape must be given by dims and shape_m1
    - print error msg with signature arguments
    - if reshape=True: returns shape (n, 3) - required for position init and setter
    - if allow_None: return None
    - if extend_dim_to2: add a dimension if input is only (1, 2, 3) - required for sensor pixel
    """
    if allow_None and inp is None:
        return None

    is_array_like(
        inp,
        f"Input {sig_name} must be {sig_type}; instead received type {type(inp)!r}.",
    )
    inp = make_float_array(
        inp,
        f"Input {sig_name} must contain only float compatible entries.",
    )
    check_array_shape(
        inp,
        dims=dims,
        shape_m1=shape_m1,
        length=length,
        msg=(
            f"Input {sig_name} must be {sig_type}; "
            f"instead received array-like with shape {inp.shape}."
        ),
    )
    if isinstance(reshape, tuple):
        return np.reshape(inp, reshape)

    if forbid_negative0 and np.any(inp <= 0):
        msg = f"Input parameter {sig_name} cannot have values <= 0."
        raise MagpylibBadUserInput(msg)
    return inp


def check_format_input_vector2(
    inp,
    shape,
    param_name,
):
    """checks vector input and returns in formatted form
    - inp must be array-like
    - convert inp to ndarray with dtype float
    - make sure that inp.ndim = target_ndim, None dimensions are ignored
    """
    is_array_like(
        inp,
        f"Input {param_name} must be array-like; instead received type {type(inp)!r}.",
    )
    inp = make_float_array(
        inp,
        f"Input {param_name} must contain only float compatible entries.",
    )
    for d1, d2 in zip(inp.shape, shape, strict=False):
        if d2 is not None and d1 != d2:
            msg = (
                f"Input {param_name} must have shape {shape}; "
                f"instead received shape {inp.shape}."
            )
            raise ValueError(msg)
    return inp


def check_format_input_vertices(inp, minlength=2):
    """checks vertices input and returns in formatted form
    - vector check with dim = (n, 3) but n must be >=2
    """
    inp = check_format_input_numeric(
        inp,
        dtype=float,
        shapes=((None, 3), (None, None, 3)),
        name="vertices",
        allow_None=True,
    )

    if inp is not None and inp.shape[-2] < minlength:
        msg = (
            f"Input vertices must have at least {minlength} vertices; "
            f"instead received {inp.shape[0]}."
        )
        raise MagpylibBadUserInput(msg)
    return inp


def check_format_input_cylinder_segment(inp):
    """checks vertices input and returns in formatted form
    - vector check with dim = (5) or none
    - check if d1<d2, phi1<phi2
    - check if phi2-phi1 > 360
    - return error msg
    """
    inp = check_format_input_vector(
        inp,
        dims=(1,),
        shape_m1=5,
        sig_name="CylinderSegment.dimension",
        sig_type=(
            "array-like of the form (r1, r2, h, phi1, phi2) with 0<=r1<r2, h>0, "
            "phi1<phi2, and phi2-phi1<=360"
        ),
        allow_None=True,
    )

    if inp is None:
        return None

    r1, r2, h, phi1, phi2 = inp
    case2 = r1 > r2
    case3 = phi1 > phi2
    case4 = (phi2 - phi1) > 360
    case5 = (r1 < 0) | (r2 <= 0) | (h <= 0)
    if case2 | case3 | case4 | case5:
        msg = (
            f"Input CylinderSegment.dimension must be array-like of the form "
            f"(r1, r2, h, phi1, phi2) with 0<=r1<r2, h>0, phi1<phi2 and phi2-phi1<=360; "
            f"instead received {inp!r}."
        )
        raise MagpylibBadUserInput(msg)
    return inp


def check_format_input_backend(inp):
    """checks show-backend input and returns Non if bad input value"""
    backends = [*SUPPORTED_PLOTTING_BACKENDS, "auto"]
    if inp is None:
        inp = default_settings.display.backend
    if inp in backends:
        return inp
    msg = f"Input backend must be one of {[*backends, None]}; instead received {inp!r}."
    raise MagpylibBadUserInput(msg)


def check_format_input_observers(inp, pixel_agg=None):
    """
    checks observers input and returns a list of sensor objects
    """
    # pylint: disable=raise-missing-from
    from magpylib._src.obj_classes.class_Collection import Collection  # noqa: PLC0415
    from magpylib._src.obj_classes.class_Sensor import Sensor  # noqa: PLC0415

    # make bare Sensor, bare Collection into a list
    if isinstance(inp, Collection | Sensor):
        inp = (inp,)

    # note: bare pixel is automatically made into a list by Sensor

    # any good input must now be list/tuple/array
    if not isinstance(inp, list | tuple | np.ndarray):
        raise MagpylibBadUserInput(wrong_obj_msg(inp, allow="observers"))

    # empty list
    if len(inp) == 0:
        raise MagpylibBadUserInput(wrong_obj_msg(inp, allow="observers"))

    # now inp can still be [pos_vec, sens, coll] or just a pos_vec

    try:  # try if input is just a pos_vec
        inp = np.array(inp, dtype=float)
        pix_shapes = [(1, 3) if inp.shape == (3,) else inp.shape]
        return [_src.obj_classes.class_Sensor.Sensor(pixel=inp)], pix_shapes
    except (TypeError, ValueError) as err:  # if not, it must be [pos_vec, sens, coll]
        sensors = []
        for obj_item in inp:
            obj = obj_item
            if isinstance(obj, Sensor):
                sensors.append(obj)
            elif isinstance(obj, Collection):
                child_sensors = format_obj_input(obj, allow="sensors")
                if not child_sensors:
                    raise MagpylibBadUserInput(
                        wrong_obj_msg(obj, allow="observers")
                    ) from err
                sensors.extend(child_sensors)
            else:  # if its not a Sensor or a Collection it can only be a pos_vec
                try:
                    obj = np.array(obj, dtype=float)
                    sensors.append(_src.obj_classes.class_Sensor.Sensor(pixel=obj))
                except Exception:  # or some unwanted crap
                    raise MagpylibBadUserInput(
                        wrong_obj_msg(obj, allow="observers")
                    ) from err

        # all pixel shapes must be the same
        pix_shapes = [
            (1, 3) if (s.pixel is None or s.pixel.shape == (3,)) else s.pixel.shape
            for s in sensors
        ]
        if pixel_agg is None and not all_same(pix_shapes):
            msg = (
                "Input observers must have similar shapes when pixel_agg is None; "
                f"instead received shapes {pix_shapes}."
            )
            raise MagpylibBadUserInput(msg) from err
        return sensors, pix_shapes


def check_format_input_obj(
    inp,
    allow: str,
    recursive=True,
    typechecks=False,
) -> list:
    """
    Returns a flat list of all wanted objects in input.
    Parameters
    ----------
    input: can be
        - objects
    allow: str
        Specify which object types are wanted, separate by +,
        e.g. sensors+collections+sources
    recursive: bool
        Flatten Collection objects
    """
    from magpylib._src.obj_classes.class_BaseExcitations import BaseSource  # noqa: I001, PLC0415
    from magpylib._src.obj_classes.class_Collection import Collection  # noqa: PLC0415
    from magpylib._src.obj_classes.class_Sensor import Sensor  # noqa: PLC0415

    # select wanted
    wanted_types = ()
    if "sources" in allow.split("+"):
        wanted_types += (BaseSource,)
    if "sensors" in allow.split("+"):
        wanted_types += (Sensor,)
    if "collections" in allow.split("+"):
        wanted_types += (Collection,)

    obj_list = []
    for obj in inp:
        # add to list if wanted type
        if isinstance(obj, wanted_types):
            obj_list.append(obj)

        # recursion
        if isinstance(obj, Collection) and recursive:
            obj_list += check_format_input_obj(
                obj,
                allow=allow,
                recursive=recursive,
                typechecks=typechecks,
            )

        # typechecks
        # pylint disable possibly-used-before-assignment
        if typechecks and not isinstance(obj, BaseSource | Sensor | Collection):
            msg = (
                f"Input objects must be {allow} or a flat list thereof; "
                f"instead received {type(obj)!r}."
            )
            raise MagpylibBadUserInput(msg)

    return obj_list


############################################################################################
############################################################################################
# SHOW AND GETB CHECKS


def check_dimensions(sources):
    """check if all sources have dimension (or similar) initialized"""
    for src in sources:
        for arg in ("dimension", "diameter", "vertices"):
            if hasattr(src, arg):
                if getattr(src, arg) is None:
                    msg = f"Input {arg} of {src} must be set."
                    raise MagpylibMissingInput(msg)
                break


def check_excitations(sources):
    """check if all sources have excitation initialized"""
    for src in sources:
        for arg in ("polarization", "current", "moment"):
            if hasattr(src, arg):
                if getattr(src, arg) is None:
                    msg = f"Input {arg} of {src} must be set."
                    raise MagpylibMissingInput(msg)
                break


def check_format_pixel_agg(pixel_agg):
    """
    check if pixel_agg input is acceptable
    return the respective NumPy function
    """

    PIXEL_AGG_ERR_MSG = (
        "Input pixel_agg must be a reference to a NumPy callable that reduces "
        "an array shape like 'mean', 'std', 'median', 'min', ...; "
        f"instead received {pixel_agg!r}."
    )

    if pixel_agg is None:
        return None

    # test NumPy reference
    try:
        pixel_agg_func = getattr(np, pixel_agg)
    except AttributeError as err:
        raise AttributeError(PIXEL_AGG_ERR_MSG) from err

    # test pixel agg function reduce
    x = np.array([[[(1, 2, 3)] * 2] * 3] * 4)
    if not isinstance(pixel_agg_func(x), numbers.Number):
        raise AttributeError(PIXEL_AGG_ERR_MSG)

    return pixel_agg_func


def check_getBH_output_type(output):
    """check if getBH output is acceptable"""
    acceptable = ("ndarray", "dataframe")
    if output not in acceptable:
        msg = f"Input output must be one of {acceptable}; instead received {output!r}."
        raise ValueError(msg)
    if output == "dataframe" and find_spec("pandas") is None:  # pragma: no cover
        msg = (
            "Input output='dataframe' requires Pandas installation, "
            "see https://pandas.pydata.org/docs/getting_started/install.html"
        )
        raise ModuleNotFoundError(msg)
    return output


def check_input_canvas_update(canvas_update, canvas):
    """check if canvas_update is acceptable also depending on canvas input"""
    acceptable = (True, False, "auto", None)
    if canvas_update not in acceptable:
        msg = (
            f"The canvas_update must be one of {acceptable}; "
            f"instead received {canvas_update!r}."
        )
        raise ValueError(msg)
    return canvas is None if canvas_update in (None, "auto") else canvas_update

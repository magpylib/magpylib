""" input checks code"""
import inspect
import numbers

import numpy as np
from scipy.spatial.transform import Rotation

from magpylib import _src
from magpylib._src.defaults.defaults_classes import default_settings
from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib._src.exceptions import MagpylibMissingInput
from magpylib._src.utility import format_obj_input
from magpylib._src.utility import LIBRARY_SENSORS
from magpylib._src.utility import LIBRARY_SOURCES
from magpylib._src.utility import wrong_obj_msg


#################################################################
#################################################################
# FUNDAMENTAL CHECKS


def all_same(lst: list) -> bool:
    """test if all list entries are the same"""
    return lst[1:] == lst[:-1]


def is_array_like(inp, msg: str):
    """test if inp is array_like: type list, tuple or ndarray
    inp: test object
    msg: str, error msg
    """
    if not isinstance(inp, (list, tuple, np.ndarray)):
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


def check_array_shape(inp: np.ndarray, dims: tuple, shape_m1: int, msg: str):
    """check if inp shape is allowed
    inp: test object
    dims: list, list of allowed dims
    shape_m1: shape of lowest level, if 'any' allow any shape
    msg: str, error msg
    """
    if inp.ndim in dims:
        if inp.shape[-1] == shape_m1:
            return None
        if shape_m1 == "any":
            return None
    raise MagpylibBadUserInput(msg)


def check_input_zoom(inp):
    """check show zoom input"""
    if not isinstance(inp, numbers.Number):
        raise MagpylibBadUserInput(
            "Input parameter `zoom` must be a number `zoom>=0`.\n"
            f"Instead received {inp}."
        )
    if inp < 0:
        raise MagpylibBadUserInput(
            "Input parameter `zoom` must be a number `zoom>=0`.\n"
            f"Instead received {inp}."
        )


def check_input_animation(inp):
    """check show animation input"""
    ERR_MSG = (
        "Input parameter `animation` must be boolean or a positive number.\n"
        f"Instead received {inp}."
    )
    if not isinstance(inp, numbers.Number):
        raise MagpylibBadUserInput(ERR_MSG)
    if inp < 0:
        raise MagpylibBadUserInput(ERR_MSG)


#################################################################
#################################################################
# SIMPLE CHECKS


def check_start_type(inp):
    """start input must be int or str"""
    if not (
        isinstance(inp, (int, np.integer)) or (isinstance(inp, str) and inp == "auto")
    ):
        raise MagpylibBadUserInput(
            f"Input parameter `start` must be integer value or 'auto'.\n"
            f"Instead received {repr(inp)}."
        )


def check_degree_type(inp):
    """degrees input must be bool"""
    if not isinstance(inp, bool):
        raise MagpylibBadUserInput(
            "Input parameter `degrees` must be boolean (`True` or `False`).\n"
            f"Instead received {repr(inp)}."
        )


def check_field_input(inp, origin):
    """check field input"""
    if isinstance(inp, str):
        if inp == "B":
            return True
        if inp == "H":
            return False
    raise MagpylibBadUserInput(
        f"{origin} input can only be `field='B'` or `field='H'`.\n"
        f"Instead received {repr(inp)}."
    )


def validate_field_func(val):
    """test if field function for custom source is valid
    - needs to be a callable
    - input and output shape must match
    """
    if val is None:
        return None

    if not callable(val):
        raise MagpylibBadUserInput(
            "Input parameter `field_func` must be a callable.\n"
            f"Instead received {type(val).__name__}."
        )

    fn_args = inspect.getfullargspec(val).args
    if fn_args[:2] != ["field", "observers"]:
        raise MagpylibBadUserInput(
            "Input parameter `field_func` must have two positional args"
            " called 'field' and 'observers'.\n"
            f"Instead received a callable where the first two args are: {fn_args[:2]}"
        )

    for field in ["B", "H"]:
        out = val(field, np.array([[1, 2, 3], [4, 5, 6]]))
        if out is not None:
            if not isinstance(out, np.ndarray):
                raise MagpylibBadUserInput(
                    "Input parameter `field_func` must be a callable that returns B- and H-field"
                    "as numpy ndarray.\n"
                    f"Instead it returns type {type(out)} for {field}-field."
                )
            if out.shape != (2, 3):
                raise MagpylibBadUserInput(
                    "Input parameter `field_func` must be a callable that returns B- and H-field"
                    " as numpy ndarray with shape (n,3), when `observers` input is shape (n,3).\n"
                    f"Instead it returns shape {out.shape} for {field}-field for input shape (2,3)"
                )

    return None


#################################################################
#################################################################
# CHECK - FORMAT


def check_format_input_orientation(inp, init_format=False):
    """checks orientation input returns in formatted form
    - inp must be None or Rotation object
    - transform None to unit rotation as quat (0,0,0,1)
    if init_format: (for move method)
        return inp and inpQ
    else: (for init and setter)
        return inpQ in shape (-1,4)

    This function is used for setter and init only -> shape (1,4) and (4,) input
    creates same behavior.
    """
    # check type
    if not isinstance(inp, (Rotation, type(None))):
        raise MagpylibBadUserInput(
            f"Input parameter `orientation` must be `None` or scipy `Rotation` object.\n"
            f"Instead received type {type(inp)}."
        )
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
    - input must be array_like or None or 0
    """
    if isinstance(inp, numbers.Number) and inp == 0:
        return np.array((0.0, 0.0, 0.0))

    return check_format_input_vector(
        inp,
        dims=(1, 2),
        shape_m1=3,
        sig_name="anchor",
        sig_type="`None` or `0` or array_like (list, tuple, ndarray) with shape (3,)",
        allow_None=True,
    )


def check_format_input_axis(inp):
    """check rotate_from_angax axis input and return in formatted form
    - input must be array_like or str
    - if string 'x'->(1,0,0), 'y'->(0,1,0), 'z'->(0,0,1)
    - convert inp to ndarray with dtype float
    - inp shape must be (3,)
    - axis must not be (0,0,0)
    - return as ndarray shape (3,)
    """
    if isinstance(inp, str):
        if inp == "x":
            return np.array((1, 0, 0))
        if inp == "y":
            return np.array((0, 1, 0))
        if inp == "z":
            return np.array((0, 0, 1))
        raise MagpylibBadUserInput(
            "Input parameter `axis` must be array_like shape (3,) or one of ['x', 'y', 'z'].\n"
            f"Instead received string {inp}.\n"
        )

    inp = check_format_input_vector(
        inp,
        dims=(1,),
        shape_m1=3,
        sig_name="axis",
        sig_type="array_like (list, tuple, ndarray) with shape (3,) or one of ['x', 'y', 'z']",
    )

    if np.all(inp == 0):
        raise MagpylibBadUserInput("Input parameter `axis` must not be (0,0,0).\n")
    return inp


def check_format_input_angle(inp):
    """check rotate_from_angax angle input and return in formatted form
    - must be scalar (int/float) or array_like
    - if scalar
        - return float
    - if array_like
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
        sig_type="int, float or array_like (list, tuple, ndarray) with shape (n,)",
    )


def check_format_input_scalar(
    inp, sig_name, sig_type, allow_None=False, forbid_negative=False
):
    """check sclar input and return in formatted form
    - must be scalar or None (if allowed)
    - must be float compatible
    - tranform into float
    """
    if allow_None:
        if inp is None:
            return None

    ERR_MSG = (
        f"Input parameter `{sig_name}` must be {sig_type}.\n"
        f"Instead received {repr(inp)}."
    )

    if not isinstance(inp, numbers.Number):
        raise MagpylibBadUserInput(ERR_MSG)

    inp = float(inp)

    if forbid_negative:
        if inp < 0:
            raise MagpylibBadUserInput(ERR_MSG)
    return inp


def check_format_input_vector(
    inp,
    dims,
    shape_m1,
    sig_name,
    sig_type,
    reshape=False,
    allow_None=False,
    forbid_negative0=False,
):
    """checks vector input and returns in formatted form
    - inp must be array_like
    - convert inp to ndarray with dtype float
    - inp shape must be given by dims and shape_m1
    - print error msg with signature arguments
    - if reshape=True: returns shape (n,3) - required for position init and setter
    - if allow_None: return None
    - if extend_dim_to2: add a dimension if input is only (1,2,3) - required for sensor pixel
    """
    if allow_None:
        if inp is None:
            return None

    is_array_like(
        inp,
        f"Input parameter `{sig_name}` must be {sig_type}.\n"
        f"Instead received type {type(inp)}.",
    )
    inp = make_float_array(
        inp,
        f"Input parameter `{sig_name}` must contain only float compatible entries.\n",
    )
    check_array_shape(
        inp,
        dims=dims,
        shape_m1=shape_m1,
        msg=(
            f"Input parameter `{sig_name}` must be {sig_type}.\n"
            f"Instead received array_like with shape {inp.shape}."
        ),
    )
    if reshape:
        return np.reshape(inp, (-1, 3))
    if forbid_negative0:
        if np.any(inp <= 0):
            raise MagpylibBadUserInput(
                f"Input parameter `{sig_name}` cannot have values <= 0."
            )
    return inp


def check_format_input_vertices(inp):
    """checks vertices input and returns in formatted form
    - vector check with dim = (n,3) but n must be >=2
    """
    inp = check_format_input_vector(
        inp,
        dims=(2,),
        shape_m1=3,
        sig_name="vertices",
        sig_type="`None` or array_like (list, tuple, ndarray) with shape (n,3)",
        allow_None=True,
    )

    if inp is not None:
        if inp.shape[0] < 2:
            raise MagpylibBadUserInput(
                "Input parameter `vertices` must have more than one vertex."
            )
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
            "array_like of the form (r1, r2, h, phi1, phi2) with r1<r2,"
            "phi1<phi2 and phi2-phi1<=360"
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
        raise MagpylibBadUserInput(
            f"Input parameter `CylinderSegment.dimension` must be array_like of the form"
            f" (r1, r2, h, phi1, phi2) with 0<=r1<r2, h>0, phi1<phi2 and phi2-phi1<=360,\n"
            f"but received {inp} instead."
        )
    return inp


def check_format_input_backend(inp):
    """checks show-backend input and returns Non if bad input value"""
    if inp is None:
        inp = default_settings.display.backend
    if inp in ("matplotlib", "plotly"):
        return inp
    raise MagpylibBadUserInput(
        "Input parameter `backend` must be one of `('matplotlib', 'plotly', None)`.\n"
        f"Instead received {inp}."
    )


def check_format_input_observers(inp, pixel_agg=None):
    """
    checks observers input and returns a list of sensor objects
    """
    # pylint: disable=raise-missing-from
    # pylint: disable=protected-access

    # make bare Sensor, bare Collection into a list
    if getattr(inp, "_object_type", "") in ("Collection", "Sensor"):
        inp = (inp,)

    # note: bare pixel is automatically made into a list by Sensor

    # any good input must now be list/tuple/array
    if not isinstance(inp, (list, tuple, np.ndarray)):
        raise MagpylibBadUserInput(wrong_obj_msg(inp, allow="observers"))

    # empty list
    if len(inp) == 0:
        raise MagpylibBadUserInput(wrong_obj_msg(inp, allow="observers"))

    # now inp can still be [pos_vec, sens, coll] or just a pos_vec

    try:  # try if input is just a pos_vec
        inp = np.array(inp, dtype=float)
        pix_shapes = [(1, 3) if inp.shape == (3,) else inp.shape]
        return [_src.obj_classes.Sensor(pixel=inp)], pix_shapes
    except (TypeError, ValueError):  # if not, it must be [pos_vec, sens, coll]
        sensors = []
        for obj in inp:
            if getattr(obj, "_object_type", "") == "Sensor":
                sensors.append(obj)
            elif getattr(obj, "_object_type", "") == "Collection":
                child_sensors = format_obj_input(obj, allow="sensors")
                if not child_sensors:
                    raise MagpylibBadUserInput(wrong_obj_msg(obj, allow="observers"))
                sensors.extend(child_sensors)
            else:  # if its not a Sensor or a Collection it can only be a pos_vec
                try:
                    obj = np.array(obj, dtype=float)
                    sensors.append(_src.obj_classes.Sensor(pixel=obj))
                except Exception:  # or some unwanted crap
                    raise MagpylibBadUserInput(wrong_obj_msg(obj, allow="observers"))

        # all pixel shapes must be the same
        pix_shapes = [
            (1, 3) if s.pixel.shape == (3,) else s.pixel.shape for s in sensors
        ]
        if pixel_agg is None and not all_same(pix_shapes):
            raise MagpylibBadUserInput(
                "Different observer input shape detected."
                " All observer inputs must be of similar shape, unless a"
                " numpy pixel aggregator is provided, e.g. `pixel_agg='mean'`!"
            )
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
    # select wanted
    wanted_types = []
    if "sources" in allow.split("+"):
        wanted_types += list(LIBRARY_SOURCES)
    if "sensors" in allow.split("+"):
        wanted_types += list(LIBRARY_SENSORS)
    if "collections" in allow.split("+"):
        wanted_types += ["Collection"]

    if typechecks:
        all_types = list(LIBRARY_SOURCES) + list(LIBRARY_SENSORS) + ["Collection"]

    obj_list = []
    for obj in inp:
        obj_type = getattr(obj, "_object_type", None)

        # add to list if wanted type
        if obj_type in wanted_types:
            obj_list.append(obj)

        # recursion
        if (obj_type == "Collection") and recursive:
            obj_list += check_format_input_obj(
                obj,
                allow=allow,
                recursive=recursive,
                typechecks=typechecks,
            )

        # typechecks
        if typechecks:
            if not obj_type in all_types:
                raise MagpylibBadUserInput(
                    f"Input objects must be {allow} or a flat list thereof.\n"
                    f"Instead received {type(obj)}."
                )

    return obj_list


############################################################################################
############################################################################################
# SHOW AND GETB CHECKS


def check_dimensions(sources):
    """check if all sources have dimension (or similar) initialized"""
    # pylint: disable=protected-access
    for s in sources:
        obj_type = getattr(s, "_object_type", None)
        if obj_type in ("Cuboid", "Cylinder", "CylinderSegment"):
            if s.dimension is None:
                raise MagpylibMissingInput(f"Parameter `dimension` of {s} must be set.")
        elif obj_type in ("Sphere", "Loop"):
            if s.diameter is None:
                raise MagpylibMissingInput(f"Parameter `diameter` of {s} must be set.")
        elif obj_type == "Line":
            if s.vertices is None:
                raise MagpylibMissingInput(f"Parameter `vertices` of {s} must be set.")


def check_excitations(sources, custom_field=None):
    """check if all sources have exitation initialized"""
    # pylint: disable=protected-access
    for s in sources:
        obj_type = getattr(s, "_object_type", None)
        if obj_type in ("Cuboid", "Cylinder", "Sphere", "CylinderSegment"):
            if s.magnetization is None:
                raise MagpylibMissingInput(
                    f"Parameter `magnetization` of {s} must be set."
                )
        elif obj_type in ("Loop", "Line"):
            if s.current is None:
                raise MagpylibMissingInput(f"Parameter `current` of {s} must be set.")
        elif obj_type == "Dipole":
            if s.moment is None:
                raise MagpylibMissingInput(f"Parameter `moment` of {s} must be set.")
        elif (obj_type == "CustomSource") and (custom_field is not None):
            if s.field_func is None:
                raise MagpylibMissingInput(
                    f"Cannot compute {custom_field}-field because input parameter"
                    f"`field_func` of {s} has undefined {custom_field}-field computation."
                )
            if s.field_func(custom_field, np.zeros((1, 3))) is None:
                raise MagpylibMissingInput(
                    f"Cannot compute {custom_field}-field because input parameter"
                    f"`field_func` of {s} has undefined {custom_field}-field computation."
                )


def check_format_pixel_agg(pixel_agg):
    """
    check if pixel_agg input is acceptable
    return the respective numpy function
    """

    PIXEL_AGG_ERR_MSG = (
        "Input `pixel_agg` must be a reference to a numpy callable that reduces"
        + " an array shape like 'mean', 'std', 'median', 'min', ...\n"
        + f"Instead received {pixel_agg}."
    )

    if pixel_agg is None:
        return None

    # test numpy reference
    try:
        pixel_agg_func = getattr(np, pixel_agg)
    except AttributeError as err:
        raise AttributeError(PIXEL_AGG_ERR_MSG) from err

    # test pixel agg function reduce
    x = np.array([[[(1, 2, 3)] * 2] * 3] * 4)
    if not isinstance(pixel_agg_func(x), numbers.Number):
        raise AttributeError(PIXEL_AGG_ERR_MSG)

    return pixel_agg_func

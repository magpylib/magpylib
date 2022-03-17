""" input checks code"""

import numbers
import numpy as np
from scipy.spatial.transform import Rotation
from magpylib._src.exceptions import (
    MagpylibBadUserInput,
    MagpylibMissingInput,
)
from magpylib._src.defaults.defaults_classes import default_settings
from magpylib import _src
from magpylib._src.utility import wrong_obj_msg


#################################################################
#################################################################
# FUNDAMENTAL CHECKS


def all_same(lst: list) -> bool:
    """test if all list entries are the same"""
    return lst[1:] == lst[:-1]


def is_array_like(inp, msg:str):
    """ test if inp is array_like: type list, tuple or ndarray
    inp: test object
    msg: str, error msg
    """
    if not isinstance(inp, (list, tuple, np.ndarray)):
        raise MagpylibBadUserInput(msg)

def make_float_array(inp, msg:str):
    """transform inp to array with dtype=float, throw error with bad input
    inp: test object
    msg: str, error msg
    """
    try:
        inp_array = np.array(inp, dtype=float)
    except Exception as err:
        raise MagpylibBadUserInput(msg + f"{err}") from err
    return inp_array


def check_array_shape(inp: np.ndarray, dims:tuple, shape_m1:int, msg:str):
    """check if inp shape is allowed
    inp: test object
    dims: list, list of allowed dims
    shape_m1: shape of lowest level, if 'any' allow any shape
    msg: str, error msg
    """
    if inp.ndim in dims:
        if inp.shape[-1]==shape_m1:
            return None
        if shape_m1 == 'any':
            return None
    raise MagpylibBadUserInput(msg)


def check_input_zoom(inp):
    """check show zoom input"""
    if not isinstance(inp, numbers.Number):
        raise MagpylibBadUserInput(
            "Input parameter `zoom` must be a number `zoom>=0`.\n"
            f"Instead received {inp}."
        )
    if inp<0:
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
    if inp<0:
        raise MagpylibBadUserInput(ERR_MSG)


#################################################################
#################################################################
# SIMPLE CHECKS


def check_start_type(inp):
    """start input must be int or str"""
    if not (isinstance(inp, (int, np.integer)) or (isinstance(inp, str) and inp == 'auto')):
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
        if inp == 'B':
            return True
        if inp == 'H':
            return False
    raise MagpylibBadUserInput(
        f"{origin} input can only be `field='B'` or `field='H'`.\n"
        f"Instead received {repr(inp)}."
    )


def validate_field_lambda(val, bh):
    """test if field function for custom source is valid
    - needs to be a callable
    - input and output shape must match
    """
    if val is not None:
        if not callable(val):
            raise MagpylibBadUserInput(
                f"Input parameter `field_{bh}_lambda` must be a callable."
            )

        out = val(np.array([[1, 2, 3], [4, 5, 6]]))
        out_shape = np.array(out).shape
        case2 = out_shape!=(2, 3)

        if case2:
            raise MagpylibBadUserInput(
                f"Input parameter `field_{bh}_lambda` must be a callable function"
                " and return a field ndarray of shape (n,3) when its `observer`"
                " input is of shape (n,3).\n"
                f"Instead received shape {out_shape}.")
    return val


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
            f"Instead received type {type(inp)}.")
    # handle None input and compute inpQ
    if inp is None:
        inpQ = np.array((0,0,0,1))
        inp=Rotation.from_quat(inpQ)
    else:
        inpQ = inp.as_quat()
    # return
    if init_format:
        return np.reshape(inpQ, (-1,4))
    return inp, inpQ


def check_format_input_anchor(inp):
    """ checks rotate anchor input and return in formatted form
    - input must be array_like or None or 0
    """
    if isinstance(inp, numbers.Number) and inp == 0:
        return np.array((0.0, 0.0, 0.0))

    return check_format_input_vector(inp,
        dims=(1,2),
        shape_m1=3,
        sig_name='anchor',
        sig_type='`None` or `0` or array_like (list, tuple, ndarray) with shape (3,)',
        allow_None=True)


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
        if inp == 'x':
            return np.array((1,0,0))
        if inp == 'y':
            return np.array((0,1,0))
        if inp == 'z':
            return np.array((0,0,1))
        raise MagpylibBadUserInput(
            "Input parameter `axis` must be array_like shape (3,) or one of ['x', 'y', 'z'].\n"
            f"Instead received string {inp}.\n")

    inp = check_format_input_vector(inp,
        dims=(1,),
        shape_m1=3,
        sig_name='axis',
        sig_type="array_like (list, tuple, ndarray) with shape (3,) or one of ['x', 'y', 'z']")

    if np.all(inp==0):
        raise MagpylibBadUserInput(
            "Input parameter `axis` must not be (0,0,0).\n")
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

    return check_format_input_vector(inp,
        dims=(1,),
        shape_m1='any',
        sig_name='angle',
        sig_type='int, float or array_like (list, tuple, ndarray) with shape (n,)')


def check_format_input_scalar(
    inp,
    sig_name,
    sig_type,
    allow_None=False,
    forbid_negative=False):
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
        f"Instead received {repr(inp)}.")

    if not isinstance(inp, numbers.Number):
        raise MagpylibBadUserInput(ERR_MSG)

    inp = float(inp)

    if forbid_negative:
        if inp<0:
            raise MagpylibBadUserInput(ERR_MSG)
    return inp


def check_format_input_vector(inp,
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

    is_array_like(inp,
        f"Input parameter `{sig_name}` must be {sig_type}.\n"
        f"Instead received type {type(inp)}."
    )
    inp = make_float_array(inp,
        f"Input parameter `{sig_name}` must contain only float compatible entries.\n"
    )
    check_array_shape(inp, dims=dims, shape_m1=shape_m1, msg=(
        f"Input parameter `{sig_name}` must be {sig_type}.\n"
        f"Instead received array_like with shape {inp.shape}.")
    )
    if reshape:
        return np.reshape(inp, (-1,3))
    if forbid_negative0:
        if np.any(inp<=0):
            raise MagpylibBadUserInput(
                f"Input parameter `{sig_name}` cannot have values <= 0."
            )
    return inp


def check_format_input_vertices(inp):
    """checks vertices input and returns in formatted form
    - vector check with dim = (n,3) but n must be >=2
    """
    inp = check_format_input_vector(inp,
        dims=(2,),
        shape_m1=3,
        sig_name='vertices',
        sig_type='`None` or array_like (list, tuple, ndarray) with shape (n,3)',
        allow_None=True)

    if inp is not None:
        if inp.shape[0]<2:
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
    inp = check_format_input_vector(inp,
        dims=(1,),
        shape_m1=5,
        sig_name='CylinderSegment.dimension',
        sig_type=(
            'array_like of the form (r1, r2, h, phi1, phi2) with r1<r2,'
            'phi1<phi2 and phi2-phi1<=360'),
        allow_None=True)

    if inp is None:
        return None

    r1, r2, h, phi1, phi2 = inp
    case2 = r1>r2
    case3 = phi1>phi2
    case4 = (phi2 - phi1)>360
    case5 = (r1<0) | (r2<=0) | (h<=0)
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
    if inp in ('matplotlib', 'plotly'):
        return inp
    raise MagpylibBadUserInput(
        "Input parameter `backend` must be one of `('matplotlib', 'plotly', None)`.\n"
        f"Instead received {inp}.")


def check_format_input_observers(inp):
    """
    checks observer input and returns a list of sensor objects
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

    try: # try if input is just a pos_vec
        inp = np.array(inp, dtype=float)
        return [_src.obj_classes.Sensor(pixel=inp)]
    except (TypeError, ValueError): # if not, it must be [pos_vec, sens, coll]
        sensors=[]
        for obj in inp:
            if getattr(obj, "_object_type", "") == "Sensor":
                sensors.append(obj)
            elif getattr(obj, "_object_type", "") == "Collection":
                if not obj.sensors:
                    raise MagpylibBadUserInput(wrong_obj_msg(obj, allow="observers"))
                sensors.extend(obj.sensors)
            else: # if its not a Sensor or a Collection it can only be a pos_vec
                try:
                    obj = np.array(obj, dtype=float)
                    sensors.append(_src.obj_classes.Sensor(pixel=obj))
                except Exception: # or some unwanted crap
                    raise MagpylibBadUserInput(wrong_obj_msg(obj, allow="observers"))

        # all pixel shapes must be the same
        pix_shapes = [s._pixel.shape for s in sensors]
        if not all_same(pix_shapes):
            raise MagpylibBadUserInput(
                'Different observer input detected.'
                ' All sensor pixel and position vector inputs must'
                ' be of similar shape !')
        return sensors


############################################################################################
############################################################################################
# SHOW AND GETB CHECKS

def check_dimensions(sources):
    """check if all sources have dimension (or similar) initialized"""
    # pylint: disable=protected-access
    for s in sources:
        obj_type = getattr(s, '_object_type', None)
        if obj_type in ('Cuboid', 'Cylinder', 'CylinderSegment'):
            if s.dimension is None:
                raise MagpylibMissingInput(f"Parameter `dimension` of {s} must be set.")
        elif obj_type in ('Sphere', 'Loop'):
            if s.diameter is None:
                raise MagpylibMissingInput(f"Parameter `diameter` of {s} must be set.")
        elif obj_type == 'Line':
            if s.vertices is None:
                raise MagpylibMissingInput(f"Parameter `vertices` of {s} must be set.")


def check_excitations(sources):
    """check if all sources have exitation initialized"""
    # pylint: disable=protected-access
    for s in sources:
        obj_type = getattr(s, '_object_type', None)
        if obj_type in ('Cuboid', 'Cylinder', 'Sphere', 'CylinderSegment'):
            if s.magnetization is None:
                raise MagpylibMissingInput(f"Parameter `magnetization` of {s} must be set.")
        elif obj_type in ('Loop', 'Line'):
            if s.current is None:
                raise MagpylibMissingInput(f"Parameter `current` of {s} must be set.")
        elif obj_type == 'Dipole':
            if s.moment is None:
                raise MagpylibMissingInput(f"Parameter `moment` of {s} must be set.")

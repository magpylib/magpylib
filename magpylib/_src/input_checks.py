""" input checks code"""

import numpy as np
from scipy.spatial.transform import Rotation
from magpylib._src.exceptions import (
    MagpylibBadUserInput,
    MagpylibBadInputShape,
    MagpylibMissingInput,
)

def is_array_like(inp, origin):
    """ test if inp is array_like: type list, tuple or ndarray"""
    if not isinstance(inp, (list, tuple, np.ndarray)):
        raise MagpylibBadUserInput(
            f"Input parameter `{origin}` must be array_like (list, tuple, ndarray).\n"
            f"Instead received type {type(inp)}."
        )

def make_float_array(inp, origin):
    """transform inp to array with dtype=float, throw error with bad input
    inp: test object
    origin: str, error msg reference
    """
    try:
        inp_array = np.array(inp, dtype=float)
    except Exception as err:
        raise MagpylibBadUserInput(
            f"Input parameter `{origin}` must contain only float compatible entries,\n"
            f"{err}"
            ) from err
    return inp_array

def check_array_shape(inp: np.ndarray, dims:tuple, shape_m1:int, origin:str, origin_shape:str):
    """check if inp shape is allowed
    inp: test object
    dims: list, list of allowed dims
    shape_m1: shape of lowest level
    origin: str, error msg reference
    """
    if inp.ndim in dims:
        if inp.shape[-1]==shape_m1:
            return None
    raise MagpylibBadUserInput(
            f"Input parameter `{origin}` must have shape {origin_shape}.\n"
            f"Instead received shape {inp.shape}."
        )

def check_orientation_type(inp):
    """orientation input mut be scipy Rotation or None"""
    if not isinstance(inp, (Rotation, type(None))):
        raise MagpylibBadUserInput(
            f"Input parameter `orientation` must be `None` or scipy `Rotation` object,\n"
            f"received {repr(inp)} instead."
        )

###########################

def check_format_input_position(inp):
    """checks input and return formatted ndarray of shape (n,3).
    - inp must be array_like
    - convert inp to ndarray with dtype float
    - inp shape must be (3,) or (n,3)
    returns shape (n,3)

    This function is used for setter and init only -> (1,3) and (3,) input
    creates same behavior.
    """
    is_array_like(inp, origin='position')
    inp = make_float_array(inp, origin='position')
    check_array_shape(inp, dims=(1,2), shape_m1=3, origin='position', origin_shape='(3,) or (n,3)')
    return np.reshape(inp, (-1,3))


def check_format_input_orientation(inp):
    """checks input and return formatted ndarray of shape (n,4).
    - inp must be None or Rotation object
    - transform None to unit rotation as quat (0,0,0,1)    
    returns shape (n,4)

    This function is used for setter and init only -> (1,4) and (4,) input
    creates same behavior.
    """
    check_orientation_type(inp)
    inpQ = np.array([(0,0,0,1)]) if inp is None else inp.as_quat()
    return np.reshape(inpQ, (-1,4))


def check_field_input(inp, origin):
    """check field input"""
    if isinstance(inp, str):
        if inp == 'B':
            return True
        if inp == 'H':
            return False
    raise MagpylibBadUserInput(
        f"{origin} input can only be `field='B'` or `field='H'`,\n"
        f"received shape {repr(inp)} instead."
    )


def check_dimensions(sources):
    """check if all sources have dimension (or similar) initialized"""
    # pylint: disable=protected-access
    for s in sources:
        obj_type = getattr(s, '_object_type', None)
        if obj_type in ("Cuboid", "Cylinder"):
            if np.isnan(s.dimension).any():
                raise MagpylibMissingInput(f"Parameter `dimension` of {s} must be set.")
        elif obj_type in ("Sphere", "Loop"):
            if s.diameter is None:
                raise MagpylibMissingInput(f"Parameter `diameter` of {s} must be set.")
        elif obj_type == "Line":
            if None in s.vertices[0]:
                raise MagpylibMissingInput(f"Parameter `vertices` of {s} must be set.")


def check_excitations(sources):
    """check if all sources have exitation initialized"""
    # pylint: disable=protected-access
    for s in sources:
        obj_type = getattr(s, '_object_type', None)
        if obj_type in ("Cuboid", "Cylinder", "Sphere"):
            if np.isnan(s.magnetization).any():
                raise MagpylibMissingInput(f"Parameter `magnetization` of {s} must be set.")
        elif obj_type in ("Loop", "Line"):
            if s.current is None:
                raise MagpylibMissingInput(f"Parameter `current` of {s} must be set.")
        elif obj_type == "Dipole":
            if np.isnan(s.moment).any():
                raise MagpylibMissingInput(f"Parameter `moment` of {s} must be set.")


def check_path_format(inp, origin):
    """test if input has the correct shape (3,) or (n,3)"""
    case1 = (inp.shape[-1] == 3)
    case2 = (inp.ndim in (1, 2))
    if not case1 & case2:
        raise MagpylibBadInputShape(
            f"Input parameter {origin} has bad shape. Must be shape (3,) or (n,3),\n"
            f"received shape {inp.shape} instead."
        )


def check_anchor_type(anch):
    """anchor input must be vector or None or 0"""
    if not (isinstance(anch, (list, tuple, np.ndarray, type(None))) or anch == 0):
        raise MagpylibBadUserInput(
            f"Input parameter `anchor` must be `None`, `0` or array_like with shape (3,) or (n,3),\n"
            f"received {repr(anch)} instead."
        )


def check_anchor_format(anch):
    """must be shape (n,3), (3,) or 0"""
    case0 = (anch.ndim==0) and anch==0
    case1 = (anch.ndim==1) and len(anch)==3
    case2 = (anch.ndim==2) and (anch.shape[-1]==3)
    if not case0 | case1 | case2:
        raise MagpylibBadInputShape(
            f"Input parameter `anchor` must be `None`, `0` or array_like with shape (3,) or (n,3),\n"
            f"received {repr(anch)} instead."
        )





def check_start_type(start):
    """start input must be int or str"""
    if not (isinstance(start, (int, np.int_)) or start == 'auto'):
        raise MagpylibBadUserInput(
            f"Input parameter `start` must be int or 'auto',\n"
            f"received {repr(start)} instead."
        )


def check_angle_type(inp):
    """angle input must be scalar or vector"""
    if not isinstance(inp, (int, float, list, tuple, np.ndarray, np.int_, np.float_, range)):
        raise MagpylibBadUserInput(
            f"Input parameter `angle` must be int, float, or array_like with shape (n,),\n"
            f"but received {repr(inp)} instead."
        )

def check_angle_format(inp):
    """angle format must be scalar or of shape (N,)"""
    if isinstance(inp, np.ndarray):
        if inp.ndim not in (0,1):
            raise MagpylibBadInputShape(
                f"Input parameter `angle` must be int, float, or array_like with shape (n,),\n"
                f"but received {repr(inp)} instead."
            )


def check_axis_type(inp):
    """axis input must be vector or str"""
    if not isinstance(inp, (list, tuple, np.ndarray, str)):
        raise MagpylibBadUserInput(
            f"Input parameter `axis` must be str 'x', 'y', 'z', or array_like with shape (3,),\n"
            f"but received {repr(inp)} instead."
        )


def check_axis_format(inp):
    """
    - must be of shape (3,)
    - must not be (0,0,0)
    """
    case1 = not inp.shape==(3,)
    case2 = np.all(inp==0)
    if case1 | case2:
        raise MagpylibBadUserInput(
            f"Input parameter `axis` must be str 'x', 'y', 'z', or non-zero array_like with shape (3,),\n"
            f"but received {repr(inp)} instead."
        )


def check_degree_type(inp):
    """degrees input must be bool"""
    if not isinstance(inp, bool):
        raise MagpylibBadUserInput(
            f"Input parameter `degrees` must be bool (`True` or `False`),\n"
            f"but received {repr(inp)} instead."
        )


def check_scalar_type(inp, origin):
    """scalar input must be int or float or nan"""
    if not (isinstance(inp, (int, float, np.int_, np.float_)) or inp is None):
        raise MagpylibBadUserInput(
            f"{origin} input must be scalar (int or float),\n"
            f"but received {repr(inp)} instead."
        )


def check_vector_type(inp, origin):
    """
    - vector input must be list, tuple or ndarray
    - return error msg with reference to origin
    """
    if not isinstance(inp, (list, tuple, np.ndarray)):
        raise MagpylibBadUserInput(
            f"{origin} input must be array_like,\n"
            f"but received {repr(inp)} instead."
        )


def check_vector_format(inp, shape, origin):
    """
    - check if vector input has correct format
    - return error msg with reference to origin
    """
    if not inp.shape == shape:
        raise MagpylibBadInputShape(
            f"Bad {origin} input shape. Must be shape {shape}."
            f"but received shape {inp.shape} instead."
        )


def check_input_cyl_sect(inp):
    """
    - check if cylinder dimension has correct format (5,)
    - check if d1<d2, phi1<phi2
    - check if phi2-phi1 > 360
    - return error msg
    """
    trigger = False

    case1 = inp.shape==(5,)
    if case1:
        d1, d2, _, phi1, phi2 = inp
        case2 = d1>=d2
        case3 = phi1>=phi2
        case4 = (phi2 - phi1)>360
        if case2 | case3 | case4:
            trigger=True
    else:
        trigger=True
    if trigger:
        raise MagpylibBadUserInput(
        f"CylinderSegment input format must be (r1,r2,h,phi1,phi2) with r1<r2, phi1<phi2 and phi2-phi1<=360,\n"
        f"but received {inp} instead."
        )


def check_position_format(inp, origin):
    """
    - test if input has the correct shape (...,3)
    - return error msg with reference to origin
    """
    if not inp.shape[-1] == 3:
        raise MagpylibBadInputShape(
            f"Bad {origin} input shape. Must be (...,3),\n"
            f"but received shape {inp.shape} instead."
        )


def check_vertex_format(inp):
    """
    - test if input has the correct shape (N,3) with N >= 2
    - return err msg
    """
    case1 = not inp.shape[-1]==3
    case2 = not inp.ndim==2
    case3 = not inp.shape[0]>1
    if case1 | case2 | case3:
        raise MagpylibBadInputShape(
            "Bad shape of input parameter `vertex`. Must be (n,3) with n>1,\n"
            f"but received shape {inp.shape} instead."
        )


def validate_field_lambda(val, bh):
    """
    test if field function for custom source is valid
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
        if out_shape!=(2, 3):
            raise MagpylibBadUserInput(
                f"Input parameter field_{bh}_lambda shape and output "
                "must match and be of dimension (n,3)\n"
                f"received shape={out_shape} instead"
        )
    return val

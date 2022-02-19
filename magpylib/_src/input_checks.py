""" input checks code"""

import numpy as np
from scipy.spatial.transform import Rotation
from magpylib._src.exceptions import (
    MagpylibBadUserInput,
    MagpylibBadInputShape,
    MagpylibMissingInput,
)

#################################################################
#################################################################
# FUNDAMENTAL CHECKS

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

def make_float(inp, msg:str):
    """transform inp to float, throw error with bad input
    inp: test object
    msg: str, error msg
    """
    try:
        inp = float(inp)
    except Exception as err:
        raise MagpylibBadUserInput(msg + f"{err}") from err
    return inp

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



#################################################################
#################################################################
# SIMPLE CHECKS

def check_start_type(inp):
    """start input must be int or str"""
    if not (isinstance(inp, (int, np.int_)) or inp == 'auto'):
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
    if np.isscalar(inp) and inp == 0:
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
    if np.isscalar(inp):
        return float(inp)

    return check_format_input_vector(inp,
        dims=(1,),
        shape_m1='any',
        sig_name='angle',
        sig_type='int, float or array_like (list, tuple, ndarray) with shape (n,)')


def check_format_input_scalar(inp, signature, allow_None=True):
    """check sclar input and return in formatted form
    - must be scalar or None (if allowed)
    - must be float compatible
    - tranform into float
    """
    if allow_None:
        if inp is None:
            return None

    if not np.isscalar(inp):
        raise MagpylibBadUserInput(
            f"Input parameter `{signature}` must be scalar (int or float) or `None`.\n"
            f"Instead received {repr(inp)}."
        )
    inp = make_float(inp,
        f"Input parameter `{signature}` input must be float compatible.\n"
    )
    return inp


def check_format_input_vector(inp,
    dims,
    shape_m1,
    sig_name,
    sig_type,
    reshape=False,
    allow_None=False):
    """checks vector input and returns in formatted form
    - inp must be array_like
    - convert inp to ndarray with dtype float
    - inp shape must be given by dims and shape_m1
    - print error msg with signature arguments
    - if reshape=True: returns shape (n,3) - required for position init and setter
    - if allow_None: return None
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
    return inp


def check_format_input_vertices(inp):
    """checks vertices input and returns in formatted form
    - vector check with dim = (n,3) but n must be >=2
    """
    inp = check_format_input_vector(inp,
        dims=(2,),
        shape_m1=3,
        sig_name='vertices',
        sig_type='array_like (list, tuple, ndarray) with shape (n,3)',
        allow_None=True)

    if inp is not None:
        if inp.shape[0]<2:
            raise MagpylibBadUserInput(
                "Input parameter `vertices` must have more than one vertex position.\n"
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
        sig_type='array_like of the form (r1, r2, h, phi1, phi2) with r1<r2, phi1<phi2 and phi2-phi1<=360',
        allow_None=True)

    if inp is None:
        return None

    d1, d2, _, phi1, phi2 = inp
    case2 = d1>=d2
    case3 = phi1>=phi2
    case4 = (phi2 - phi1)>360
    if case2 | case3 | case4:
        raise MagpylibBadUserInput(
            f"Input parameter `CylinderSegment.dimension` must be array_like of the form (r1, r2, h, phi1, phi2) with r1<r2, phi1<phi2 and phi2-phi1<=360,\n"
            f"but received {inp} instead."
        )
    return inp





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


# def check_anchor_type(anch):
#     """anchor input must be vector or None or 0"""
#     if not (isinstance(anch, (list, tuple, np.ndarray, type(None))) or anch == 0):
#         raise MagpylibBadUserInput(
#             f"Input parameter `anchor` must be `None`, `0` or array_like with shape (3,) or (n,3),\n"
#             f"received {repr(anch)} instead."
#         )


# def check_anchor_format(anch):
#     """must be shape (n,3), (3,) or 0"""
#     case0 = (anch.ndim==0) and anch==0
#     case1 = (anch.ndim==1) and len(anch)==3
#     case2 = (anch.ndim==2) and (anch.shape[-1]==3)
#     if not case0 | case1 | case2:
#         raise MagpylibBadInputShape(
#             f"Input parameter `anchor` must be `None`, `0` or array_like with shape (3,) or (n,3),\n"
#             f"received {repr(anch)} instead."
#         )








# def check_angle_type(inp):
#     """angle input must be scalar or vector"""
#     if not isinstance(inp, (int, float, list, tuple, np.ndarray, np.int_, np.float_, range)):
#         raise MagpylibBadUserInput(
#             f"Input parameter `angle` must be int, float, or array_like with shape (n,),\n"
#             f"but received {repr(inp)} instead."
#         )

# def check_angle_format(inp):
#     """angle format must be scalar or of shape (N,)"""
#     if isinstance(inp, np.ndarray):
#         if inp.ndim not in (0,1):
#             raise MagpylibBadInputShape(
#                 f"xxInput parameter `angle` must be int, float, or array_like with shape (n,),\n"
#                 f"but received {repr(inp)} instead."
#             )


# def check_axis_type(inp):
#     """axis input must be vector or str"""
#     if not isinstance(inp, (list, tuple, np.ndarray, str)):
#         raise MagpylibBadUserInput(
#             f"Input parameter `axis` must be str 'x', 'y', 'z', or array_like with shape (3,),\n"
#             f"but received {repr(inp)} instead."
#         )


# def check_axis_format(inp):
#     """
#     - must be of shape (3,)
#     - must not be (0,0,0)
#     """
#     case1 = not inp.shape==(3,)
#     case2 = np.all(inp==0)
#     if case1 | case2:
#         raise MagpylibBadUserInput(
#             f"Input parameter `axis` must be str 'x', 'y', 'z', or non-zero array_like with shape (3,),\n"
#             f"but received {repr(inp)} instead."
#         )





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

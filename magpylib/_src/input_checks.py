""" input checks code"""

import numpy as np
from scipy.spatial.transform import Rotation
from magpylib._src.exceptions import (
    MagpylibBadUserInput,
    MagpylibBadInputShape,
    MagpylibMissingInput,
)


def check_field_input(inp, origin):
    """
    check field input
    """
    if isinstance(inp, str):
        if inp == 'B':
            return True
        if inp == 'H':
            return False
    msg = f'{origin} input can only be "field=B" or "field=H".'
    raise MagpylibBadUserInput(msg)


def check_dimensions(sources):
    """check if all sources have dimension (or similar) initialized"""
    # pylint: disable=protected-access
    for s in sources:
        obj_type = getattr(s, '_object_type', None)
        if obj_type in ("Cuboid", "Cylinder"):
            if np.isnan(s.dimension).any():
                raise MagpylibMissingInput(f"{s} dimension must be initialized.")
        elif obj_type in ("Sphere", "Loop"):
            if s.diameter is None:
                raise MagpylibMissingInput(f"{s} diameter must be initialized.")
        elif obj_type == "Line":
            if None in s.vertices[0]:
                raise MagpylibMissingInput(f"{s} vertices must be initialized.")


def check_excitations(sources):
    """check if all sources have dimension (or similar) initialized"""
    # pylint: disable=protected-access
    for s in sources:
        obj_type = getattr(s, '_object_type', None)
        if obj_type in ("Cuboid", "Cylinder", "Sphere"):
            if np.isnan(s.magnetization).any():
                raise MagpylibMissingInput(f"{s} magnetization must be initialized.")
        elif obj_type in ("Loop", "Line"):
            if s.current is None:
                raise MagpylibMissingInput(f"{s} current must be initialized.")
        elif obj_type == "Dipole":
            if np.isnan(s.moment).any():
                raise MagpylibMissingInput(f"{s} moment must be initialized.")


def check_path_format(inp, origin):
    """test if input has the correct shape (3,) or (N,3)"""
    msg = f"Bad {origin} input shape. Must be (3,) or (N,3)"
    if not inp.shape[-1] == 3:
        raise MagpylibBadInputShape(msg)
    if not inp.ndim in (1, 2):
        raise MagpylibBadInputShape(msg)


def check_anchor_type(anch):
    """anchor input must be vector or None or 0"""
    if not (isinstance(anch, (list, tuple, np.ndarray, type(None))) or anch == 0):
        msg = "anchor input must be 0, None or a vector (list, tuple or ndarray)."
        raise MagpylibBadUserInput(msg)


def check_anchor_format(anch):
    """must be shape (N,3), (3,) or 0"""
    if not np.all(anch == np.array(0)):
        msg = "Bad anchor input. Must be None, 0 or shape (3,) or (N,3)."
        if not anch.shape[-1] == 3:
            raise MagpylibBadInputShape(msg)
        if not anch.ndim in (1, 2):
            raise MagpylibBadInputShape(msg)


def check_rot_type(inp):
    """rotation input mut be scipy Rotation"""
    if not isinstance(inp, (Rotation, type(None))):
        msg = "orientation input must be None or scipy Rotation object."
        raise MagpylibBadUserInput(msg)


def check_start_type(start):
    """start input must be int or str"""
    if not (isinstance(start, (int, np.int_)) or start == 'auto'):
        msg = 'start input must be int or str ("auto")'
        raise MagpylibBadUserInput(msg)


def check_angle_type(angle):
    """angle input must be scalar or vector"""
    if not isinstance(angle, (int, float, list, tuple, np.ndarray, np.int_, np.float_)):
        msg = (
            "angle input must be scalar (int, float) or vector (list, tuple, ndarray)."
        )
        raise MagpylibBadUserInput(msg)


def check_angle_format(angle):
    """angle format must be scalar or of shape (N,)"""
    if isinstance(angle, np.ndarray):
        if angle.ndim not in (0,1):
            msg = "Bad angle input shape. Must be scalar or 1D vector."
            raise MagpylibBadInputShape(msg)


def check_axis_type(ax):
    """axis input must be vector or str"""
    if not isinstance(ax, (list, tuple, np.ndarray, str)):
        msg = "axis input must be a vector (list, tuple or ndarray) or a \
            str ('x', 'y', 'z')."
        raise MagpylibBadUserInput(msg)


def check_axis_format(axis):
    """
    - must be of shape (3,)
    - must not be (0,0,0)
    """
    msg = "Bad axis input shape. Must be shape (3,)."
    if not axis.shape == (3,):
        raise MagpylibBadInputShape(msg)
    msg = "Bad axis input. Cannot be (0,0,0)."
    if np.all(axis == 0):
        raise MagpylibBadUserInput(msg)


def check_degree_type(deg):
    """degrees input must be bool"""
    if not isinstance(deg, bool):
        msg = "degrees input must be bool (True or False)."
        raise MagpylibBadUserInput(msg)


def check_scalar_type(inp, origin):
    """scalar input must be int or float or nan"""
    if not (isinstance(inp, (int, float, np.int_, np.float_)) or inp is None):
        msg = origin + " input must be scalar (int or float)."
        raise MagpylibBadUserInput(msg)


# def check_scalar_init(inp, origin):
#     """ check if scalar input was initialized (former None)
#     """
#     if inp is None:
#         msg = origin + ' must be initialized.'
#         raise MagpylibBadUserInput(msg)


def check_vector_type(inp, origin):
    """
    - vector input must be list, tuple or ndarray
    - return error msg with reference to origin
    """
    if not isinstance(inp, (list, tuple, np.ndarray)):
        msg = origin + " input must be vector type (list, tuple or ndarray)."
        raise MagpylibBadUserInput(msg)


def check_vector_format(inp, shape, origin):
    """
    - check if vector input has correct format
    - return error msg with reference to origin
    """
    if not inp.shape == shape:
        msg = f"Bad {origin} input shape. Must be shape {shape}."
        raise MagpylibBadInputShape(msg)


def check_input_cyl_sect(inp):
    """
    - check if cylinder dimension has correct format (5,)
    - check if d1<d2, phi1<phi2
    - check if phi2-phi1 > 360
    - return error msg
    """
    if not inp.shape == (5,):
        msg = """Bad dimension input shape. Dimension must be (d1,d2,h,phi1,phi2)."""
        raise MagpylibBadInputShape(msg)

    d1, d2, _, phi1, phi2 = inp
    if d1 >= d2:
        msg = "d1 must be smaller than d2."
        raise MagpylibBadUserInput(msg)

    if phi1 >= phi2:
        msg = "phi1 must be smaller than phi2."
        raise MagpylibBadUserInput(msg)

    if (phi2 - phi1) > 360:
        msg = "phi2-phi1 cannot be larger than 360°."
        raise MagpylibBadUserInput(msg)


def check_position_format(inp, origin):
    """
    - test if input has the correct shape (...,3)
    - return error msg with reference to origin
    """
    msg = f"Bad {origin} input shape. Must be (...,3)."
    if not inp.shape[-1] == 3:
        raise MagpylibBadInputShape(msg)


def check_vertex_format(inp):
    """
    - test if input has the correct shape (N,3) with N >= 2
    - return err msg
    """
    msg = "Bad vertex input shape. Must be (N,3) with N>1"
    if not inp.shape[-1] == 3:
        raise MagpylibBadInputShape(msg)
    if not inp.ndim == 2:
        raise MagpylibBadInputShape(msg)
    if not inp.shape[0] > 1:
        raise MagpylibBadInputShape(msg)


def validate_field_lambda(val, bh):
    """
    test if field function for custom source is valid
    - needs to be a callable
    - input and output schape must match
    """
    if val is not None:
        assert callable(val), f"field_{bh}_lambda must be a callable"
        out = val(np.array([[1, 2, 3], [4, 5, 6]]))
        out_shape = np.array(out).shape
        assert out_shape == (2, 3), (
            f"field_{bh}_lambda input shape and output "
            "shape must match and be of dimension (n,3)\n"
            f"received shape={out_shape} instead"
        )
    return val

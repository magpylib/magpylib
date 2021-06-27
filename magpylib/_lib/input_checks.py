""" input checks code"""

import numpy as np
from scipy.spatial.transform import Rotation
from magpylib._lib.exceptions import (MagpylibBadUserInput,
    MagpylibBadInputShape)


def check_path_format(inp, origin):
    """ test if input has the correct shape (3,) or (N,3)
    """
    msg = f'Bad {origin} input shape. Must be (3,) or (N,3)'
    if not inp.shape[-1] == 3:
        raise MagpylibBadInputShape(msg)
    if not inp.ndim in (1,2) :
        raise MagpylibBadInputShape(msg)


def check_anchor_type(anch):
    """ anchor input must be vector or None or 0
    """
    if not (isinstance(anch, (list, tuple, np.ndarray, type(None))) or anch==0):
        msg = 'anchor input must be 0, None or a vector (list, tuple or ndarray).'
        raise MagpylibBadUserInput(msg)

def check_anchor_format(anch):
    """ must be shape (3,) or 0
    """
    if anch is not None:
        if not np.all(anch==np.array(0)):
            if not anch.shape==(3,):
                msg = 'Bad anchor input. Must be None, 0 or shape (3,).'
                raise MagpylibBadInputShape(msg)

def check_rot_type(inp):
    """ rotation input mut be scipy Rotation
    """
    if not isinstance(inp, (Rotation, type(None))):
        msg = 'rot input must be None or scipy Rotation object.'
        raise MagpylibBadUserInput(msg)


def check_start_type(start):
    """ start input must be int or str
    """
    if not (isinstance(start, int) or start=='append'):
        msg = 'start input must be int or str ("attach")'
        raise MagpylibBadUserInput(msg)


def check_increment_type(inrc):
    """ incremnt input must be bool
    """
    if not isinstance(inrc, bool):
        msg = 'increment input must be bool (True or False).'
        raise MagpylibBadUserInput(msg)


def check_angle_type(angle):
    """ angle input must be scalar or vector
    """
    if not isinstance(angle, (int, float, list, tuple, np.ndarray)):
        msg = 'angle input must be scalar (int, float) or vector (list, tuple, ndarray).'
        raise MagpylibBadUserInput(msg)

def check_angle_format(angle):
    """ angle format must be of shape (N,)
    """
    if not angle.ndim == 1:
        msg = 'Bad angle input shape. Must be scalar or 1D vector.'
        raise MagpylibBadInputShape(msg)


def check_axis_type(ax):
    """ axis input must be vector or str
    """
    if not isinstance(ax, (list, tuple, np.ndarray, str)):
        msg = 'axis input must be a vector (list, tuple or ndarray) or a \
            str (\'x\', \'y\', \'z\').'
        raise MagpylibBadUserInput(msg)

def check_axis_format(axis):
    """
    - must be of shape (3,)
    - must not be (0,0,0)
    """
    msg = 'Bad axis input shape. Must be shape (3,).'
    if not axis.shape == (3,):
        raise MagpylibBadInputShape(msg)
    msg = 'Bad axis input. Cannot be (0,0,0).'
    if np.all(axis == 0):
        raise MagpylibBadUserInput(msg)


def check_degree_type(deg):
    """ degrees input must be bool
    """
    if not isinstance(deg, bool):
        msg = 'degrees input must be bool (True or False).'
        raise MagpylibBadUserInput(msg)



def check_scalar_type(inp, origin):
    """ scalar input must be int or float
    """
    if not isinstance(inp, (int, float)):
        msg = origin + ' input must be scalar (int or float).'
        raise MagpylibBadUserInput(msg)

def check_scalar_init(inp, origin):
    """ check if scalar input was initialized (former None)
    """
    if inp is None:
        msg = origin + ' input required.'
        raise MagpylibBadUserInput(msg)



def check_vector_type(inp, origin):
    """
    - vector input must be list, tuple or ndarray
    - return error msg with reference to origin
    """
    if not isinstance(inp, (list, tuple, np.ndarray)):
        msg = origin + ' input must be vector type (list, tuple or ndarray).'
        raise MagpylibBadUserInput(msg)

def check_vector_init(inp, origin):
    """
    - check if vector input was initialized (former None vetor)
    - return error msg with reference to origin
    """
    if None in inp:
        msg = origin + ' input required.'
        raise MagpylibBadUserInput(msg)

def check_vector_format(inp, shape, origin):
    """
    - check if vector input has correct format
    - return error msg with reference to origin
    """
    if not inp.shape == shape:
        msg = f'Bad {origin} input shape. Must be shape f{shape}.'
        raise MagpylibBadInputShape(msg)


def check_position_format(inp, origin):
    """
    - test if input has the correct shape (...,3)
    - return error msg with reference to origin
    """
    msg = f'Bad {origin} input shape. Must be (...,3).'
    if not inp.shape[-1] == 3:
        raise MagpylibBadInputShape(msg)


def check_vertex_format(inp):
    """
    - test if input has the correct shape (N,3) with N >= 2
    - return err msg
    """
    msg = 'Bad vertex input shape. Must be (N,3) with N>1'
    if not inp.shape[-1] == 3:
        raise MagpylibBadInputShape(msg)
    if not inp.ndim == 2:
        raise MagpylibBadInputShape(msg)
    if not inp.shape[0] > 1:
        raise MagpylibBadInputShape(msg)

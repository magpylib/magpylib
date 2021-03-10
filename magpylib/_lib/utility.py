""" some utility functions"""
from typing import Sequence
import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib._lib.exceptions import MagpylibBadUserInput
from magpylib import _lib

def rotobj_from_angax(angle: float, axis: np.ndarray) -> R:
    """ Create rot object from angle axis input.

    Args:
    - angle (float): angle in [rad]
    - axis (arr3): dimensionless axis

    Returns:
    - R: scipy rotation object
    """
    ang = float(angle)
    len_ax = np.linalg.norm(axis)
    if len_ax == 0:
        rotvec = np.zeros(3)
    else:
        rotvec = axis/len_ax*ang
    rotobj = R.from_rotvec(rotvec)

    return rotobj


def format_obj_input(objects: Sequence) -> list:
    """ tests and flattens potential input sources (sources, Collections, sequences)

    ### Args:
    - sources (sequence): input sources

    ### Returns:
    - list: flattened, ordered list f sources

    ### Info:
    - exits if invalid sources are given
    """
    # avoid circular imports
    Box = _lib.obj_classes.Box
    Cylinder = _lib.obj_classes.Cylinder
    Collection = _lib.obj_classes.Collection
    Sensor = _lib.obj_classes.Sensor

    obj_list = []
    for obj in objects:
        if isinstance(obj, (tuple, list)):
            obj_list += format_obj_input(obj) # recursive flattening
        elif isinstance(obj, Collection):
            obj_list += obj.sources
        elif isinstance(obj, (
                Box,
                Cylinder,
                Sensor)):
            obj_list += [obj]
        else:
            msg = 'Unknown input object type.'
            raise MagpylibBadUserInput(msg)

    return obj_list


def check_duplicates(src_list: Sequence) -> list:
    """ checks for and eliminates source duplicates in a list of sources

    ### Args:
    - src_list (list): list with source objects

    ### Returns:
    - list: src_list with duplicates removed
    """
    src_list_new = []
    for src in src_list:
        if src not in src_list_new:
            src_list_new += [src]

    if len(src_list_new) != len(src_list):
        print('WARNING: Eliminating duplicate sources')

    return src_list_new


def test_path_format(inp):
    """ check if each object path has same length
    of obj.pos and obj.rot

    Parameters
    ----------
    inp: single BaseGeo or list of BaseGeo objects

    Returns
    -------
    no return
    """
    # pylint: disable=protected-access
    if not isinstance(inp,list):
        inp = [inp]
    result = all(len(obj._pos) == len(obj._rot) for obj in inp)

    if not result:
        msg = 'Bad path format (rot-pos with different lengths)'
        raise MagpylibBadUserInput(msg)


def get_good_path_length(obj_list: list) -> bool:
    """ check if all paths have good format and
    are either length 1 or same length m
    exits if

    Parameters
    ----------
    obj_list: list of BaseGeo objects

    Returns
    -------
    path length m
    """
    # pylint: disable=protected-access
    test_path_format(obj_list)

    path_lenghts = [len(obj._pos) for obj in obj_list]
    m = max(path_lenghts)

    if all(pl in (1,m) for pl in path_lenghts):
        return m

    msg = 'Bad path format (different path lengths !=1 detected)'
    raise MagpylibBadUserInput(msg)


def all_same(lst:list)->bool:
    """ test if all list entries are the same
    """
    return lst[1:]==lst[:-1]


def only_allowed_src_types(src_list):
    """
    return only allowed objects. Throw a warning when something is eliminated.
    """
    Box = _lib.obj_classes.Box
    Cylinder = _lib.obj_classes.Cylinder
    new_list = []
    for src in src_list:
        if isinstance(src, (Box, Cylinder)):
            new_list += [src]
        else:
            print(f'Warning, cannot add {src.__repr__()} to Collection.')
    return new_list

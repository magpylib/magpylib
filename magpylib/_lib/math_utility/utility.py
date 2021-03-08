""" some utility functions"""
from typing import Sequence
import numpy as np
from scipy.spatial.transform import Rotation as R
import magpylib as mag3
from magpylib._lib.exceptions import MagpylibBadUserInput


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
    obj_list = []
    for obj in objects:
        if isinstance(obj, (tuple, list)):
            obj_list += format_obj_input(obj) # recursive flattening
        elif isinstance(obj, mag3.Collection):
            obj_list += obj.sources
        elif isinstance(obj, (
                mag3.magnet.Box,  #avoid circ imports
                mag3.magnet.Cylinder,
                mag3.Sensor)):
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
    src_set = set(src_list)
    if len(src_set) != len(src_list):
        print('WARNING: Eliminating duplicate sources')
        src_list = list(src_set)
    return src_list


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

""" some utility functions"""

import sys
from typing import Sequence
import numpy as np
from scipy.spatial.transform import Rotation as R
import magpylib as mag3


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


def format_src_input(sources: Sequence) -> list:
    """ tests and flattens potential input sources (sources, Collections, sequences)

    ### Args:
    - sources (sequence): input sources

    ### Returns:
    - list: flattened, ordered list f sources

    ### Info:
    - exits if invalid sources are given
    """

    src_list = []
    for src in sources:
        if isinstance(src, (tuple, list)):
            src_list += format_src_input(src) # recursive flattening
        elif isinstance(src, mag3.Collection):
            src_list += src.sources
        elif isinstance(src, (
                mag3.magnet.Box,  #avoid circ imports
                mag3.magnet.Cylinder,
                mag3.Sensor)):
            src_list += [src]
        else:
            sys.exit('ERROR: format_src_input() - bad sources input')

    return src_list


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


def good_path_format(inp: list) -> bool:
    """ check if each object path has same length
    of obj.pos and obj.rot

    Parameters
    ----------
    inp: single BaseGeo or list of BaseGeo objects

    Returns
    -------
    True if all object have good path format
    """
    # pylint: disable=protected-access
    if not isinstance(inp,list):
        inp = [inp]
    return all([len(obj._pos) == len(obj._rot) for obj in inp])


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
    if not good_path_format(obj_list):
        sys.exit('ERROR (get_good_path_length): Bad path format')

    path_lenghts = [len(obj._pos) for obj in obj_list]
    m = max(path_lenghts)

    if all([pl in (1,m) for pl in path_lenghts]):
        return m

    sys.exit('ERROR (get_good_path_length): Bad path lengths')


def check_allowed_keys(allowed_keys, kwargs, func_name):
    """ Thows a warning if kwargs contains a key that
        is not in allowed_keys

    Parameters:
    -----------
    allowed_keys (list): list of allowed keys

    kwargs (dict): input dictionary

    func_name (string): function name to throw proper warning
    """
    keys = kwargs.keys()
    complement = [i for i in keys if i not in allowed_keys]
    if complement:
        print('WARNING: ' + func_name + ' - unknown input kwarg, ', complement)


def all_same(lst:list)->bool:
    """ test if all list entries are the same
    """
    return lst[1:]==lst[:-1]

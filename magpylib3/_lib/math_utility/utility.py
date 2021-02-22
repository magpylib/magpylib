""" some utility functions"""

from operator import pos
import sys
from typing import Sequence
import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib3 import _lib



def rotobj_from_angax(angle: float, axis: np.ndarray) -> R: 
    """ Create rot object from angle axis input.

    Args:
    - angle (float): angle in [rad]
    - axis (arr3): dimensionless axis

    Returns:
    - R: scipy rotation object
    """
    
    ang = np.float(angle)
    Lax = np.linalg.norm(axis)
    if Lax == 0:
        rotvec = np.zeros(3)
    else:
        rotvec = axis/Lax*ang
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
    for s in sources:
        if isinstance(s, (tuple,list)):
            src_list += format_src_input(s) # recursive flattening
        elif isinstance(s,_lib.obj_classes.Collection):
            src_list += s._sources
        elif isinstance(s,(
                _lib.obj_classes.Box,  #avoid circ imports
                _lib.obj_classes.Cylinder)):
            src_list += [s]
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


def same_path_length(obj_list: list) -> None:
    """ check if all objects in obj_list have same path length

    Parameters
    ----------
    obj_list: list of source objects

    Returns
    -------
    True if 
    """
    lp = len(obj_list[0]._pos)
    pos_check = all([len(obj._pos)==lp for obj in obj_list])
    rot_check = all([len(obj._rot.as_quat())==lp for obj in obj_list])

    return pos_check and rot_check
""" some utility functions"""

import sys
from typing import Sequence
import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib3 import _lib



def rotobj_from_angle_axis(angle: float, axis: np.ndarray) -> R: 
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


# def rotobj_from_rot_input(rot: Union[None, Sequence, R]) -> R: # pylint: disable=unsubscriptable-object
#     """ Create rot object from rot input.

#     Args:
#     - rot (None, Sequence, R):  None generates a unit rotation, (angle,axis) or rotation object type.

#     Returns:
#     - R: scipy rotation object
#     """
    
#     # default input, set to unit orient
#     if rot is None:
#         return R.from_rotvec((0,0,0))
    
#     # rotation object input
#     if isinstance(rot,R):
#         rotobj = rot
    
#     # standard angle-axis input
#     else:
#         ang = np.float(rot[0])/180*np.pi
#         ax = np.array(rot[1],dtype=np.float64)
#         Lax = np.linalg.norm(ax)
#         if Lax == 0:
#             rotvec = np.zeros(3)
#         else:
#             rotvec = ax/Lax*ang
#         rotobj = R.from_rotvec(rotvec)
    
#     return rotobj


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
            print('ERROR format_src_input: bad sources input')
            sys.exit()
    
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

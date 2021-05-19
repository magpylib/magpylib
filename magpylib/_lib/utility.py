""" some utility functions"""
from typing import Sequence
import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib._lib.exceptions import MagpylibBadUserInput
from magpylib import _lib


def format_getBH_class_inputs(inp):
    """
    allow *inputs "src", "src, src" but also "[src, src]"
    """
    if len(inp)==1:
        return inp[0]
    return list(inp)


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
    Sphere = _lib.obj_classes.Sphere
    Dipole = _lib.obj_classes.Dipole
    Circular = _lib.obj_classes.Circular

    obj_list = []
    for obj in objects:
        if isinstance(obj, (tuple, list)):
            obj_list += format_obj_input(obj) # recursive flattening
        elif isinstance(obj, Collection):
            obj_list += obj.sources
        elif isinstance(obj, (
                Box,
                Cylinder,
                Sphere,
                Sensor,
                Dipole,
                Circular)):
            obj_list += [obj]
        else:
            msg = 'Unknown input object type.'
            raise MagpylibBadUserInput(msg)

    return obj_list


def format_src_inputs(sources: list) -> list:
    """
    checks if sources format is 1D [src1, src2, src3, col1, ...]
    returns a flattened list of sources

    ### Args:
    - sources

    ### Returns:
    - list: flattened, ordered list of sources

    ### Info:
    - raises an error if sources format is bad
    """
    # avoid circular imports
    src_class_types = (
        _lib.obj_classes.Box,
        _lib.obj_classes.Cylinder,
        _lib.obj_classes.Sphere,
        _lib.obj_classes.Dipole,
        _lib.obj_classes.Circular)
    Collection = _lib.obj_classes.Collection

    src_list = []
    for src in sources:
        if isinstance(src, Collection):
            src_list += src.sources
        elif isinstance(src, src_class_types):
            src_list += [src]
        else:
            raise MagpylibBadUserInput('Unknown input object type.')

    return src_list


def format_obs_inputs(observers) -> list:
    """
    checks if observer input is one of the following:
        - bare Sensor
        - tupe or ndarray
        - list
            - list of sensor, tuple, sensor, ...
            - list of positions

    ### Args:
    - observers

    ### Returns:
    - list of sensors

    ### Info:
    - raises an error if sources format is bad
    """

    # import type, avoid circular imports
    Sensor = _lib.obj_classes.Sensor

    # case 1: sensor
    if isinstance(observers, Sensor):
        return [observers]
    # case 2: tuple or ndarray of possitions
    if isinstance(observers, (tuple,np.ndarray)):
        return [Sensor(pos_pix=observers)]
    #case 3: list
    if isinstance(observers, list):
        # case 3a: [sens, possis, sens, sens, ...]
        if any(isinstance(obs,Sensor) for obs in observers):
            sensors = []
            for obs in observers:
                if isinstance(obs, Sensor):
                    sensors += [obs]
                elif isinstance(obs, (list, tuple, np.ndarray)):
                    sensors += [Sensor(pos_pix=obs)]
                else:
                    raise MagpylibBadUserInput('Unknown input object type.')
        # case 3b: list of positions
        else:
            sensors = [Sensor(pos_pix=observers)]

    return sensors


def check_static_sensor_orient(sensors):
    """ test which sensors have a static orientation
    """
    #pylint: disable=protected-access
    static_sensor_rot = []
    for sens in sensors:
        if len(sens._pos)==1:           # no sensor path (sensor is static)
            static_sensor_rot += [True]
        else:                           # there is a sensor path
            rot = sens.rot.as_quat()
            if np.all(rot == rot[0]):          # path with static orient (e.g. translation)
                static_sensor_rot += [True]
            else:                              # sensor rotation changes along path
                static_sensor_rot += [False]
    return static_sensor_rot



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
    Sphere = _lib.obj_classes.Sphere
    Dipole = _lib.obj_classes.Dipole
    Circular = _lib.obj_classes.Circular
    new_list = []
    for src in src_list:
        if isinstance(src, (Box, Cylinder, Sphere, Dipole, Circular)):
            new_list += [src]
        else:
            print(f'Warning, cannot add {src.__repr__()} to Collection.')
    return new_list

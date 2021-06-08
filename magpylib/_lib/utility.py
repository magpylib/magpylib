""" some utility functions"""
from typing import Sequence
import numpy as np
#from scipy.spatial.transform import Rotation as R
from magpylib._lib.exceptions import MagpylibBadUserInput
from magpylib import _lib
from magpylib._lib.config import Config
from magpylib._lib.input_checks import check_position_format

def format_star_input(inp):
    """
    *inputs are always wrapped in tuple. Formats *inputs of form "src", "src, src"
    but also "[src, src]" or ""(src,src") so that 1D lists/tuples come out.
    """
    if len(inp)==1:
        return inp[0]
    return list(inp)


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
    Line = _lib.obj_classes.Line

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
                Circular,
                Line)):
            obj_list += [obj]
        else:
            msg = 'Unknown input object type.'
            raise MagpylibBadUserInput(msg)

    return obj_list


def format_src_inputs(sources) -> list:
    """
    - input: allow only bare src objects or 1D lists/tuple of src and col
    - out: sources, src_list

    ### Args:
    - sources

    ### Returns:
    - sources: ordered list of sources
    - src_list: ordered list of sources with flattened collections

    ### Info:
    - raises an error if sources format is bad
    """
    # avoid circular imports
    src_class_types = (
        _lib.obj_classes.Box,
        _lib.obj_classes.Cylinder,
        _lib.obj_classes.Sphere,
        _lib.obj_classes.Dipole,
        _lib.obj_classes.Circular,
        _lib.obj_classes.Line)
    Collection = _lib.obj_classes.Collection

    # bare source -> list
    if not isinstance(sources, (list,tuple)):
        sources = [sources]

    # flatten collections
    src_list = []
    for src in sources:
        if isinstance(src, Collection):
            src_list += src.sources
        elif isinstance(src, src_class_types):
            src_list += [src]
        else:
            raise MagpylibBadUserInput('Unknown source type of input.')

    return list(sources), src_list


def format_obs_inputs(observers) -> list:
    """
    checks if observer input is one of the following:
        - case1:  bare Sensor
        - case2: ndarray (can only be possis)
        - case3: list or tuple
            - 3a: list/tuple of sensor, tuple, sensor, list, ...
            - 3b: list of positions

    returns an ordered 1D list of sensors

    ### Info:
    - raises an error if sources format is bad
    """
    # pylint: disable=too-many-branches
    # import type, avoid circular imports
    Sensor = _lib.obj_classes.Sensor

    msg = 'Unknown observer input type. Must be Sensor, list, tuple or ndarray'

    # case 1: sensor
    if isinstance(observers, Sensor):
        return [observers]

    # case 2: ndarray of possitions
    if isinstance(observers, np.ndarray):
        if Config.CHECK_INPUTS:
            check_position_format(observers, 'observer position')
        return [Sensor(pos_pix=observers)]

    #case 3: list or tuple
    if isinstance(observers, (list, tuple)):

        # case 3a: [sens, possis, sens, sens, ...]
        if any(isinstance(obs, Sensor) for obs in observers):
            sensors = []
            for obs in observers:
                if isinstance(obs, Sensor):
                    sensors += [obs]
                elif isinstance(obs, (list, tuple, np.ndarray)):
                    if Config.CHECK_INPUTS:
                        check_position_format(np.array(obs), 'observer position')
                    sensors += [Sensor(pos_pix=obs)]
                else:
                    raise MagpylibBadUserInput(msg)

        # case 3b: list/tuple of positions
        else:
            if Config.CHECK_INPUTS:
                check_position_format(np.array(observers), 'observer position')
            sensors = [Sensor(pos_pix=observers)]
    else:
        raise MagpylibBadUserInput(msg)

    return sensors


def check_static_sensor_orient(sensors):
    """ test which sensors have a static orientation
    """
    #pylint: disable=protected-access
    static_sensor_rot = []
    for sens in sensors:
        if len(sens._position)==1:           # no sensor path (sensor is static)
            static_sensor_rot += [True]
        else:                           # there is a sensor path
            rot = sens.orientation.as_quat()
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
    result = all(len(obj._position) == len(obj._orientation) for obj in inp)

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

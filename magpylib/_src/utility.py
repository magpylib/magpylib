""" some utility functions"""
from typing import Sequence
import numpy as np
#from scipy.spatial.transform import Rotation as R
from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib import _src
from magpylib._src.config import Config
from magpylib._src.input_checks import check_position_format


def close(arg1: np.ndarray, arg2: np.ndarray) -> np.ndarray:
    """
    determine if arg1 and arg2 lie close to each other
    input: ndarray, shape (n,) or numpy-interpretable scalar
    output: ndarray, dtype=bool
    """
    EDGESIZE = Config.EDGESIZE
    return np.isclose(arg1, arg2, rtol=0, atol=EDGESIZE)


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
    # pylint: disable=protected-access

    obj_list = []
    for obj in objects:
        if isinstance(obj, (tuple, list)):
            obj_list += format_obj_input(obj) # recursive flattening
        else:
            try:
                if obj._object_type == 'Collection':
                    obj_list += obj.sources
                elif obj._object_type in (
                    'Cuboid',
                    'Cylinder',
                    'CylinderSegment',
                    'Sphere',
                    'Sensor',
                    'Dipole',
                    'Circular',
                    'Line'):
                    obj_list += [obj]
            except Exception as error:
                raise MagpylibBadUserInput('Unknown input object type.') from error

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
    # pylint: disable=protected-access

    src_class_types = (
        'Cuboid', 'Cylinder', 'CylinderSegment', 'Sphere', 'Dipole', 'Circular', 'Line')

    # if bare source make into list
    if not isinstance(sources, (list,tuple)):
        sources = [sources]

    # flatten collections
    src_list = []
    try:
        for src in sources:
            if src._object_type == 'Collection':
                src_list += src.sources
            elif src._object_type in src_class_types:
                src_list += [src]
            else:
                raise MagpylibBadUserInput
    except Exception as error:
        raise MagpylibBadUserInput('Unknown source type of input.') from error

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
    Sensor = _src.obj_classes.Sensor

    msg = 'Unknown observer input type. Must be Sensor, list, tuple or ndarray'

    # case 1: sensor
    if isinstance(observers, Sensor):
        return [observers]

    # case 2: ndarray of possitions
    if isinstance(observers, np.ndarray):
        if Config.CHECK_INPUTS:
            check_position_format(observers, 'observer position')
        return [Sensor(pixel=observers)]

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
                    sensors += [Sensor(pixel=obs)]
                else:
                    raise MagpylibBadUserInput(msg)

        # case 3b: list/tuple of positions
        else:
            if Config.CHECK_INPUTS:
                check_position_format(np.array(observers), 'observer position')
            sensors = [Sensor(pixel=observers)]
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
    return only allowed objects - e.g. no sensors. Throw a warning when something is eliminated.
    """
    # pylint: disable=protected-access

    src_class_types = ('Cuboid', 'Cylinder', 'CylinderSegment', 'Sphere', 'Dipole',
        'Circular', 'Line')
    new_list = []
    for src in src_list:
        if src._object_type in src_class_types:
            new_list += [src]
        else:
            if Config.CHECK_INPUTS:
                print(f'Warning, cannot add {src.__repr__()} to Collection.')
    return new_list

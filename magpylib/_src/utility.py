""" some utility functions"""
from typing import Sequence
import numpy as np

# from scipy.spatial.transform import Rotation as R
from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib import _src
from magpylib._src.defaults.defaults_classes import default_settings as Config
from magpylib._src.input_checks import check_position_format

LIBRARY_SOURCES = (
    "Cuboid",
    "Cylinder",
    "CylinderSegment",
    "Sphere",
    "Dipole",
    "Loop",
    "Line",
    "CustomSource",
)

LIBRARY_SENSORS = ("Sensor",)

ALLOWED_SOURCE_MSG = f"""Sources must be either
- one of {LIBRARY_SOURCES}
- Collection with at least one of the above
- 1D list thereof"""

ALLOWED_OBSERVER_MSG = """Observers must be either
- array_like positions of shape (N1, N2, ..., 3)
- Sensor object
- Collection with at least one Sensor
- 1D list thereof"""

ALLOWED_SENSORS_MSG = """Sensors must be either
- Sensor object
- Collection with at least one Sensor
- 1D list thereof"""


def wrong_obj_msg(*objs, allow="sources"):
    """return error message for wrong object type provided"""
    assert len(objs) <= 1, "only max one obj allowed"
    allowed = allow.split("+")
    prefix = "No" if len(allowed) == 1 else "Bad"
    msg = f"{prefix} {'/'.join(allowed)} provided"
    if "sources" in allowed:
        msg += "\n" + ALLOWED_SOURCE_MSG
    if "observers" in allowed:
        msg += "\n" + ALLOWED_OBSERVER_MSG
    if "sensors" in allowed:
        msg += "\n" + ALLOWED_SENSORS_MSG
    if objs:
        obj = objs[0]
        msg += f"\nreceived {obj!r} of type {type(obj).__name__!r} instead." ""
    return msg


def close(arg1: np.ndarray, arg2: np.ndarray) -> np.ndarray:
    """
    determine if arg1 and arg2 lie close to each other
    input: ndarray, shape (n,) or numpy-interpretable scalar
    output: ndarray, dtype=bool
    """
    EDGESIZE = Config.edgesize
    return np.isclose(arg1, arg2, rtol=0, atol=EDGESIZE)


def format_star_input(inp):
    """
    *inputs are always wrapped in tuple. Formats *inputs of form "src", "src, src"
    but also "[src, src]" or ""(src,src") so that 1D lists/tuples come out.
    """
    if len(inp) == 1:
        return inp[0]
    return list(inp)


def format_obj_input(
    *objects: Sequence, allow="sources+sensors", warn=True) -> list:
    """tests and flattens potential input sources (sources, Collections, sequences)

    ### Args:
    - sources (sequence): input sources

    ### Returns:
    - list: flattened, ordered list f sources

    ### Info:
    - exits if invalid sources are given
    """
    # pylint: disable=protected-access

    obj_list = []
    flatten_collection = not 'collections' in allow.split("+")
    for obj in objects:
        if isinstance(obj, (tuple, list)):
            obj_list += format_obj_input(
                *obj, allow=allow, warn=warn
            )  # recursive flattening
        else:
            try:
                if obj._object_type == "Collection":
                    if flatten_collection:
                        obj_list += obj.children
                    else:
                        obj_list += [obj]
                elif obj._object_type in list(LIBRARY_SOURCES) + list(LIBRARY_SENSORS):
                    obj_list += [obj]
            except Exception as error:
                raise MagpylibBadUserInput(wrong_obj_msg(obj, allow=allow)) from error
    obj_list = filter_objects(obj_list, allow=allow, warn=False)
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

    # if bare source make into list
    if not isinstance(sources, (list, tuple)):
        sources = [sources]
    # flatten collections
    src_list = []
    if not sources:
        raise MagpylibBadUserInput(wrong_obj_msg(allow="sources"))
    for src in sources:
        obj_type = getattr(src, "_object_type", None)
        if obj_type == "Collection":
            if not src.sources:
                raise MagpylibBadUserInput(wrong_obj_msg(src, allow="sources"))
            src_list += src.sources
        elif obj_type in LIBRARY_SOURCES:
            src_list += [src]
        else:
            raise MagpylibBadUserInput(wrong_obj_msg(src, allow="sources"))
    return list(sources), src_list


def format_obs_inputs(observers) -> list:
    """
    checks if observer input is one of the following:
        - case1: bare Sensor
        - case2: ndarray (can only be possis)
        - case3: Collection with sensors

    returns an ordered 1D list of sensors

    ### Info:
    - raises an error if sources format is bad
    """
    # pylint: disable=too-many-branches
    # import type, avoid circular imports
    Sensor = _src.obj_classes.Sensor

    if not isinstance(observers, (list, tuple, np.ndarray)):
        observers = (observers,)
    elif len(observers) == 0:
        raise MagpylibBadUserInput(wrong_obj_msg(allow="observers"))
    elif np.isscalar(observers[0]):
        observers = (observers,)

    sensors = []
    for obs in observers:
        # pylint: disable=protected-access
        # case 1: sensor
        if isinstance(obs, Sensor):
            sensors.append(obs)

        # case 2: ndarray of positions
        elif isinstance(obs, (list, tuple, np.ndarray)):
            if Config.checkinputs:
                check_position_format(np.array(obs), "observer position")
            sensors.append(Sensor(pixel=obs))
        elif getattr(obs, "_object_type", "") == "Collection":
            if not obs.sensors:
                raise MagpylibBadUserInput(wrong_obj_msg(obs, allow="observers"))
            sensors.extend(obs.sensors)
        else:
            raise MagpylibBadUserInput(wrong_obj_msg(obs, allow="observers"))

    return sensors


def check_static_sensor_orient(sensors):
    """test which sensors have a static orientation"""
    # pylint: disable=protected-access
    static_sensor_rot = []
    for sens in sensors:
        if len(sens._position) == 1:  # no sensor path (sensor is static)
            static_sensor_rot += [True]
        else:  # there is a sensor path
            rot = sens.orientation.as_quat()
            if np.all(rot == rot[0]):  # path with static orient (e.g. translation)
                static_sensor_rot += [True]
            else:  # sensor rotation changes along path
                static_sensor_rot += [False]
    return static_sensor_rot


def check_duplicates(obj_list: Sequence) -> list:
    """checks for and eliminates source duplicates in a list of sources

    ### Args:
    - obj_list (list): list with source objects

    ### Returns:
    - list: obj_list with duplicates removed
    """
    obj_list_new = []
    for src in obj_list:
        if src not in obj_list_new:
            obj_list_new += [src]

    if len(obj_list_new) != len(obj_list):
        print("WARNING: Eliminating duplicates")

    return obj_list_new


def test_path_format(inp):
    """check if each object path has same length
    of obj.pos and obj.rot

    Parameters
    ----------
    inp: single BaseGeo or list of BaseGeo objects

    Returns
    -------
    no return
    """
    # pylint: disable=protected-access
    if not isinstance(inp, list):
        inp = [inp]
    result = all(len(obj._position) == len(obj._orientation) for obj in inp)

    if not result:
        msg = "Bad path format (rot-pos with different lengths)"
        raise MagpylibBadUserInput(msg)


def all_same(lst: list) -> bool:
    """test if all list entries are the same"""
    return lst[1:] == lst[:-1]


def filter_objects(obj_list, allow="sources+sensors", warn=True):
    """
    return only allowed objects - e.g. no sensors. Throw a warning when something is eliminated.
    """
    # pylint: disable=protected-access
    allowed_list = []
    for allowed in allow.split("+"):
        if allowed == "sources":
            allowed_list.extend(LIBRARY_SOURCES)
        elif allowed == "sensors":
            allowed_list.extend(LIBRARY_SENSORS)
        elif allowed =='collections':
            allowed_list.extend(['Collection'])
    new_list = []
    for obj in obj_list:
        if obj._object_type in allowed_list:
            new_list += [obj]
        else:
            if Config.checkinputs and warn:
                print(f"Warning, cannot add {obj.__repr__()} to Collection.")
    return new_list

def adjust_start(start, lenop):
    """
    change start to a value inside of [0,lenop], i.e. inside of the
    old path.
    """
    if start == "append":
        start = lenop
    elif start < 0:
        start += lenop

    # fix out-of-bounds start values
    if start < 0:
        start = 0
        if Config.checkinputs:
            print("Warning: start out of path bounds. Setting start=0.")
    elif start > lenop:
        start = lenop
        if Config.checkinputs:
            print(f"Warning: start out of path bounds. Setting start={lenop}.")

    return start

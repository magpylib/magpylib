""" some utility functions"""
# pylint: disable=import-outside-toplevel
# pylint: disable=cyclic-import
# import numbers
from math import log10
from typing import Sequence

import numpy as np

from magpylib._src.exceptions import MagpylibBadUserInput


def get_allowed_sources_msg():
    "Return allowed source message"

    srcs = list(get_registered_sources())
    return f"""Sources must be either
- one of type {srcs}
- Collection with at least one of the above
- 1D list of the above
- string {srcs}"""


ALLOWED_OBSERVER_MSG = """Observers must be either
- array_like positions of shape (N1, N2, ..., 3)
- Sensor object
- Collection with at least one Sensor
- 1D list of the above"""

ALLOWED_SENSORS_MSG = """Sensors must be either
- Sensor object
- Collection with at least one Sensor
- 1D list of the above"""


def wrong_obj_msg(*objs, allow="sources"):
    """return error message for wrong object type provided"""
    assert len(objs) <= 1, "only max one obj allowed"
    allowed = allow.split("+")
    prefix = "No" if len(allowed) == 1 else "Bad"
    msg = f"{prefix} {'/'.join(allowed)} provided"
    if "sources" in allowed:
        msg += "\n" + get_allowed_sources_msg()
    if "observers" in allowed:
        msg += "\n" + ALLOWED_OBSERVER_MSG
    if "sensors" in allowed:
        msg += "\n" + ALLOWED_SENSORS_MSG
    if objs:
        obj = objs[0]
        msg += f"\nreceived {obj!r} of type {type(obj).__name__!r} instead." ""
    return msg


def format_star_input(inp):
    """
    *inputs are always wrapped in tuple. Formats *inputs of form "src", "src, src"
    but also "[src, src]" or ""(src,src") so that 1D lists/tuples come out.
    """
    if len(inp) == 1:
        return inp[0]
    return list(inp)


def format_obj_input(*objects: Sequence, allow="sources+sensors", warn=True) -> list:
    """tests and flattens potential input sources (sources, Collections, sequences)
    ### Args:
    - sources (sequence): input sources
    ### Returns:
    - list: flattened, ordered list of sources
    ### Info:
    - exits if invalid sources are given
    """
    from magpylib._src.obj_classes.class_BaseExcitations import BaseSource
    from magpylib._src.obj_classes.class_Sensor import Sensor

    obj_list = []
    flatten_collection = not "collections" in allow.split("+")
    for obj in objects:
        try:
            if isinstance(obj, (BaseSource, Sensor)):
                obj_list += [obj]
            else:
                if flatten_collection or isinstance(obj, (list, tuple)):
                    obj_list += format_obj_input(
                        *obj,
                        allow=allow,
                        warn=warn,
                    )  # recursive flattening
                else:
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

    from magpylib._src.obj_classes.class_BaseExcitations import BaseSource
    from magpylib._src.obj_classes.class_Collection import Collection

    # store all sources here
    src_list = []

    # if bare source make into list
    if not isinstance(sources, (list, tuple)):
        sources = [sources]

    if not sources:
        raise MagpylibBadUserInput(wrong_obj_msg(allow="sources"))

    for src in sources:
        if isinstance(src, Collection):
            child_sources = format_obj_input(src, allow="sources")
            if not child_sources:
                raise MagpylibBadUserInput(wrong_obj_msg(src, allow="sources"))
            src_list += child_sources
        elif isinstance(src, BaseSource):
            src_list += [src]
        else:
            raise MagpylibBadUserInput(wrong_obj_msg(src, allow="sources"))
    return list(sources), src_list


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


def filter_objects(obj_list, allow="sources+sensors", warn=True):
    """
    return only allowed objects - e.g. no sensors. Throw a warning when something is eliminated.
    """
    from magpylib._src.obj_classes.class_BaseExcitations import BaseSource
    from magpylib._src.obj_classes.class_Sensor import Sensor
    from magpylib._src.obj_classes.class_Collection import Collection

    # select wanted
    allowed_classes = ()
    if "sources" in allow.split("+"):
        allowed_classes += (BaseSource,)
    if "sensors" in allow.split("+"):
        allowed_classes += (Sensor,)
    if "collections" in allow.split("+"):
        allowed_classes += (Collection,)
    new_list = []
    for obj in obj_list:
        if isinstance(obj, allowed_classes):
            new_list += [obj]
        else:
            if warn:
                print(f"Warning, cannot add {obj!r} to Collection.")
    return new_list


_UNIT_PREFIX = {
    -24: "y",  # yocto
    -21: "z",  # zepto
    -18: "a",  # atto
    -15: "f",  # femto
    -12: "p",  # pico
    -9: "n",  # nano
    -6: "µ",  # micro
    -3: "m",  # milli
    0: "",
    3: "k",  # kilo
    6: "M",  # mega
    9: "G",  # giga
    12: "T",  # tera
    15: "P",  # peta
    18: "E",  # exa
    21: "Z",  # zetta
    24: "Y",  # yotta
}


def unit_prefix(number, unit="", precision=3, char_between="") -> str:
    """
    displays a number with given unit and precision and uses unit prefixes for the exponents from
    yotta (y) to Yocto (Y). If the exponent is smaller or bigger, falls back to scientific notation.
    Parameters
    ----------
    number : int, float
        can be any number
    unit : str, optional
        unit symbol can be any string, by default ""
    precision : int, optional
        gives the number of significant digits, by default 3
    char_between : str, optional
        character to insert between number of prefix. Can be " " or any string, if a space is wanted
        before the unit symbol , by default ""
    Returns
    -------
    str
        returns formatted number as string
    """
    digits = int(log10(abs(number))) // 3 * 3 if number != 0 else 0
    prefix = _UNIT_PREFIX.get(digits, "")

    if prefix == "":
        digits = 0
    new_number_str = f"{number / 10 ** digits:.{precision}g}"
    return f"{new_number_str}{char_between}{prefix}{unit}"


def add_iteration_suffix(name):
    """
    adds iteration suffix. If name already ends with an integer it will continue iteration
    examples:
        'col' -> 'col_01'
        'col' -> 'col_01'
        'col1' -> 'col2'
        'col_02' -> 'col_03'
    """
    # pylint: disable=import-outside-toplevel
    import re

    m = re.search(r"\d+$", name)
    n = "00"
    endstr = None
    midchar = "_" if name[-1] != "_" else ""
    if m is not None:
        midchar = ""
        n = m.group()
        endstr = -len(n)
    name = f"{name[:endstr]}{midchar}{int(n)+1:0{len(n)}}"
    return name


def cart_to_cyl_coordinates(observer):
    """
    cartesian observer positions to cylindrical coordinates
    observer: ndarray, shape (n,3)
    """
    x, y, z = observer.T
    r, phi = np.sqrt(x**2 + y**2), np.arctan2(y, x)
    return r, phi, z


def cyl_field_to_cart(phi, Br, Bphi=None):
    """
    transform Br,Bphi to Bx, By
    """
    if Bphi is not None:
        Bx = Br * np.cos(phi) - Bphi * np.sin(phi)
        By = Br * np.sin(phi) + Bphi * np.cos(phi)
    else:
        Bx = Br * np.cos(phi)
        By = Br * np.sin(phi)

    return Bx, By


def rec_obj_remover(parent, child):
    """remove known child from parent collection"""
    # pylint: disable=protected-access
    from magpylib._src.obj_classes.class_Collection import Collection

    for obj in parent:
        if obj == child:
            parent._children.remove(child)
            parent._update_src_and_sens()
            return True
        if isinstance(obj, Collection):
            if rec_obj_remover(obj, child):
                break
    return None


def get_subclasses(cls, recursive=False):
    """Return a dictionary of subclasses by name,"""
    sub_cls = {}
    for class_ in cls.__subclasses__():
        sub_cls[class_.__name__] = class_
        if recursive:
            sub_cls.update(get_subclasses(class_, recursive=recursive))
    return sub_cls


def get_registered_sources():
    """Return all registered sources"""
    # pylint: disable=import-outside-toplevel
    from magpylib._src.obj_classes.class_BaseExcitations import (
        BaseCurrent,
        BaseMagnet,
        BaseSource,
    )

    return {
        k: v
        for k, v in get_subclasses(BaseSource, recursive=True).items()
        if not v in (BaseCurrent, BaseMagnet, BaseSource)
    }


def is_notebook() -> bool:  # pragma: no cover
    """Check if execution is within a IPython environment"""
    # pylint: disable=import-outside-toplevel
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        if shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def open_animation(filepath, embed=True):
    """Display video or gif file using tkinter or IPython"""
    # pylint: disable=import-outside-toplevel
    if is_notebook():
        if filepath.endswith(".gif"):
            from IPython.display import Image as IPyImage, display

            display(IPyImage(data=filepath, embed=embed))
        elif filepath.endswith(".mp4"):
            from IPython.display import Video, display

            display(Video(data=filepath, embed=embed))
        else:  # pragma: no cover
            raise TypeError("Filetype not supported, only 'mp4 or 'gif' allowed")
    else:
        import webbrowser

        webbrowser.open(filepath)

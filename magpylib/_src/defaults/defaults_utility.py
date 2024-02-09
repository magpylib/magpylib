"""utilities for creating property classes"""
# pylint: disable=too-many-branches
import collections.abc
import re
from copy import deepcopy
from functools import lru_cache

import numpy as np
import param
from matplotlib.colors import CSS4_COLORS as mcolors

from magpylib._src.defaults.defaults_values import DEFAULTS


COLORS_SHORT_TO_LONG = {
    "r": "red",
    "g": "green",
    "b": "blue",
    "y": "yellow",
    "m": "magenta",
    "c": "cyan",
    "k": "black",
    "w": "white",
}


class _DefaultType:
    """Special keyword value.

    The instance of this class may be used as the default value assigned to a
    keyword if no other obvious default (e.g., `None`) is suitable,

    """

    __instance = None

    def __new__(cls):
        # ensure that only one instance exists
        if not cls.__instance:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __repr__(self):  # pragma: no cover
        return "<default>"


_DefaultValue = _DefaultType()


def get_defaults_dict(arg=None, flatten=False, separator=".") -> dict:
    """Return default dict or sub-dict based on `arg` (e.g. get_default_dict('display.style')).

    Returns
    -------
    dict
        default sub dict

    flatten: bool
        If `True`, the nested dictionary gets flatten out with provided separator for the
        dictionary keys

    separator: str
        the separator to be used when flattening the dictionary. Only applies if
        `flatten=True`
    """

    dict_ = deepcopy(DEFAULTS_VALUES)
    if arg is not None:
        for v in arg.split(separator):
            dict_ = dict_[v]
    if flatten:
        dict_ = linearize_dict(dict_, separator=separator)
    return dict_


def update_nested_dict(d, u, same_keys_only=False, replace_None_only=False) -> dict:
    """updates recursively dictionary 'd' from  dictionary 'u'

    Parameters
    ----------
    d : dict
       dictionary to be updated
    u : dict
        dictionary to update with
    same_keys_only : bool, optional
        if `True`, only key found in `d` get updated and no new items are created,
        by default False
    replace_None_only : bool, optional
        if `True`, only key/value pair from `d`where `value=None` get updated from `u`,
        by default False

    Returns
    -------
    dict
        updated dictionary
    """
    if not isinstance(d, collections.abc.Mapping):
        if d is None or not replace_None_only:
            d = u.copy()
        return d
    new_dict = deepcopy(d)
    for k, v in u.items():
        if k in new_dict or not same_keys_only:
            if isinstance(v, collections.abc.Mapping):
                new_dict[k] = update_nested_dict(
                    new_dict.get(k, {}),
                    v,
                    same_keys_only=same_keys_only,
                    replace_None_only=replace_None_only,
                )
            elif new_dict.get(k, None) is None or not replace_None_only:
                if not same_keys_only or k in new_dict:
                    new_dict[k] = u[k]
    return new_dict


def magic_to_dict(kwargs, separator="_") -> dict:
    """decomposes recursively a dictionary with keys with underscores into a nested dictionary
    example : {'magnet_color':'blue'} -> {'magnet': {'color':'blue'}}
    see: https://plotly.com/python/creating-and-updating-figures/#magic-underscore-notation

    Parameters
    ----------
    kwargs : dict
        dictionary of keys to be decomposed into a nested dictionary

    separator: str, default='_'
        defines the separator to apply the magic parsing with
    Returns
    -------
    dict
        nested dictionary
    """
    assert isinstance(kwargs, dict), "kwargs must be a dictionary"
    assert isinstance(separator, str), "separator must be a string"
    new_kwargs = {}
    for k, v in kwargs.items():
        keys = k.split(separator)
        if len(keys) == 1:
            new_kwargs[keys[0]] = v
        else:
            val = {separator.join(keys[1:]): v}
            if keys[0] in new_kwargs and isinstance(new_kwargs[keys[0]], dict):
                new_kwargs[keys[0]].update(val)
            else:
                new_kwargs[keys[0]] = val
    for k, v in new_kwargs.items():
        if isinstance(v, dict):
            new_kwargs[k] = magic_to_dict(v, separator=separator)
    return new_kwargs


def linearize_dict(kwargs, separator=".") -> dict:
    """linearizes `kwargs` dictionary using the provided `separator
    Parameters
    ----------
    kwargs : dict
        dictionary of keys linearized into an flat dictionary

    separator: str, default='.'
        defines the separator to be applied on the final dictionary keys

    Returns
    -------
    dict
        flat dictionary with keys names using a separator

    Examples
    --------
    >>> from magpylib._src.defaults.defaults_utility import linearize_dict
    >>> from pprint import pprint
    >>> mydict = {
    ...     'line': {'width': 1, 'style': 'solid', 'color': None},
    ...     'marker': {'size': 1, 'symbol': 'o', 'color': None}
    ... }
    >>> flat_dict = linearize_dict(mydict, separator='.')
    >>> pprint(flat_dict)
    {'line.color': None,
     'line.style': 'solid',
     'line.width': 1,
     'marker.color': None,
     'marker.size': 1,
     'marker.symbol': 'o'}
    """
    assert isinstance(kwargs, dict), "kwargs must be a dictionary"
    assert isinstance(separator, str), "separator must be a string"
    dict_ = {}
    for k, v in kwargs.items():
        if isinstance(v, dict):
            d = linearize_dict(v, separator=separator)
            for key, val in d.items():
                dict_[f"{k}{separator}{key}"] = val
        else:
            dict_[k] = v
    return dict_


@lru_cache(maxsize=1000)
def color_validator(color_input, allow_None=True, parent_name=""):
    """validates color inputs based on chosen `backend', allows `None` by default.

    Parameters
    ----------
    color_input : str
        color input as string
    allow_None : bool, optional
        if `True` `color_input` can be `None`, by default True
    parent_name : str, optional
        name of the parent class of the validator, by default ""

    Returns
    -------
    color_input
        returns input if validation succeeds

    Raises
    ------
    ValueError
        raises ValueError inf validation fails
    """
    if allow_None and color_input is None:
        return color_input

    fail = True
    # check if greyscale
    isfloat = True
    try:
        float(color_input)
    except (ValueError, TypeError):
        isfloat = False
    if isfloat:
        color_new = color_input = float(color_input)
        if 0 <= color_new <= 1:
            c = int(color_new * 255)
            color_new = f"#{c:02x}{c:02x}{c:02x}"
    elif isinstance(color_input, (tuple, list)):
        color_new = tuple(color_input)
        if len(color_new) == 4:  # trim opacity
            color_new = color_new[:-1]
        if len(color_new) == 3:
            # transform matplotlib colors scaled from 0-1 to rgb colors
            if all(isinstance(c, float) for c in color_new):
                c = [int(255 * c) for c in color_new]
                color_new = f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
            if all(isinstance(c, int) for c in color_new):
                c = tuple(color_new)
                color_new = f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
    else:
        color_new = color_input
    if isinstance(color_new, str):
        color_new = COLORS_SHORT_TO_LONG.get(color_new, color_new)
        color_new = color_new.replace(" ", "").lower()
        if color_new.startswith("rgb"):
            color_new = color_new[4:-1].split(",")
            try:
                for i, c in enumerate(color_new):
                    color_new[i] = int(c)
            except (ValueError, TypeError):
                color_new = ""
            if len(color_new) == 3:
                c = tuple(color_new)
                color_new = f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
        re_hex = re.compile(r"#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})")
        fail = not re_hex.fullmatch(color_new)

    if fail and str(color_new) not in mcolors:
        raise ValueError(
            f"Invalid value of type '{type(color_input)}' "
            f"received for the color property of {parent_name}"
            f"\n   Received value: {color_input!r}"
            f"\n\nThe 'color' property is a color and may be specified as:\n"
            "    - A hex string (e.g. '#ff0000')\n"
            "    - A rgb string (e.g. 'rgb(185,204,255)')\n"
            "    - A rgb tuple (e.g. (120,125,126))\n"
            "    - A number between 0 and 1 (for grey scale) (e.g. '.5' or .8)\n"
            f"    - A named CSS color:\n{list(mcolors.keys())}"
        )
    return color_new


def validate_style_keys(style_kwargs):
    """validates style kwargs based on key up to first underscore.
    checks in the defaults structures the generally available style keys"""
    styles_by_family = get_defaults_dict("display.style")
    valid_keys = {key for v in styles_by_family.values() for key in v}
    level0_style_keys = {k.split("_")[0]: k for k in style_kwargs}
    kwargs_diff = set(level0_style_keys).difference(valid_keys)
    invalid_keys = {level0_style_keys[k] for k in kwargs_diff}
    if invalid_keys:
        raise ValueError(
            f"Following arguments are invalid style properties: `{invalid_keys}`\n"
            f"\n Available style properties are: `{valid_keys}`"
        )
    return style_kwargs


def get_families(obj):
    """get obj families"""
    # pylint: disable=import-outside-toplevel
    # pylint: disable=possibly-unused-variable
    # pylint: disable=redefined-outer-name
    from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet as Magnet
    from magpylib._src.obj_classes.class_magnet_Cuboid import Cuboid
    from magpylib._src.obj_classes.class_magnet_Cylinder import Cylinder
    from magpylib._src.obj_classes.class_magnet_Sphere import Sphere
    from magpylib._src.obj_classes.class_magnet_CylinderSegment import CylinderSegment
    from magpylib._src.obj_classes.class_magnet_Tetrahedron import Tetrahedron
    from magpylib._src.obj_classes.class_magnet_TriangularMesh import TriangularMesh
    from magpylib._src.obj_classes.class_BaseExcitations import BaseCurrent as Current
    from magpylib._src.obj_classes.class_current_Circle import Circle
    from magpylib._src.obj_classes.class_current_Polyline import Polyline
    from magpylib._src.obj_classes.class_misc_Dipole import Dipole
    from magpylib._src.obj_classes.class_misc_CustomSource import CustomSource
    from magpylib._src.obj_classes.class_misc_Triangle import Triangle
    from magpylib._src.obj_classes.class_Sensor import Sensor
    from magpylib._src.display.traces_generic import MagpyMarkers as Markers

    loc = locals()
    obj_families = []
    for item, val in loc.items():
        if not item.startswith("_"):
            try:
                if isinstance(obj, val):
                    obj_families.append(item.lower())
            except TypeError:
                pass
    return obj_families


def get_style(obj, **kwargs):
    """Returns default style based on increasing priority:
    - style from defaults
    - style from object
    - style from kwargs arguments
    """
    # parse kwargs into style an non-style arguments
    style_kwargs = kwargs.get("style", {})
    style_kwargs.update(
        {k[6:]: v for k, v in kwargs.items() if k.startswith("style") and k != "style"}
    )
    style_kwargs = validate_style_keys(style_kwargs)
    # create style class instance and update based on precedence
    style = obj.style.copy()
    keys = style.as_dict().keys()
    style_kwargs = {k: v for k, v in style_kwargs.items() if k.split("_")[0] in keys}
    style.update(**style_kwargs, match_properties=True)

    return style


def update_with_nested_dict(parameterized, nested_dict):
    """updates parameterized object recursively via setters"""
    # Using `batch_call_watchers` because it has the same underlying
    # mechanism as with `param.update`
    # See https://param.holoviz.org/user_guide/Dependencies_and_Watchers.html?highlight=batch_call
    # #batch-call-watchers
    with param.parameterized.batch_call_watchers(parameterized):
        for pname, value in nested_dict.items():
            if isinstance(value, dict):
                child = getattr(parameterized, pname)
                if isinstance(child, param.Parameterized):
                    update_with_nested_dict(child, value)
                    continue
            setattr(parameterized, pname, value)


def get_current_values_from_dict(obj, kwargs, match_properties=True):
    """
    Returns the current nested dictionary of values from the given object based on the keys of the
    the given kwargs.
    Parameters
    ----------
        obj: MagicParameterized:
            MagicParameterized class instance

        kwargs, dict:
            nested dictionary of values

        same_keys_only:
            if True only keys in found in the `obj` class are allowed.

    """
    new_dict = {}
    for k, v in kwargs.items():
        try:
            if isinstance(v, dict):
                v = get_current_values_from_dict(
                    getattr(obj, k), v, match_properties=False
                )
            else:
                v = getattr(obj, k)
            new_dict[k] = v
        except AttributeError as e:
            if match_properties:
                raise AttributeError(e) from e
    return new_dict


class MagicParameterized(param.Parameterized):
    """Base Magic Parametrized class"""

    __isfrozen = False

    def __init__(self, arg=None, **kwargs):
        super().__init__()
        self._freeze()
        self.update(arg=arg, **kwargs)

    def __setattr__(self, name, value):
        if self.__isfrozen and not hasattr(self, name) and not name.startswith("_"):
            raise AttributeError(
                f"{type(self).__name__} has no parameter '{name}'"
                f"\n Available parameters are: {list(self.as_dict().keys())}"
            )
        try:
            super().__setattr__(name, value)
        except ValueError:
            if name in self.param:
                p = self.param[name]
                # pylint: disable=unidiomatic-typecheck
                if type(p) == param.ClassSelector:
                    value = {} if value is None else value
                    if isinstance(value, dict):
                        value = type(getattr(self, name))(**value)
                        return
                if isinstance(value, (list, tuple, np.ndarray)):
                    if isinstance(p, param.List):
                        super().__setattr__(name, list(value))
                    elif isinstance(p, param.Tuple):
                        super().__setattr__(name, tuple(value))
                    return
            raise

    def _freeze(self):
        self.__isfrozen = True

    def update(self, arg=None, match_properties=True, **kwargs):
        """
        Updates the class properties with provided arguments, supports magic underscore notation

        Parameters
        ----------
        arg : dict
            Dictionary of properties to be updated
        match_properties: bool
            If `True`, checks if provided properties over keyword arguments are matching the current
            object properties. An error is raised if a non-matching property is found.
            If `False`, the `update` method does not raise any error when an argument is not
            matching a property.
        kwargs :
            Keyword/value pair of properties to be updated

        Returns
        -------
        ParameterizedType
            Updated parameterized object
        """
        if arg is None:
            arg = {}
        elif isinstance(arg, MagicParameterized):
            arg = arg.as_dict()
        else:
            arg = arg.copy()
        arg.update(kwargs)
        if arg:
            arg = magic_to_dict(arg)
            current_dict = get_current_values_from_dict(
                self, arg, match_properties=match_properties
            )
            new_dict = update_nested_dict(
                current_dict,
                arg,
                same_keys_only=not match_properties,
                replace_None_only=False,
            )
            update_with_nested_dict(self, new_dict)
        return self

    def as_dict(self, flatten=False, separator="_"):
        """
        returns recursively a nested dictionary with all properties objects of the class

        Parameters
        ----------
        flatten: bool
            If `True`, the nested dictionary gets flatten out with provided separator for the
            dictionary keys

        separator: str
            the separator to be used when flattening the dictionary. Only applies if
            `flatten=True`
        """
        dict_ = {}
        for key, val in self.param.values().items():
            if key == "name":
                continue
            if hasattr(val, "as_dict"):
                dict_[key] = val.as_dict()
            else:
                dict_[key] = val
        if flatten:
            dict_ = linearize_dict(dict_, separator=separator)
        return dict_

    def copy(self):
        """returns a copy of the current class instance"""
        return type(self)(**self.as_dict())


DEFAULTS_VALUES = magic_to_dict(
    {k: v["default"] for k, v in DEFAULTS.items()}, separator="."
)

"""utilities for creating property classes"""

# pylint: disable=too-many-branches

import re
from copy import deepcopy
from functools import lru_cache

import numpy as np
from matplotlib.colors import CSS4_COLORS as mcolors

from magpylib._src.defaults.defaults_values import DEFAULTS
from magpylib._src.display.api import DisplayBackend

SUPPORTED_PLOTTING_BACKENDS = ("matplotlib", "plotly", "pyvista")


def get_registered_backends():
    """Return the names of the display backends registered so far.

    Unlike `SUPPORTED_PLOTTING_BACKENDS`, which lists the built-in backends
    only, this reflects backends registered at runtime through
    `register_backend`.
    """
    return tuple(DisplayBackend.backends)


ALLOWED_SYMBOLS = (".", "+", "D", "d", "s", "x", "o")

ALLOWED_LINESTYLES = (
    "solid",
    "dashed",
    "dotted",
    "dashdot",
    "loosely dotted",
    "loosely dashdotted",
    "-",
    "--",
    "-.",
    ".",
    ":",
    (0, (1, 1)),
)

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
    keyword if no other obvious default (e.g., None) is suitable,

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


def get_defaults_dict(arg=None) -> dict:
    """returns default dict or sub-dict based on `arg`.
    (e.g. `get_defaults_dict('display.style')`)

    Returns
    -------
    dict
        default sub dict
    """

    dict_ = deepcopy(DEFAULTS)
    if arg is not None:
        for v in arg.split("."):
            dict_ = dict_[v]
    return dict_


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


def color_validator(color_input, allow_None=True, parent_name=""):
    """validates color inputs based on chosen `backend', allows `None` by default.

    Parameters
    ----------
    color_input : str
        Color input as string.
    allow_None : bool, optional
        If ``True``, ``color_input`` can be ``None``, by default ``True``.
    parent_name : str, optional
        name of the parent class of the validator, by default ''.

    Returns
    -------
    color_input
        returns input if validation succeeds

    Raises
    ------
    ValueError
        raises ValueError inf validation fails
    """
    if isinstance(color_input, list | np.ndarray):
        color_input = tuple(np.asarray(color_input).tolist())
    try:
        return _color_validator_cached(color_input, allow_None, parent_name)
    except TypeError:  # unhashable input (e.g. nested list) cannot use the cache
        return _color_validator_cached.__wrapped__(color_input, allow_None, parent_name)


@lru_cache(maxsize=1000)
def _color_validator_cached(color_input, allow_None=True, parent_name=""):
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
    elif isinstance(color_input, tuple | list):
        color_new = tuple(color_input)
        if len(color_new) == 4:  # trim opacity
            color_new = color_new[:-1]
        if len(color_new) == 3:
            # transform Matplotlib colors scaled from 0-1 to rgb colors
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
        msg = (
            f"Invalid value of type '{type(color_input)}' "
            f"received for the color property of {parent_name}"
            f"\n   Received value: {color_input!r}"
            f"\n\nThe 'color' property is a color and may be specified as:\n"
            "    - A hex string (e.g. '#ff0000')\n"
            "    - A rgb string (e.g. 'rgb(185, 204, 255)')\n"
            "    - A rgb tuple (e.g. (120, 125, 126))\n"
            "    - A number between 0 and 1 (for grey scale) (e.g. '.5' or .8)\n"
            f"    - A named CSS color:\n{list(mcolors.keys())}"
        )
        raise ValueError(msg)
    return color_new


@lru_cache(maxsize=1)
def _valid_style_keys():
    """generally available style keys, derived from the hardcoded defaults"""
    styles_by_family = DEFAULTS["display"]["style"]
    return frozenset(key for v in styles_by_family.values() for key in v)


def validate_style_keys(style_kwargs):
    """validates style kwargs based on key up to first underscore.
    checks in the defaults structures the generally available style keys"""
    valid_keys = _valid_style_keys()
    level0_style_keys = {k.split("_")[0]: k for k in style_kwargs}
    kwargs_diff = set(level0_style_keys).difference(valid_keys)
    invalid_keys = {level0_style_keys[k] for k in kwargs_diff}
    if invalid_keys:
        msg = (
            f"The following style properties are invalid: {invalid_keys}. "
            f"Available style properties are: {valid_keys}."
        )
        raise ValueError(msg)
    return style_kwargs

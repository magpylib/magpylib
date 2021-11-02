"""utilites for creating property classes"""

from copy import deepcopy
import collections.abc
from magpylib._lib.defaults import DEFAULTS

MAGPYLIB_FAMILIES = {
    "Line": ("currents",),
    "Circular": ("currents",),
    "Cuboid": ("magnets",),
    "Cylinder": ("magnets",),
    "Sphere": ("magnets",),
    "CylinderSegment": ("magnets",),
    "Sensor": ("sensors",),
    "Dipole": ("dipoles",),
    "Marker": ("markers",),
}

SYMBOLS_MATPLOTLIB_TO_PLOTLY = {
    ".": "circle",
    "o": "circle-open",
    "+": "cross",
    "D": "diamond",
    "d": "diamond-open",
    "s": "square",
    "x": "x",
}

LINESTYLES_MATPLOTLIB_TO_PLOTLY = {
    "solid": "solid",
    "-": "solid",
    "dashed": "dash",
    "--": "dash",
    "dashdot": "dashdot",
    "-.": "dashdot",
    "dotted": "dot",
    ".": "dot",
    (0, (1, 1)): "dot",
    "loosely dotted": "longdash",
    "loosely dashdotted": "longdashdot",
}

COLORS_MATPLOTLIB_TO_PLOTLY = {
    "r": "red",
    "g": "green",
    "b": "blue",
    "y": "yellow",
    "m": "magenta",
    "c": "cyan",
    "k": "black",
    "w": "white",
}

SIZE_FACTORS_MATPLOTLIB_TO_PLOTLY = {
    "line_width": 2.2,
    "marker_size": 0.35,
}


def get_defaults_dict(arg=None) -> dict:
    """returns default dict or sub-dict based on `arg`

    Returns
    -------
    dict
        defaut sub dict

    Examples
    --------
    >>> get_default_dict('display.styles')
    """

    dict_ = deepcopy(DEFAULTS)
    if arg is not None:
        for v in arg.split("."):
            dict_ = dict_[v]
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
    for k, v in u.items():
        if k in d or not same_keys_only:
            if isinstance(d, collections.abc.Mapping):
                if isinstance(v, collections.abc.Mapping):
                    r = update_nested_dict(
                        d.get(k, {}),
                        v,
                        same_keys_only=same_keys_only,
                        replace_None_only=replace_None_only,
                    )
                    d[k] = r
                elif d[k] is None or not replace_None_only:
                    d[k] = u[k]
            else:
                d = {k: u[k]}
    return d


def magic_to_dict(kwargs, separator="_") -> dict:
    """decomposes recursively a dictionary with keys with underscores into a nested dictionary
    example : {'magnet_color':'blue'} -> {'magnet': {'color':'blue'}}
    see: https://plotly.com/python/creating-and-updating-figures/#magic-underscore-notation

    Parameters
    ----------
    kwargs : dict
        dictionary of keys to be decomposed into a nested dictionary

    separator: str, default='_'
        defines the sperator to apply the magic parsing with
    Returns
    -------
    dict
        nested dictionary
    """
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
            new_kwargs[k] = magic_to_dict(v)
    return new_kwargs


def linearize_dict(kwargs, separator=".") -> dict:
    """linearizes `kwargs` dictionary using the provided `separator
    Parameters
    ----------
    kwargs : dict
        dictionary of keys linearized into an flat dictionary

    separator: str, default='.'
        defines the sperator to be applied on the final dictionary keys

    Returns
    -------
    dict
        flat dictionary with keys names using a separator

    Examples
    --------
    >>> mydict = {
        'line': {'width': 1, 'style': 'solid', 'color': None},
        'marker': {'size': 1, 'symbol': 'o', 'color': None}
    }
    >>> linearize_dict(mydict, separator='.')
    {'line.width': 1,
     'line.style': 'solid',
     'line.color': None,
     'marker.size': 1,
     'marker.symbol': 'o',
     'marker.color': None}
    """
    dict_ = {}
    for k, v in kwargs.items():
        if isinstance(v, dict):
            d = linearize_dict(v, separator=separator)
            for key, val in d.items():
                dict_[f"{k}{separator}{key}"] = val
        else:
            dict_[k] = v
    return dict_


def color_validator(color_input, allow_None=True, parent_name="", backend="matplotlib"):
    """validates color inputs based on chosen `backend', allows `None` by default.

    Parameters
    ----------
    color_input : str
        color input as string
    allow_None : bool, optional
        if `True` `color_input` can be `None`, by default True
    parent_name : str, optional
        name of the parent class of the validator, by default ""
    backend: str, optional
        plotting backend to validate with. One of `['matplotlib','plotly']`

    Returns
    -------
    color_input
        returns input if validation succeeds

    Raises
    ------
    ValueError
        raises ValueError inf validation fails
    """
    if not allow_None or color_input is not None:
        # pylint: disable=import-outside-toplevel
        color_input = COLORS_MATPLOTLIB_TO_PLOTLY.get(color_input, color_input)
        if backend == "matplotlib":
            import re

            hex_fail = True
            if isinstance(color_input, str):
                color_input = color_input.replace(" ", "").lower()
                re_hex = re.compile(r"#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})")
                hex_fail = not re_hex.fullmatch(color_input)
            from matplotlib.colors import CSS4_COLORS as mcolors

            if hex_fail and str(color_input) not in mcolors:
                raise ValueError(
                    f"""\nInvalid value of type '{type(color_input)}' """
                    f"""received for the color property of {parent_name}"""
                    f"""\n   Received value: '{color_input}'"""
                    f"""\n\nThe 'color' property is a color and may be specified as:\n"""
                    """    - A hex string (e.g. '#ff0000')\n"""
                    f"""    - A named CSS color:\n{list(mcolors.keys())}"""
                )
        else:
            from _plotly_utils.basevalidators import ColorValidator

            cv = ColorValidator(plotly_name="color", parent_name=parent_name)
            color_input = cv.validate_coerce(color_input)
    return color_input


class BaseProperties:
    """
    Base Class to represent only the property attributes defined at initialization, after which the
    class is frozen. This prevents user to create any attributes that are not defined as properties.

    Raises
    ------
    AttributeError
        raises AttributeError if the object is not a property
    """ """"""

    __isfrozen = False

    def __init__(self, **kwargs):
        input_dict = {k: None for k in self._property_names_generator()}
        if kwargs:
            magic_kwargs = magic_to_dict(kwargs)
            diff = set(magic_kwargs.keys()).difference(set(input_dict.keys()))
            for attr in diff:
                raise AttributeError(
                    f"""{type(self).__name__} has no attribute '{attr}'"""
                )
            input_dict.update(magic_kwargs)
        for k, v in input_dict.items():
            setattr(self, k, v)
        self._freeze()

    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise AttributeError(
                f"{type(self).__name__} has no property '{key}'"
                f"\n Available properties are: {list(self._property_names_generator())}"
            )
        object.__setattr__(self, key, value)

    def _freeze(self):
        self.__isfrozen = True

    def _property_names_generator(self):
        """returns a generator with class properties only"""
        return (
            attr
            for attr in dir(self)
            if isinstance(getattr(type(self), attr, None), property)
        )

    def __repr__(self):
        params = self._property_names_generator()
        dict_str = ", ".join(f"{k}={repr(getattr(self,k))}" for k in params)
        return f"{type(self).__name__}({dict_str})"

    def as_dict(self):
        """returns recursively a nested dictionary with all properties objects of the class"""
        params = self._property_names_generator()
        dict_ = {}
        for k in params:
            val = getattr(self, k)
            if hasattr(val, "as_dict"):
                dict_[k] = val.as_dict()
            else:
                dict_[k] = val
        return dict_

    def update(
        self, arg=None, _match_properties=True, _replace_None_only=False, **kwargs
    ):
        """
        updates the class properties with provided arguments, supports magic underscore notation

        Returns
        -------
        self
        """
        if arg is None:
            arg = {}
        if kwargs:
            arg.update(magic_to_dict(kwargs))
        current_dict = self.as_dict()
        new_dict = update_nested_dict(
            current_dict,
            arg,
            same_keys_only=not _match_properties,
            replace_None_only=_replace_None_only,
        )
        for k, v in new_dict.items():
            setattr(self, k, v)
        return self

    def copy(self):
        """returns a copy of the current class instance"""
        return type(self)(**self.as_dict())

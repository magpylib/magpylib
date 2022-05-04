"""utilities for creating property classes"""
import collections.abc
from copy import deepcopy

from magpylib._src.defaults.defaults_values import DEFAULTS


SUPPORTED_PLOTTING_BACKENDS = ("matplotlib", "plotly")

MAGPYLIB_FAMILIES = {
    "Line": ("current",),
    "Loop": ("current",),
    "Cuboid": ("magnet",),
    "Cylinder": ("magnet",),
    "Sphere": ("magnet",),
    "CylinderSegment": ("magnet",),
    "Sensor": ("sensor",),
    "Dipole": ("dipole",),
    "Marker": ("markers",),
}

SYMBOLS_MATPLOTLIB_TO_PLOTLY = {
    ".": "circle",
    "o": "circle",
    "+": "cross",
    "D": "diamond",
    "d": "diamond",
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
    ":": "dot",
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
    "marker_size": 0.7,
}


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
        if not isinstance(d, collections.abc.Mapping):
            if d is None or not replace_None_only:
                d = u
        elif k in d or not same_keys_only:
            if isinstance(v, collections.abc.Mapping):
                r = update_nested_dict(
                    d.get(k, {}),
                    v,
                    same_keys_only=same_keys_only,
                    replace_None_only=replace_None_only,
                )
                d[k] = r
            elif d.get(k, None) is None or not replace_None_only:
                if not same_keys_only or k in d:
                    d[k] = u[k]
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
            new_kwargs[k] = magic_to_dict(v)
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
    color_input_original = color_input
    if not allow_None or color_input is not None:
        # pylint: disable=import-outside-toplevel
        color_input = COLORS_MATPLOTLIB_TO_PLOTLY.get(color_input, color_input)
        import re

        hex_fail = True
        # pylint: disable=W0702
        try:  # check if greyscale
            c = float(color_input)
            if c < 0 or c > 1:
                msg = (
                    "When setting a grey tone, value must be between 0 and 1"
                    f"""\n   Received value: '{color_input_original}'"""
                )
                raise ValueError(msg)
            c = int(c * 255)
            color_input = f"#{c:02x}{c:02x}{c:02x}"
        except:
            pass

        if isinstance(color_input, (tuple, list)):

            if len(color_input) == 4: # do not allow opacity values for now
                color_input = color_input[:-1]
            if len(color_input) != 3:
                raise ValueError(
                    "Input color must be of shape (3,) or (4,)."
                    f"Instead received {color_input}"
                )
            # transform matplotlib colors scaled from 0-1 to rgb colors
            if not isinstance(color_input[0], int):
                color_input = [int(255*c) for c in color_input]
            c = tuple(color_input)
            color_input = f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"

        if isinstance(color_input, str):
            color_input = color_input.replace(" ", "").lower()
            if color_input.startswith("rgb"):
                try:
                    c = color_input[4:-1].split(",")
                    c = tuple(int(c) for c in c)
                    color_input = f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
                except:
                    pass
            re_hex = re.compile(r"#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})")
            hex_fail = not re_hex.fullmatch(color_input)

        from matplotlib.colors import CSS4_COLORS as mcolors

        if hex_fail and str(color_input) not in mcolors:
            raise ValueError(
                f"""\nInvalid value of type '{type(color_input)}' """
                f"""received for the color property of {parent_name}"""
                f"""\n   Received value: '{color_input_original}'"""
                f"""\n\nThe 'color' property is a color and may be specified as:\n"""
                """    - A hex string (e.g. '#ff0000')\n"""
                """    - A rgb string (e.g. 'rgb(185,204,255))\n"""
                """    - A rgb tuple (e.g. (120,125,126))\n"""
                """    - A number between 0 and 1 (for grey scale) (e.g. '.5' or .8)\n"""
                f"""    - A named CSS color:\n{list(mcolors.keys())}"""
            )
    return color_input


def validate_property_class(val, name, class_, parent):
    """validator for sub property"""
    if isinstance(val, dict):
        val = class_(**val)
    elif val is None:
        val = class_()
    if not isinstance(val, class_):
        raise ValueError(
            f"the `{name}` property of `{type(parent).__name__}` must be an instance \n"
            f"of `{class_}` or a dictionary with equivalent key/value pairs \n"
            f"but received {repr(val)} instead"
        )
    return val


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


class MagicProperties:
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
                    f"{type(self).__name__} has no property '{attr}'"
                    f"\n Available properties are: {list(self._property_names_generator())}"
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

    def as_dict(self, flatten=False, separator="."):
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
        params = self._property_names_generator()
        dict_ = {}
        for k in params:
            val = getattr(self, k)
            if hasattr(val, "as_dict"):
                dict_[k] = val.as_dict()
            else:
                dict_[k] = val
        if flatten:
            dict_ = linearize_dict(dict_, separator=separator)
        return dict_

    def update(
        self, arg=None, _match_properties=True, _replace_None_only=False, **kwargs
    ):
        """
        Updates the class properties with provided arguments, supports magic underscore notation

        Parameters
        ----------

        _match_properties: bool
            If `True`, checks if provided properties over keyword arguments are matching the current
            object properties. An error is raised if a non-matching property is found.
            If `False`, the `update` method does not raise any error when an argument is not
            matching a property.

        _replace_None_only:
            updates matching properties that are equal to `None` (not already been set)


        Returns
        -------
        self
        """
        if arg is None:
            arg = {}
        if kwargs:
            arg.update(kwargs)
        arg = magic_to_dict(arg)
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
        return deepcopy(self)

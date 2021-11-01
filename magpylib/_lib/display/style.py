"""Collection of class for display styles"""
# pylint: disable=C0302

import collections.abc

_MAGPYLIB_FAMILIES = {
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
_SYMBOLS_MATPLOTLIB_TO_PLOTLY = {
    ".": "circle",
    "o": "circle-open",
    "+": "cross",
    "D": "diamond",
    "d": "diamond-open",
    "s": "square",
    "x": "x",
}

_LINESTYLES_MATPLOTLIB_TO_PLOTLY = {
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

_COLORS_MATPLOTLIB_TO_PLOTLY = {
    "r": "red",
    "g": "green",
    "b": "blue",
    "y": "yellow",
    "m": "magenta",
    "c": "cyan",
    "k": "black",
    "w": "white",
}

_DEFAULT_STYLES = {
    "base": {
        "path": {
            "line": {"width": 1, "style": "solid", "color": None},
            "marker": {"size": 1, "symbol": "o", "color": None},
        },
        "description": {"show": True, "text": None},
        "opacity": 1,
        "mesh3d": {"show": True, "data": None},
        "color": None,
    },
    "magnets": {
        "magnetization": {
            "show": True,
            "size": 1,
            "color": {
                "north": "#E71111",
                "middle": "#DDDDDD",
                "south": "#00B050",
                "transition": 0.2,
            },
        }
    },
    "currents": {"current": {"show": True, "size": 1}},
    "sensors": {"size": 1, "pixel": {"size": 1, "color": None, "symbol": "o"}},
    "dipoles": {"size": 1, "pivot": "middle"},
    "markers": {"marker": {"size": 2, "color": "grey", "symbol": "x"}},
}


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


def magic_to_dict(kwargs) -> dict:
    """decomposes recursively a dictionary with keys with underscores into a nested dictionary
    example : {'magnet_color':'blue'} -> {'magnet': {'color':'blue'}}
    see: https://plotly.com/python/creating-and-updating-figures/#magic-underscore-notation

    Parameters
    ----------
    kwargs : dict
        dictionary of keys to be decomposed into a nested dictionary

    Returns
    -------
    dict
        nested dictionary
    """
    new_kwargs = {}
    for k, v in kwargs.items():
        keys = k.split("_")
        if len(keys) == 1:
            new_kwargs[keys[0]] = v
        else:
            val = {"_".join(keys[1:]): v}
            if keys[0] in new_kwargs and isinstance(new_kwargs[keys[0]], dict):
                new_kwargs[keys[0]].update(val)
            else:
                new_kwargs[keys[0]] = val
    for k, v in new_kwargs.items():
        if isinstance(v, dict):
            new_kwargs[k] = magic_to_dict(v)
    return new_kwargs


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
        color_input = _COLORS_MATPLOTLIB_TO_PLOTLY.get(color_input, color_input)
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


def get_style(obj, **kwargs):
    """
    returns default style based on increasing priority:
    - style from Config
    - style from object
    - style from kwargs arguments
    """
    # parse kwargs into style an non-style arguments
    style_kwargs = kwargs.get("style", {})
    style_kwargs.update(
        {k[6:]: v for k, v in kwargs.items() if k.startswith("style") and k != "style"}
    )

    # retrive default style dictionary,
    styles_by_family = default_style.as_dict()

    # construct object specific dictionary base on style family and default style
    obj_type = getattr(obj, "_object_type", None)
    obj_families = _MAGPYLIB_FAMILIES.get(obj_type, [])

    obj_style_dict = {
        **styles_by_family["base"],
        **{
            k: v
            for fam in obj_families
            for k, v in styles_by_family.get(fam, {}).items()
        },
    }

    # create style class instance and update based on precedence
    obj_style = getattr(obj, "style", None)
    style = obj_style.copy() if obj_style is not None else BaseGeoStyle()
    style.update(**style_kwargs, _match_properties=False)
    style.update(**obj_style_dict, _match_properties=False, _replace_None_only=True)

    return style


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


class BaseStyle(BaseProperties):
    """
    Base class for display styling options of all objects to be displayed

    Properties
    ----------
    name : str, default=None
        name of the class instance, can be any string.

    description: dict or Description, default=None
        object description properties

    color: str, default=None
        a valid css color. Can also be one of `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`

    opacity: float, default=None
        object opacity between 0 and 1, where 1 is fully opaque and 0 is fully transparent.
    """

    def __init__(
        self,
        name=None,
        description=None,
        color=None,
        opacity=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            color=color,
            opacity=opacity,
            **kwargs,
        )

    @property
    def name(self):
        """name of the class instance, can be any string"""
        return self._name

    @name.setter
    def name(self, val):
        self._name = val if val is None else str(val)

    @property
    def description(self):
        """Description class with 'text' and 'show' properties"""
        return self._description

    @description.setter
    def description(self, val):
        if isinstance(val, dict):
            val = Description(**val)
        if isinstance(val, Description):
            self._description = val
        elif val is None:
            self._description = Description()
        else:
            raise ValueError(
                f"the `description` property of `{type(self).__name__}` must be an instance \n"
                "of `Description` or a dictionary with equivalent key/value pairs \n"
                f"but received {repr(val)} instead"
            )

    @property
    def color(self):
        """a valid css color. Can also be one of `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`"""
        return self._color

    @color.setter
    def color(self, val):
        self._color = color_validator(val, parent_name=f"{type(self).__name__}")

    @property
    def opacity(self):
        """object opacity between 0 and 1, where 1 is fully opaque and 0 is fully transparent"""
        return self._opacity

    @opacity.setter
    def opacity(self, val):
        assert val is None or (
            isinstance(val, (float, int)) and val >= 0 and val <= 1
        ), (
            "opacity must be a value betwen 0 and 1\n"
            f"but received {repr(val)} instead"
        )
        self._opacity = val


class BaseGeoStyle(BaseStyle):
    """
    Base class for display styling options of `BaseGeo` objects

    Properties
    ----------
    name : str, default=None
        name of the class instance, can be any string.

    description: dict or Description, default=None
        object description properties

    color: str, default=None
        a valid css color. Can also be one of `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`

    opacity: float, default=None
        object opacity between 0 and 1, where 1 is fully opaque and 0 is fully transparent.

    path: dict or PathStyle, default=None
        an instance of `PathStyle` or dictionary of equivalent key/value pairs, defining the object
        path marker and path line properties.

    mesh3d: dict or Mesh3d, default=None
        an instance of `Mesh3d` or dictionary of equivalent key/value pairs. Defines properties for
        an additional user-defined mesh3d object which is positioned relatively to the main object
        to be displayed and moved automatically with it. This feature also allows the user to
        replace the original 3d representation of the object.
    """

    def __init__(
        self,
        name=None,
        description=None,
        color=None,
        opacity=None,
        path=None,
        mesh3d=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            color=color,
            opacity=opacity,
            path=path,
            mesh3d=mesh3d,
            **kwargs,
        )

    @property
    def path(self):
        """an instance of `PathStyle` or dictionary of equivalent key/value pairs, defining the
        object path marker and path line properties"""
        return self._path

    @path.setter
    def path(self, val):
        if isinstance(val, dict):
            val = PathStyle(**val)
        if isinstance(val, PathStyle):
            self._path = val
        elif val is None:
            self._path = PathStyle()
        else:
            raise ValueError(
                f"the `path` property of `{type(self).__name__}` must be an instance \n"
                "of `PathStyle` or a dictionary with equivalent key/value pairs \n"
                f"but received {repr(val)} instead"
            )

    @property
    def mesh3d(self):
        """
        an instance of `Mesh3d` or dictionary of equivalent key/value pairs. Defines properties for
        an additional user-defined mesh3d object which is positioned relatively to the main object
        to be displayed and moved automatically with it. This feature also allows the user to
        replace the original 3d representation of the object.
        """
        return self._mesh3d

    @mesh3d.setter
    def mesh3d(self, val):
        if isinstance(val, dict):
            val = Mesh3d(**val)
        if isinstance(val, Mesh3d):
            self._mesh3d = val
        elif val is None:
            self._mesh3d = Mesh3d()
        else:
            raise ValueError(
                f"the `mesh3d` property of `{type(self).__name__}` must be an instance \n"
                "of `Mesh3d` or a dictionary with equivalent key/value pairs \n"
                f"but received {repr(val)} instead"
            )


class Description(BaseProperties):
    """
    Defines properties for a description object

    Properties
    ----------
    text: str, default=None
        Object description text

    show: bool, default=None
        if `True`, adds legend entry suffix based on value
    """

    def __init__(self, text=None, show=None, **kwargs):
        super().__init__(text=text, show=show, **kwargs)

    @property
    def text(self):
        """
        texts/hides mesh3d object based on provided data:
        - True: texts mesh
        - False: hides mesh
        - 'inplace': replace object representation
        """
        return self._text

    @text.setter
    def text(self, val):
        self._text = val if val is None else str(val)

    @property
    def show(self):
        """if `True`, adds legend entry suffix based on value"""
        return self._show

    @show.setter
    def show(self, val):
        assert val is None or isinstance(val, bool), (
            f"the `show` property of {type(self).__name__} must be either `True` or `False`\n"
            f"but received {repr(val)} instead"
        )
        self._show = val


class Mesh3d(BaseProperties):
    """
    Defines properties for an additional user-defined mesh3d object which is positioned relatively
    to the main object to be displayed and moved automatically with it. This feature also allows
    the user to replace the original 3d representation of the object

    Properties
    ----------
    show : bool, default=None
        shows/hides mesh3d object based on provided data:
        - True: shows mesh
        - False: hides mesh
        - 'inplace': replace object representation

    data: dict, default=None
        dictionary containing the `x,y,z,i,j,k` keys/values pairs for a mesh3d object

    """

    def __init__(self, data=None, show=None, **kwargs):
        super().__init__(data=data, show=show, **kwargs)

    @property
    def show(self):
        """
        shows/hides mesh3d object based on provided data:
        - True: shows mesh
        - False: hides mesh
        - 'inplace': replace object representation
        """
        return self._show

    @show.setter
    def show(self, val):
        assert val is None or isinstance(val, bool) or val == "inplace", (
            f"the `show` property of {type(self).__name__} must be "
            f"one of `[True, False, 'inplace']`"
            f" but received {repr(val)} instead"
        )
        self._show = val

    @property
    def data(self):
        """dictionary containing the `x,y,z,i,j,k` keys/values pairs for a mesh3d object"""
        return self._data

    @data.setter
    def data(self, val):
        # pylint: disable=import-outside-toplevel
        assert val is None or (
            isinstance(val, dict) and all(key in val for key in "xyzijk")
        ), (
            "data must be a dict-like object containing the `x,y,z,i,j,k` keys/values pairs"
            f" but received {repr(val)} instead"
        )
        self._data = val


class Magnetization(BaseProperties):
    """
    Defines magnetization styling properties

    Properties
    ----------
    show : bool, default=None
        if `True` shows magnetization direction based on active plotting backend

    size: float, default=None
        arrow size of the magnetization direction (for the matplotlib backend)
        only applies if `show=True`

    color: dict, MagnetizationColor, default=None
        color properties showing the magnetization direction (for the plotly backend)
        only applies if `show=True`
    """

    def __init__(self, show=None, size=None, color=None, **kwargs):
        super().__init__(show=show, size=size, color=color, **kwargs)

    @property
    def show(self):
        """if `True` shows magnetization direction based on active plotting backend"""
        return self._show

    @show.setter
    def show(self, val):
        assert val is None or isinstance(val, bool), (
            "show must be either `True` or `False` \n"
            f" but received {repr(val)} instead"
        )
        self._show = val

    @property
    def size(self):
        """positive float for relative arrow size to magnet size"""
        return self._size

    @size.setter
    def size(self, val):
        assert val is None or isinstance(val, (int, float)) and val >= 0, (
            f"the `size` property of {type(self).__name__} must be a positive number"
            f" but received {repr(val)} instead"
        )
        self._size = val

    @property
    def color(self):
        """color properties showing the magnetization direction (for the plotly backend)
        only applies if `show=True`"""
        return self._color

    @color.setter
    def color(self, val):
        if isinstance(val, dict):
            val = MagnetizationColor(**val)
        if isinstance(val, MagnetizationColor):
            self._color = val
        elif val is None:
            self._color = MagnetizationColor()
        else:
            raise ValueError(
                f"the `color` property of `{type(self).__name__}` must be an instance \n"
                "of `MagnetizationColor` or a dictionary with equivalent key/value pairs \n"
                f"but received {repr(val)} instead"
            )


class MagnetizationColor(BaseProperties):
    """
    Defines the magnetization direction color styling properties.

    Note: This feature is only relevant for the plotly backend.

    Properties
    ----------
    north: str, default=None
        defines the color of the magnetic north pole

    south: str, default=None
        defines the color of the magnetic south pole

    middle: str, default=None
        defines the color between the magnetic poles
        if set to `auto`, the middle color will get a color from the color sequence cylcing over
        objects to be displayed

    transition: float, default=None
        sets the transition smoothness between poles colors.
        - `transition=0`: discrete transition
        - `transition=1`: smoothest transition
        - can be any value in-between 0 and 1"
    """

    def __init__(self, north=None, south=None, middle=None, transition=None, **kwargs):
        super().__init__(
            north=north,
            middle=middle,
            south=south,
            transition=transition,
            **kwargs,
        )

    @property
    def north(self):
        """the color of the magnetic north pole"""
        return self._north

    @north.setter
    def north(self, val):
        self._north = color_validator(val)

    @property
    def south(self):
        """the color of the magnetic south pole"""
        return self._south

    @south.setter
    def south(self, val):
        self._south = color_validator(val)

    @property
    def middle(self):
        """the color between the magnetic poles
        if set to `auto`, the middle color will get a color from the color sequence cylcing over
        objects to be displayed"""
        return self._middle

    @middle.setter
    def middle(self, val):
        if val != "auto" and not val is False:
            val = color_validator(val)
        self._middle = val

    @property
    def transition(self):
        """sets the transition smoothness between poles colors.
        `transition=0`: discrete transition
        `transition=1`: smoothest transition
        can be any value in-between 0 and 1"""
        return self._transition

    @transition.setter
    def transition(self, val):
        assert (
            val is None or isinstance(val, (float, int)) and val >= 0 and val <= 1
        ), "color transition must be a value betwen 0 and 1"
        self._transition = val


class Magnets(BaseProperties):
    """
    Defines the specific styling properties of objects of the `magnets` family

    Properties
    ----------
    magnetization: dict or Magnetization, default=None

    """

    def __init__(self, magnetization=None, **kwargs):
        super().__init__(magnetization=magnetization, **kwargs)

    @property
    def magnetization(self):
        """Magnetization class with 'north', 'south', 'middle' and 'transition' values
        or a dictionary with equivalent key/value pairs"""
        return self._magnetization

    @magnetization.setter
    def magnetization(self, val):
        if isinstance(val, dict):
            val = Magnetization(**val)
        if isinstance(val, Magnetization):
            self._magnetization = val
        elif val is None:
            self._magnetization = Magnetization()
        else:
            raise ValueError(
                f"the `magnetization` property of `{type(self).__name__}` must be an instance \n"
                "of `Magnetization` or a dictionary with equivalent key/value pairs \n"
                f"but received {repr(val)} instead"
            )


class MagnetStyle(BaseGeoStyle, Magnets):
    """Defines the styling properties of objects of the `magnets` family with base properties"""


class Sensors(BaseProperties):
    """
    Defines the specific styling properties of objects of the `sensors` family

    Properties
    ----------
    size: float, default=None
        positive float for relative sensor to canvas size

    pixel: dict, Pixel, default=None
        `Pixel` class or dict with equivalent key/value pairs (e.g. `color`, `size`)
    """

    def __init__(self, size=None, pixel=None, **kwargs):
        super().__init__(size=size, pixel=pixel, **kwargs)

    @property
    def size(self):
        """positive float for relative sensor to canvas size"""
        return self._size

    @size.setter
    def size(self, val):
        assert val is None or isinstance(val, (int, float)) and val >= 0, (
            f"the `size` property of {type(self).__name__} must be a positive number"
            f" but received {repr(val)} instead"
        )
        self._size = val

    @property
    def pixel(self):
        """`Pixel` class or dict with equivalent key/value pairs (e.g. `color`, `size`)"""
        return self._pixel

    @pixel.setter
    def pixel(self, val):
        if isinstance(val, dict):
            val = Pixel(**val)
        if isinstance(val, Pixel):
            self._pixel = val
        elif val is None:
            self._pixel = Pixel()
        else:
            raise ValueError(
                f"the `pixel` property of `{type(self).__name__}` must be an instance \n"
                "of `Pixel` or a dictionary with equivalent key/value pairs \n"
                f"but received {repr(val)} instead"
            )


class SensorStyle(BaseGeoStyle, Sensors):
    """Defines the styling properties of objects of the `sensors` family with base properties"""


class Pixel(BaseProperties):
    """
    Defines the styling properties of sensor pixels

    Properties
    ----------
    size: float, default=None
        defines the relative pixel size:
        - matplotlib backend: pixel size is the marker size
        - plotly backend:  relative size to the distance of nearest neighboring pixel

    - color: str, default=None
        defines the pixel color
    """

    def __init__(self, size=1, color=None, **kwargs):
        super().__init__(size=size, color=color, **kwargs)

    @property
    def size(self):
        """positive float, the relative pixel size:
        - matplotlib backend: pixel size is the marker size
        - plotly backend:  relative size to the distance of nearest neighboring pixel"""
        return self._size

    @size.setter
    def size(self, val):
        assert val is None or isinstance(val, (int, float)) and val >= 0, (
            f"the `size` property of {type(self).__name__} must be a positive number"
            f" but received {repr(val)} instead"
        )
        self._size = val

    @property
    def color(self):
        """the pixel color"""
        return self._color

    @color.setter
    def color(self, val):
        self._color = color_validator(val, parent_name=f"{type(self).__name__}")


class Currents(BaseProperties):
    """
    Defines the specific styling properties of objects of the `currents` family

    Properties
    ----------
    show: bool, default=None
        if `True` current direction is shown with an arrow
    size: defines the size of the arrows
    """

    def __init__(self, current=None, **kwargs):
        super().__init__(current=current, **kwargs)

    @property
    def current(self):
        """ArrowStyle class with 'show', 'size' properties"""
        return self._current

    @current.setter
    def current(self, val):
        if isinstance(val, dict):
            val = ArrowStyle(**val)
        if isinstance(val, ArrowStyle):
            self._current = val
        elif val is None:
            self._current = ArrowStyle()
        else:
            raise ValueError(
                f"the `current` property of `{type(self).__name__}` must be an instance \n"
                "of `ArrowStyle` or a dictionary with equivalent key/value pairs \n"
                f"but received {repr(val)} instead"
            )


class CurrentStyle(BaseGeoStyle, Currents):
    """Defines the styling properties of objects of the `currents` family and base properties"""


class ArrowStyle(BaseProperties):
    """
    Defines the styling properties of current arrows

    Properties
    ----------
    show: bool, default=None
        if `True` current direction is shown with an arrow

    size: float
        positive number defining the size of the arrows
    """

    def __init__(self, show=None, size=None, **kwargs):
        super().__init__(show=show, size=size, **kwargs)

    @property
    def show(self):
        """show/hide arrow showing current direction"""
        return self._show

    @show.setter
    def show(self, val):
        assert val is None or isinstance(val, bool), (
            f"the `show` property of {type(self).__name__} must be either `True` or `False`"
            f" but received {repr(val)} instead"
        )
        self._show = val

    @property
    def size(self):
        """positive number defining the size of the arrows"""
        return self._size

    @size.setter
    def size(self, val):
        assert val is None or isinstance(val, (int, float)) and val >= 0, (
            f"the `size` property of {type(self).__name__} must be a positive number"
            f" but received {repr(val)} instead"
        )
        self._size = val


class Markers(BaseProperties):
    """
    Defines the styling properties of plot markers

    Properties
    ----------
    size: float, default=None
        marker size
    color: str, default=None
        marker color
    symbol: str, default=None
        marker symbol. Can be one of `['.', 'o', '+', 'D', 'd', 's', 'x']`

    """

    def __init__(self, size=None, color=None, symbol=None, **kwargs):
        super().__init__(size=size, color=color, symbol=symbol, **kwargs)

    @property
    def size(self):
        """marker size"""
        return self._size

    @size.setter
    def size(self, val):
        assert val is None or isinstance(val, (int, float)) and val >= 0, (
            f"the `size` property of {type(self).__name__} must be a positive number"
            f" but received {repr(val)} instead"
        )
        self._size = val

    @property
    def color(self):
        """marker color"""
        return self._color

    @color.setter
    def color(self, val):
        self._color = color_validator(val)

    @property
    def symbol(self):
        """marker symbol. Can be one of `['.', 'o', '+', 'D', 'd', 's', 'x']`"""
        return self._symbol

    @symbol.setter
    def symbol(self, val):
        assert val is None or val in _SYMBOLS_MATPLOTLIB_TO_PLOTLY, (
            f"the `symbol` property of {type(self).__name__} must be one of"
            f"{list(_SYMBOLS_MATPLOTLIB_TO_PLOTLY.keys())}"
            f" but received {repr(val)} instead"
        )
        self._symbol = val


class MarkersStyle(BaseStyle):
    """
    Defines the styling properties of the markers trace

    Properties
    ----------
    marker: dict, Markers, default=None
        Markers class with 'color', 'symbol', 'size' properties, or dictionary with equivalent
        key/value pairs
    """

    def __init__(self, marker=None, **kwargs):
        super().__init__(marker=marker, **kwargs)

    @property
    def marker(self):
        """Markers class with 'color', 'symbol', 'size' properties"""
        return self._marker

    @marker.setter
    def marker(self, val):
        if isinstance(val, dict):
            val = Markers(**val)
        if isinstance(val, Markers):
            self._marker = val
        elif val is None:
            self._marker = Markers()
        else:
            raise ValueError(
                f"the `marker` property of `{type(self).__name__}` must be an instance \n"
                "of `Markers` or a dictionary with equivalent key/value pairs \n"
                f"but received {repr(val)} instead"
            )


class Dipoles(BaseProperties):
    """
    Defines the specific styling properties of the objects of the `dipoles` family

    Properties
    ----------
    size: float, default=None
        positive float for relative dipole to size to canvas size

    pivot: str, default=None
        the part of the arrow that is anchored to the X, Y grid.
        The arrow rotates about this point. Can be one of `['tail', 'middle', 'tip']`
    """

    def __init__(self, size=None, pivot=None, **kwargs):
        self._allowed_pivots = ("tail", "middle", "tip")
        super().__init__(size=size, pivot=pivot, **kwargs)

    @property
    def size(self):
        """positive float for relative dipole to size to canvas size"""
        return self._size

    @size.setter
    def size(self, val):
        assert val is None or isinstance(val, (int, float)) and val >= 0, (
            f"the `size` property of {type(self).__name__} must be a positive number"
            f" but received {repr(val)} instead"
        )
        self._size = val

    @property
    def pivot(self):
        """The part of the arrow that is anchored to the X, Y grid.
        The arrow rotates about this point. Can be one of `['tail', 'middle', 'tip']`"""
        return self._pivot

    @pivot.setter
    def pivot(self, val):
        assert val is None or val in (self._allowed_pivots), (
            f"the `pivot` property of {type(self).__name__} must be one of "
            f"{self._allowed_pivots}\n"
            f" but received {repr(val)} instead"
        )
        self._pivot = val


class DipoleStyle(BaseGeoStyle, Dipoles):
    """Defines the styling properties of the objects of the `dipoles` family and base properties"""


class PathStyle(BaseProperties):
    """
    Defines the styling properties of an object's path

    Properties
    ----------
    marker: dict, Markers, default=None
        Markers class with 'color', 'symbol', 'size' properties, or dictionary with equivalent
        key/value pairs

    line: dict, LineStyle, default=None
        LineStyle class with 'color', 'symbol', 'size' properties, or dictionary with equivalent
        key/value pairs

    """

    def __init__(self, marker=None, line=None, **kwargs):
        super().__init__(marker=marker, line=line, **kwargs)

    @property
    def marker(self):
        """Markers class with 'color', 'type', 'width' properties"""
        return self._marker

    @marker.setter
    def marker(self, val):
        if isinstance(val, dict):
            val = Markers(**val)
        if isinstance(val, Markers):
            self._marker = val
        elif val is None:
            self._marker = Markers()
        else:
            raise ValueError(
                f"the `marker` property of `{type(self).__name__}` must be an instance \n"
                "of `Markers` or a dictionary with equivalent key/value pairs \n"
                f"but received {repr(val)} instead"
            )

    @property
    def line(self):
        """LineStyle class with 'color', 'type', 'width' properties"""
        return self._line

    @line.setter
    def line(self, val):
        if isinstance(val, dict):
            val = LineStyle(**val)
        if isinstance(val, LineStyle):
            self._line = val
        elif val is None:
            self._line = LineStyle()
        else:
            raise ValueError(
                f"the `line` property of `{type(self).__name__}` must be an instance \n"
                "of `LineStyle` or a dictionary with equivalent key/value pairs \n"
                f"but received {repr(val)} instead"
            )


class LineStyle(BaseProperties):
    """
    Defines Line styling properties

    Properties
    ----------
    style: str, default=None
        Can be one of:
        `['solid', '-', 'dashed', '--', 'dashdot', '-.', 'dotted', '.', (0, (1, 1)),
        'loosely dotted', 'loosely dashdotted']`

    color: str, default=None
        line color

    width: float, default=None
        positive number that defines the line width

    """

    def __init__(self, style=None, color=None, width=None, **kwargs):
        super().__init__(style=style, color=color, width=width, **kwargs)

    @property
    def style(self):
        """line style"""
        return self._style

    @style.setter
    def style(self, val):
        assert val is None or val in _LINESTYLES_MATPLOTLIB_TO_PLOTLY, (
            f"the `style` property of {type(self).__name__} must be one of"
            f"{list(_LINESTYLES_MATPLOTLIB_TO_PLOTLY.keys())}"
            f" but received {repr(val)} instead"
        )
        self._style = val

    @property
    def color(self):
        """line color"""
        return self._color

    @color.setter
    def color(self, val):
        self._color = color_validator(val)

    @property
    def width(self):
        """positive number that defines the line width"""
        return self._width

    @width.setter
    def width(self, val):
        assert val is None or isinstance(val, (int, float)) and val >= 0, (
            f"the `width` property of {type(self).__name__} must be a positive number"
            f" but received {repr(val)} instead"
        )
        self._width = val


class MagpylibStyle(BaseProperties):
    """
    Base class containing styling properties for all object families. The properties of the
    sub-classes get set to hard coded defaults at class instantiation

    Properties
    ----------
    base: dict, BaseGeoStyle, default=None
        base properties common to all families

    magnets: dict, Magnets, default=None
        magnets properties

    currents: dict, Currents, default=None
        currents properties

    dipoles: dict, Dipoles, default=None
        dipoles properties

    sensors: dict, Sensors, default=None
        sensors properties

    markers: dict, Markers, default=None
        markers properties
    """

    def __init__(
        self,
        base=None,
        magnets=None,
        currents=None,
        dipoles=None,
        sensors=None,
        markers=None,
        **kwargs,
    ):
        super().__init__(
            base=base,
            magnets=magnets,
            currents=currents,
            dipoles=dipoles,
            sensors=sensors,
            markers=markers,
            **kwargs,
        )
        self.reset()

    def reset(self):
        """Resets all nested properties to their hard coded default values"""
        self.update(_DEFAULT_STYLES, _match_properties=False)

    @property
    def base(self):
        """BaseStyle class with 'color', 'type', 'width' properties"""
        return self._base

    @base.setter
    def base(self, val):
        if isinstance(val, dict):
            val = BaseGeoStyle(**val)
        if isinstance(val, BaseGeoStyle):
            self._base = val
        elif val is None:
            self._base = BaseGeoStyle()
        else:
            raise ValueError(
                f"the `base` property of `{type(self).__name__}` must be an instance \n"
                "of `BaseGeoStyle` or a dictionary with equivalent key/value pairs \n"
                f"but received {repr(val)} instead"
            )

    @property
    def magnets(self):
        """MagnetStyle class with 'color', 'type', 'width' properties"""
        return self._magnets

    @magnets.setter
    def magnets(self, val):
        if isinstance(val, dict):
            val = Magnets(**val)
        if isinstance(val, Magnets):
            self._magnets = val
        elif val is None:
            self._magnets = Magnets()
        else:
            raise ValueError(
                f"the `magnets` property of `{type(self).__name__}` must be an instance \n"
                "of `Magnets` or a dictionary with equivalent key/value pairs \n"
                f"but received {repr(val)} instead"
            )

    @property
    def currents(self):
        """Currents class with 'color', 'type', 'width' properties"""
        return self._currents

    @currents.setter
    def currents(self, val):
        if isinstance(val, dict):
            val = Currents(**val)
        if isinstance(val, Currents):
            self._currents = val
        elif val is None:
            self._currents = Currents()
        else:
            raise ValueError(
                f"the `currents` property of `{type(self).__name__}` must be an instance \n"
                "of `Currents` or a dictionary with equivalent key/value pairs \n"
                f"but received {repr(val)} instead"
            )

    @property
    def dipoles(self):
        """Dipoles class with 'color', 'type', 'width' properties"""
        return self._dipoles

    @dipoles.setter
    def dipoles(self, val):
        if isinstance(val, dict):
            val = Dipoles(**val)
        if isinstance(val, Dipoles):
            self._dipoles = val
        elif val is None:
            self._dipoles = Dipoles()
        else:
            raise ValueError(
                f"the `dipoles` property of `{type(self).__name__}` must be an instance \n"
                "of `Dipoles` or a dictionary with equivalent key/value pairs \n"
                f"but received {repr(val)} instead"
            )

    @property
    def sensors(self):
        """Sensors class with 'color', 'type', 'width' properties"""
        return self._sensors

    @sensors.setter
    def sensors(self, val):
        if isinstance(val, dict):
            val = Sensors(**val)
        if isinstance(val, Sensors):
            self._sensors = val
        elif val is None:
            self._sensors = Sensors()
        else:
            raise ValueError(
                f"the `sensors` property of `{type(self).__name__}` must be an instance \n"
                "of `Sensors` or a dictionary with equivalent key/value pairs \n"
                f"but received {repr(val)} instead"
            )

    @property
    def markers(self):
        """Markers class with 'color', 'type', 'width' properties"""
        return self._markers

    @markers.setter
    def markers(self, val):
        if isinstance(val, dict):
            val = Markers(**val)
        if isinstance(val, Markers):
            self._markers = val
        elif val is None:
            self._markers = Markers()
        else:
            raise ValueError(
                f"the `markers` property of `{type(self).__name__}` must be an instance \n"
                "of `Markers` or a dictionary with equivalent key/value pairs \n"
                f"but received {repr(val)} instead"
            )


default_style = MagpylibStyle()

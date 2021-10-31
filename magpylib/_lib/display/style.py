"""Collection of class for display styles"""
# pylint: disable=C0302

import collections.abc
from magpylib._lib.config import Config

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

_DEFAULT_STYLES = {
    "sensors": {"size": 1, "pixel": {"size": 1, "color": "grey"}},
    "dipoles": {"size": 1, "pivot": 'middle'},
    "currents": {"current": {"show": True, "size": 1}},
    "markers": {"marker": {"size": 2, "color": "grey", "symbol": "x"}},
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
    "base": {
        "path": {
            "line": {"width": 1, "style": "solid"},
            "marker": {"size": 1, "symbol": "o"},
        },
        "description": True,
        "opacity": 1,
        "mesh3d": None,
    },
}


def update_nested_dict(d, u, same_keys_only=False, replace_None_only=False):
    """updates recursively 'd' from 'u'"""
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


def magic_to_dict(kwargs):
    """
    decomposes recursively a dictionary with keys with underscores into a nested dictionary
    example : {'magnet_color':'blue'} -> {'magnet': {'color':'blue'}}
    see: https://plotly.com/python/creating-and-updating-figures/#magic-underscore-notation
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


def color_validator(color_input, allow_None=True, parent_name=""):
    """validates color inputs, allows `None` by default"""
    if not allow_None or color_input is not None:
        # pylint: disable=import-outside-toplevel
        if Config.BACKEND == "plotly":
            from _plotly_utils.basevalidators import ColorValidator

            cv = ColorValidator(plotly_name="color", parent_name=parent_name)
            color_input = cv.validate_coerce(color_input)
        else:
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
                    f"""received for the color property of {parent_name})"""
                    f"""\n   Received value: '{color_input}'"""
                    f"""\n\nThe 'color' property is a color and may be specified as:\n"""
                    """    - A hex string (e.g. '#ff0000')\n"""
                    f"""    - A named CSS color:\n{list(mcolors.keys())}"""
                )
    return color_input


def get_style(obj, **kwargs):
    """
    returns default style object based on increasing priority:
    - style from Config
    - style from object
    - style from style_kwargs
    """

    style_kwargs = kwargs.get("style", {})
    style_kwargs.update(
        {k[6:]: v for k, v in kwargs.items() if k.startswith("style") and k != "style"}
    )

    styles_by_familly = default_style.as_dict()
    objects_families = {
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

    obj_type = getattr(obj, "_object_type", None)
    obj_families = objects_families.get(obj_type, [])

    obj_style_dict = {
        **styles_by_familly["base"],
        **{
            k: v
            for fam in obj_families
            for k, v in styles_by_familly.get(fam, {}).items()
        },
    }

    obj_style = getattr(obj, "style", None)

    style = obj_style.copy() if obj_style is not None else BaseStyle()
    style.update(**style_kwargs, _match_properties=False)
    style.update(**obj_style_dict, _match_properties=False, _replace_None_only=True)

    return style


class BaseStyleProperties:
    """Base Class to represent only the property attributes"""

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
        updates the class properties with provided arguments, supports magic underscore
        returns self
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


class BaseStyle(BaseStyleProperties):
    """
    Base class for display styling options of BaseGeo objects

    Properties
    ----------
    color: str, default=None
        css color

    opacity: float, default=1
        object opacity between 0 and 1

    mesh3d: plotly.graph_objects.Mesh3d, default=None
        can be set trough dict of compatible keys
    """

    def __init__(
        self,
        name=None,
        description=None,
        color=None,
        opacity=None,
        mesh3d=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            color=color,
            opacity=opacity,
            mesh3d=mesh3d,
            **kwargs,
        )

    @property
    def name(self):
        """name of object"""
        return self._name

    @name.setter
    def name(self, val):
        self._name = val if val is None else str(val)

    @property
    def description(self):
        """
        object description. Adds legend entry suffix based on value:
        - True: base object dimension are shown
        - False: no suffix is shown
        - str: user string is shown
        """
        return self._description

    @description.setter
    def description(self, val):
        self._description = val if isinstance(val, bool) or val is None else str(val)

    @property
    def color(self):
        """css color"""
        return self._color

    @color.setter
    def color(self, val):
        self._color = color_validator(val, parent_name=f"{type(self).__name__}")

    @property
    def opacity(self):
        """opacity float between 0 and 1"""
        return self._opacity

    @opacity.setter
    def opacity(self, val):
        assert val is None or (
            isinstance(val, (float, int)) and val >= 0 and val <= 1
        ), "opacity must be a value betwen 0 and 1"
        self._opacity = val

    @property
    def path(self):
        """MagColor class with 'north', 'south', 'middle' and 'transition' values"""
        return self._path

    @path.setter
    def path(self, val):
        if isinstance(val, dict):
            val = PathTraceStyle(**val)
        if isinstance(val, PathTraceStyle):
            self._path = val
        elif val is None:
            self._path = PathTraceStyle()
        else:
            raise ValueError(
                "the path property must be an instance "
                "of PathTraceStyle or a dictionary with equivalent key/value pairs"
            )

    @property
    def mesh3d(self):
        """MagColor class with 'north', 'south', 'middle', 'transition' and 'show' values"""
        return self._mesh3d

    @mesh3d.setter
    def mesh3d(self, val):
        if isinstance(val, dict):
            val = Mesh3dStyle(**val)
        if isinstance(val, Mesh3dStyle):
            self._mesh3d = val
        elif val is None:
            self._mesh3d = Mesh3dStyle()
        else:
            raise ValueError(
                "the mesh3d property must be an instance "
                "of Mesh3dStyle or a dictionary with equivalent key/value pairs"
            )


class Mesh3dStyle(BaseStyleProperties):
    """
    Mesh3d styling properties
    """

    def __init__(self, data=None, show=None, **kwargs):
        super().__init__(data=data, show=show, **kwargs)

    @property
    def show(self):
        """
        Shows/hides mesh3d object based on provided data:
        - True: shows mesh
        - False: hides mesh
        - 'inplace': replace object representation
        """
        return self._show

    @show.setter
    def show(self, val):
        assert (
            val is None or isinstance(val, bool) or val == "inplace"
        ), "show must be one of [`True`, `False`, `'inplace'`]"
        self._show = val

    @property
    def data(self):
        """plotly.graph_objects.Mesh3d or equivalent dict"""
        return self._data

    @data.setter
    def data(self, val):
        # pylint: disable=import-outside-toplevel
        assert val is None or all(
            key in val for key in "xyzijk"
        ), "data must be a dict-like object containing the `x,y,z,i,j,k` keys/values pairs"
        self._data = val


class MagnetizationStyle(BaseStyleProperties):
    """This class holds magnetization styling properties
    - size: arrow size for matplotlib backend
    - color: magnetization colors of the poles
    """

    def __init__(self, show=None, size=None, color=None, **kwargs):
        super().__init__(show=show, size=size, color=color, **kwargs)

    @property
    def show(self):
        """show magnetization direction through poles colors"""
        return self._show

    @show.setter
    def show(self, val):
        assert val is None or isinstance(
            val, bool
        ), "show must be either `True` or `False`"
        self._show = val

    @property
    def size(self):
        """positive float for relative arrow size to magnet size"""
        return self._size

    @size.setter
    def size(self, val):
        assert (
            val is None or isinstance(val, (int, float)) and val >= 0
        ), "size must be a positive number"
        self._size = val

    @property
    def color(self):
        """MagColor class with 'north', 'south', 'middle', 'transition' and 'show' values"""
        return self._color

    @color.setter
    def color(self, val):
        if isinstance(val, dict):
            val = MagColor(**val)
        if isinstance(val, MagColor):
            self._color = val
        elif val is None:
            self._color = MagColor()
        else:
            raise ValueError(
                "the magnetic color property must be an instance "
                "of MagColor or a dictionary with equivalent key/value pairs"
            )


class MagColor(BaseStyleProperties):
    """
    This class defines the magnetization color styling:
        - north
        - south
        - middle
        - transition
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
        """css color"""
        return self._north

    @north.setter
    def north(self, val):
        self._north = color_validator(val)

    @property
    def south(self):
        """css color"""
        return self._south

    @south.setter
    def south(self, val):
        self._south = color_validator(val)

    @property
    def middle(self):
        """css color"""
        return self._middle

    @middle.setter
    def middle(self, val):
        if val != "auto" and not val is False:
            val = color_validator(val)
        self._middle = val

    @property
    def transition(self):
        """sets the transition smoothness between poles colors"""
        return self._transition

    @transition.setter
    def transition(self, val):
        assert (
            val is None or isinstance(val, (float, int)) and val >= 0 and val <= 1
        ), "color transition must be a value betwen 0 and 1"
        self._transition = val


class Magnets(BaseStyleProperties):
    """
    Style class for display styling options of Magnet objects
    """

    def __init__(self, magnetization=None, **kwargs):
        super().__init__(magnetization=magnetization, **kwargs)

    @property
    def magnetization(self):
        """MagColor class with 'north', 'south', 'middle' and 'transition' values"""
        return self._magnetization

    @magnetization.setter
    def magnetization(self, val):
        if isinstance(val, dict):
            val = MagnetizationStyle(**val)
        if isinstance(val, MagnetizationStyle):
            self._magnetization = val
        elif val is None:
            self._magnetization = MagnetizationStyle()
        else:
            raise ValueError(
                "the magnetic color property must be an instance "
                "of Mag or a dictionary with equivalent key/value pairs"
            )


class MagnetStyle(BaseStyle, Magnets):
    pass


class Sensors(BaseStyleProperties):
    """
    This class holds Sensor styling properties
    - size:  size relative to canvas
    - color: sensor color
    - pixel: pixel properties, see PixelStyle
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
        """MagColor class with 'north', 'south', 'middle' and 'transition' values"""
        return self._pixel

    @pixel.setter
    def pixel(self, val):
        if isinstance(val, dict):
            val = PixelStyle(**val)
        if isinstance(val, PixelStyle):
            self._pixel = val
        elif val is None:
            self._pixel = PixelStyle()
        else:
            raise ValueError(
                "the pixel property must be an instance "
                "of PixelStyle or a dictionary with equivalent key/value pairs"
            )


class SensorStyle(BaseStyle, Sensors):
    pass
class PixelStyle(BaseStyleProperties):
    """
    This class holds sensor pixel styling properties
    - size: relative pixel size to the min distance between two pixels
    - color: pixel color
    """

    def __init__(self, size=1, color=None, **kwargs):
        super().__init__(size=size, color=color, **kwargs)

    @property
    def size(self):
        """positive float, relative pixel size to the min distance between two pixels"""
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
        """css color"""
        return self._color

    @color.setter
    def color(self, val):
        self._color = color_validator(val, parent_name=f"{type(self).__name__}")

class Currents(BaseStyleProperties):
    """
    This class holds styling properties for Line and Circular currents
    - show: if True current directin is shown with an arrow
    - size: defines the size of the arrows
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
                "the current property must be an instance"
                "of style.ArrowStyle or a dictionary with equivalent key/value pairs"
            )


class CurrentStyle(BaseStyle, Currents):
    pass

class ArrowStyle(BaseStyleProperties):
    """
    This class holds styling properties for Line and Circular currents
    - show: if True current directin is shown with an arrow
    - size: defines the size of the arrows
    """

    def __init__(self, show=None, size=None, **kwargs):
        super().__init__(show=show, size=size, **kwargs)

    @property
    def show(self):
        """show/hide current show arrow"""
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
        """positive float for relative arrow size"""
        return self._size

    @size.setter
    def size(self, val):
        assert val is None or isinstance(val, (int, float)) and val >= 0, (
            f"the `size` property of {type(self).__name__} must be a positive number"
            f" but received {repr(val)} instead"
        )
        self._size = val


class MarkerStyle(BaseStyleProperties):
    """
    This class holds marker styling properties
    - size: marker size
    - color: marker color
    - symbol: marker symbol
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
        """css color"""
        return self._color

    @color.setter
    def color(self, val):
        # wrong value will be handeled by the respective libraries since
        # value only gets created at plot creation.
        self._color = val

    @property
    def symbol(self):
        """compatible symbol string for matplotlib or plotly"""
        return self._symbol

    @symbol.setter
    def symbol(self, val):
        assert val is None or val in _SYMBOLS_MATPLOTLIB_TO_PLOTLY, (
            f"the `symbol` property of {type(self).__name__} must be one of"
            f"{list(_SYMBOLS_MATPLOTLIB_TO_PLOTLY.keys())}"
            f" but received {repr(val)} instead"
        )
        self._symbol = val


class MarkerTraceStyle(BaseStyleProperties):
    """
    This class holds marker styling properties
    - marker: MarkerStyle class
    - opacity: trace opacity
    """

    def __init__(self, marker=None, opacity=None, **kwargs):
        super().__init__(marker=marker, opacity=opacity, **kwargs)

    @property
    def marker(self):
        """MarkerStyle class with 'color', 'symbol', 'size' values"""
        return self._marker

    @marker.setter
    def marker(self, val):
        if isinstance(val, dict):
            val = MarkerStyle(**val)
        if isinstance(val, MarkerStyle):
            self._marker = val
        elif val is None:
            self._marker = MarkerStyle()
        else:
            raise ValueError(
                "the marker property must be an instance"
                "of style.MarkerStyle or a dictionary with equivalent key/value pairs"
            )

    @property
    def opacity(self):
        """opacity float between 0 and 1"""
        return self._opacity

    @opacity.setter
    def opacity(self, val):
        assert val is None or (
            isinstance(val, (float, int)) and val >= 0 and val <= 1
        ), "opacity must be a value betwen 0 and 1"
        self._opacity = val


class Dipoles(BaseStyleProperties):
    """
    This class holds Dipole styling properties
    - size:  size relative to canvas
    """

    def __init__(self, size=None, **kwargs):
        self._allowed_pivots = ('tail', 'mid', 'middle', 'tip')
        super().__init__(size=size, **kwargs)

    @property
    def size(self):
        """positive float for relative sensor to size to canvas size"""
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
        The arrow rotates about this point"""
        return self._pivot

    @pivot.setter
    def pivot(self, val):
        assert val is None or val in (self._allowed_pivots), (
            f"the `pivot` property of {type(self).__name__} must be one of "
            f"{self._allowed_pivots}"
            f" but received {repr(val)} instead"
        )
        self._pivot = val

class DipoleStyle(BaseStyle, Dipoles):
    pass

class PathTraceStyle(BaseStyleProperties):
    """
    This class holds marker styling properties
    - marker: MarkerStyle class
    """

    def __init__(self, marker=None, line=None, **kwargs):
        super().__init__(marker=marker, line=line, **kwargs)

    @property
    def marker(self):
        """MarkerStyle class with 'color', 'symbol', 'size' properties"""
        return self._marker

    @marker.setter
    def marker(self, val):
        if isinstance(val, dict):
            val = MarkerStyle(**val)
        if isinstance(val, MarkerStyle):
            self._marker = val
        elif val is None:
            self._marker = MarkerStyle()
        else:
            raise ValueError(
                "the marker property must be an instance"
                "of style.MarkerStyle or a dictionary with equivalent key/value pairs"
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
                "the line property must be an instance"
                "of style.LineStyle or a dictionary with equivalent key/value pairs"
            )


class LineStyle(BaseStyleProperties):
    """
    This class holds Line styling properties
    - style: line style (linestyle in matplotlib and line_dash in plotly)
    - color: line color
    - width: line width
    """

    def __init__(self, style=None, color=None, width=None, **kwargs):
        super().__init__(style=style, color=color, width=width, **kwargs)

    @property
    def style(self):
        """marker style"""
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
        """css color"""
        return self._color

    @color.setter
    def color(self, val):
        # wrong value will be handeled by the respective libraries since
        # value only gets created at plot creation.
        self._color = val

    @property
    def width(self):
        """compatible width string for matplotlib or plotly"""
        return self._width

    @width.setter
    def width(self, val):
        assert val is None or isinstance(val, (int, float)) and val >= 0, (
            f"the `width` property of {type(self).__name__} must be a positive number"
            f" but received {repr(val)} instead"
        )
        self._width = val


class MagpylibStyle(BaseStyleProperties):
    """Base class containing style properties for all object famillies"""

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
        """Resets all nested properties to their default values"""
        self.update(_DEFAULT_STYLES, _match_properties=False)

    @property
    def base(self):
        """BaseStyle class with 'color', 'type', 'width' properties"""
        return self._base

    @base.setter
    def base(self, val):
        if isinstance(val, dict):
            val = BaseStyle(**val)
        if isinstance(val, BaseStyle):
            self._base = val
        elif val is None:
            self._base = BaseStyle()
        else:
            raise ValueError(
                "the base property must be an instance"
                "of style.BaseStyle or a dictionary with equivalent key/value pairs"
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
                "the magnets property must be an instance"
                "of style.Magnets or a dictionary with equivalent key/value pairs"
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
                "the currents property must be an instance"
                "of style.Currents or a dictionary with equivalent key/value pairs"
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
                "the dipoles property must be an instance"
                "of style.Dipoles or a dictionary with equivalent key/value pairs"
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
                "the sensors property must be an instance"
                "of style.Sensors or a dictionary with equivalent key/value pairs"
            )

    @property
    def markers(self):
        """MarkerStyle class with 'color', 'type', 'width' properties"""
        return self._markers

    @markers.setter
    def markers(self, val):
        if isinstance(val, dict):
            val = MarkerStyle(**val)
        if isinstance(val, MarkerStyle):
            self._markers = val
        elif val is None:
            self._markers = MarkerStyle()
        else:
            raise ValueError(
                "the markers property must be an instance"
                "of style.MarkerStyle or a dictionary with equivalent key/value pairs"
            )

default_style = MagpylibStyle()

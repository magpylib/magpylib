"""Collection of class for display styles"""
import collections.abc
from magpylib._lib.config import Config


def update_nested_dict(d, u, same_keys_only=False):
    """updates recursively 'd' from 'u'"""
    for k, v in u.items():
        if k in d or not same_keys_only:
            if isinstance(d, collections.abc.Mapping):
                if isinstance(v, collections.abc.Mapping):
                    r = update_nested_dict(d.get(k, {}), v)
                    d[k] = r
                else:
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
            new_kwargs[keys[0]] = magic_to_dict(val)
    return new_kwargs


def color_validator(color_input, allow_None=True, parent_name=""):
    """validates color inputs, allows `None` by default"""
    if not allow_None or color_input is not None:
        # pylint: disable=import-outside-toplevel
        if Config.PLOTTING_BACKEND == "plotly":
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

    def get_properties_dict(self):
        """returns recursively a nested dictionary with all properties objects of the class"""
        params = self._property_names_generator()
        dict_ = {}
        for k in params:
            val = getattr(self, k)
            if hasattr(val, "get_properties_dict"):
                dict_[k] = val.get_properties_dict()
            else:
                dict_[k] = val
        return dict_

    def update(self, arg=None, _match_properties=True, **kwargs):
        """
        updates the class properties with provided arguments, supports magic underscore
        returns self
        """
        if arg is None:
            arg = {}
        if kwargs:
            arg.update(magic_to_dict(kwargs))
        current_dict = self.get_properties_dict()
        new_dict = update_nested_dict(
            current_dict, arg, same_keys_only=_match_properties
        )
        for k, v in new_dict.items():
            setattr(self, k, v)
        return self

    def copy(self):
        """returns a copy of the current class instance"""
        return type(self)(**self.get_properties_dict())


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
        if val is None:
            val = Config.AUTO_DESCRIPTION
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
    def mesh3d(self):
        """plotly.graph_objects.Mesh3d instance"""
        return self._mesh3d

    @mesh3d.setter
    def mesh3d(self, val):
        # pylint: disable=import-outside-toplevel
        if val is None:
            self._mesh3d = val
        else:
            import plotly.graph_objects as go

            self._mesh3d = go.Mesh3d(**val)


class MagStyle(BaseStyleProperties):
    """This class holds magnetization styling properties
    - size: arrow size for matplotlib backend
    - color: magnetization colors of the poles
    """

    def __init__(self, size=None, color=None, **kwargs):
        super().__init__(size=size, color=color, **kwargs)

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

    def __init__(
        self, north=None, south=None, middle=None, transition=None, show=True, **kwargs
    ):
        super().__init__(
            north=north,
            middle=middle,
            south=south,
            transition=transition,
            show=show,
            **kwargs,
        )

    @property
    def north(self):
        """css color"""
        return self._north

    @north.setter
    def north(self, val):
        if val is None:
            val = Config.COLOR_NORTH
        self._north = color_validator(val)

    @property
    def south(self):
        """css color"""
        return self._south

    @south.setter
    def south(self, val):
        if val is None:
            val = Config.COLOR_SOUTH
        self._south = color_validator(val)

    @property
    def middle(self):
        """css color"""
        return self._middle

    @middle.setter
    def middle(self, val):
        if val is None:
            val = Config.COLOR_MIDDLE
        if val != "auto" and not val is False:
            val = color_validator(val)
        self._middle = val

    @property
    def show(self):
        """css color"""
        return self._show

    @show.setter
    def show(self, val):
        assert isinstance(val, bool), "show must be either `True` or `False`"
        self._show = val

    @property
    def transition(self):
        """sets the transition smoothness between poles colors"""
        return self._transition

    @transition.setter
    def transition(self, val):
        if val is None:
            val = Config.COLOR_TRANSITION
        assert (
            isinstance(val, (float, int)) and val >= 0 and val <= 1
        ), "color transition must be a value betwen 0 and 1"
        self._transition = val


class MagnetStyle(BaseStyle):
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
            val = MagStyle(**val)
        if isinstance(val, MagStyle):
            self._magnetization = val
        elif val is None:
            self._magnetization = MagStyle()
        else:
            raise ValueError(
                "the magnetic color property must be an instance "
                "of Mag or a dictionary with equivalent key/value pairs"
            )


class SensorStyle(BaseStyle):
    """
    This class holds Sensor styling properties
    - size:  size relative to canvas
    - color: sensor color
    - pixel: pixel properties, see PixelStyle
    """

    def __init__(self, pixel=None, **kwargs):
        super().__init__(pixel=pixel, **kwargs)

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
        """css color"""
        return self._color

    @color.setter
    def color(self, val):
        if val is None:
            val = Config.PIXEL_COLOR
        self._color = color_validator(val, parent_name=f"{type(self).__name__}")

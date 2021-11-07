"""Collection of class for display styles"""
# pylint: disable=C0302

from magpylib._lib.default_utils import (
    BaseProperties,
    color_validator,
    get_defaults_dict,
    SYMBOLS_MATPLOTLIB_TO_PLOTLY,
    LINESTYLES_MATPLOTLIB_TO_PLOTLY,
    MAGPYLIB_FAMILIES,
)


def get_style_class(obj):
    """returns style class based on object type. If class has no attribute `_object_type` or is
    not found in `MAGPYLIB_FAMILIES` returns `BaseStyle` class."""
    obj_type = getattr(obj, "_object_type", None)
    style_fam = MAGPYLIB_FAMILIES.get(obj_type, None)
    if isinstance(style_fam, (list, tuple)):
        style_fam = style_fam[0]
    return STYLE_CLASSES.get(style_fam, BaseStyle)


def get_style(obj, default_settings, **kwargs):
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

    # retrieve default style dictionary, local import to avoid circular import
    # pylint: disable=import-outside-toplevel

    styles_by_family = default_settings.display.styles.as_dict()

    # construct object specific dictionary base on style family and default style
    obj_type = getattr(obj, "_object_type", None)
    obj_families = MAGPYLIB_FAMILIES.get(obj_type, [])

    obj_style_default_dict = {
        **styles_by_family["base"],
        **{
            k: v
            for fam in obj_families
            for k, v in styles_by_family.get(fam, {}).items()
        },
    }
    valid_keys = {key for k, v in styles_by_family.items() for key in v}
    level0_style_keys = {k.split("_")[0]: k for k in style_kwargs}
    kwargs_diff = set(level0_style_keys).difference(valid_keys)
    invalid_keys = {level0_style_keys[k] for k in kwargs_diff}
    if invalid_keys:
        raise ValueError(
            f"Following arguments are invalid style properties: {invalid_keys}\n"
            f"\n Available style properties start with `style_` + `{valid_keys}`"

        )
    # create style class instance and update based on precedence
    obj_style = getattr(obj, "style", None)
    style = obj_style.copy() if obj_style is not None else Base()
    style_kwargs_specific = {
        k: v
        for k, v in style_kwargs.items()
        if k.split("_")[0] in style.as_dict()
    }
    style.update(**style_kwargs_specific, _match_properties=True)
    style.update(
        **obj_style_default_dict, _match_properties=False, _replace_None_only=True
    )

    return style

def validate_property_class(val, name, class_, parent):
    '''validator for sub property'''
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
        self._description = validate_property_class(val, 'description', Description, self)

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
        assert val is None or (isinstance(val, (float, int)) and 0 <= val <= 1), (
            "opacity must be a value betwen 0 and 1\n"
            f"but received {repr(val)} instead"
        )
        self._opacity = val


class Base(BaseStyle):
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

    path: dict or Path, default=None
        an instance of `Path` or dictionary of equivalent key/value pairs, defining the object
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
        """an instance of `Path` or dictionary of equivalent key/value pairs, defining the
        object path marker and path line properties"""
        return self._path

    @path.setter
    def path(self, val):
        self._path = validate_property_class(val, 'path', Path, self)

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
        self._mesh3d = validate_property_class(val, 'mesh3d', Mesh3d, self)


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
        self._color = validate_property_class(val, 'color', MagnetizationColor, self)


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
            val is None or isinstance(val, (float, int)) and 0 <= val <= 1
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
        self._magnetization = validate_property_class(val, 'magnetization', Magnetization, self)


class MagnetStyle(Base, Magnets):
    """Defines the styling properties of objects of the `magnets` family with base properties"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        self._pixel = validate_property_class(val, 'pixel', Pixel, self)



class SensorStyle(Base, Sensors):
    """Defines the styling properties of objects of the `sensors` family with base properties"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Pixel(BaseProperties):
    """
    Defines the styling properties of sensor pixels

    Properties
    ----------
    size: float, default=None
        defines the relative pixel size:
        - matplotlib backend: pixel size is the marker size
        - plotly backend:  relative size to the distance of nearest neighboring pixel

    color: str, default=None
        defines the pixel color@property

    symbol: str, default=None
        pixel symbol. Can be one of `['.', 'o', '+', 'D', 'd', 's', 'x']`
        Only applies for matplotlib.

    """

    def __init__(self, size=1, color=None, symbol=None, **kwargs):
        super().__init__(size=size, color=color, symbol=symbol, **kwargs)

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

    @property
    def symbol(self):
        """pixel symbol. Can be one of `['.', 'o', '+', 'D', 'd', 's', 'x']`"""
        return self._symbol

    @symbol.setter
    def symbol(self, val):
        assert val is None or val in SYMBOLS_MATPLOTLIB_TO_PLOTLY, (
            f"the `symbol` property of {type(self).__name__} must be one of"
            f"{list(SYMBOLS_MATPLOTLIB_TO_PLOTLY.keys())}"
            f" but received {repr(val)} instead"
        )
        self._symbol = val


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
        """Arrow class with 'show', 'size' properties"""
        return self._current

    @current.setter
    def current(self, val):
        self._current = validate_property_class(val, 'current', Arrow, self)


class CurrentStyle(Base, Currents):
    """Defines the styling properties of objects of the `currents` family and base properties"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Arrow(BaseProperties):
    """
    Defines the styling properties of current arrows

    Properties
    ----------
    show: bool, default=None
        if `True` current direction is shown with an arrow

    size: float
        positive number defining the size of the arrows

    width: float, default=None
        positive number that defines the arrow line width
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

    @property
    def width(self):
        """positive number that defines the arrow line width"""
        return self._width

    @width.setter
    def width(self, val):
        assert val is None or isinstance(val, (int, float)) and val >= 0, (
            f"the `width` property of {type(self).__name__} must be a positive number"
            f" but received {repr(val)} instead"
        )
        self._width = val


class Marker(BaseProperties):
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
        assert val is None or val in SYMBOLS_MATPLOTLIB_TO_PLOTLY, (
            f"the `symbol` property of {type(self).__name__} must be one of"
            f"{list(SYMBOLS_MATPLOTLIB_TO_PLOTLY.keys())}"
            f" but received {repr(val)} instead"
        )
        self._symbol = val


class Markers(BaseStyle):
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
        self._marker = validate_property_class(val, 'marker', Marker, self)


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


class DipoleStyle(Base, Dipoles):
    """Defines the styling properties of the objects of the `dipoles` family and base properties"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Path(BaseProperties):
    """
    Defines the styling properties of an object's path

    Properties
    ----------
    marker: dict, Markers, default=None
        Markers class with 'color', 'symbol', 'size' properties, or dictionary with equivalent
        key/value pairs

    line: dict, Line, default=None
        Line class with 'color', 'symbol', 'size' properties, or dictionary with equivalent
        key/value pairs

    """

    def __init__(self, marker=None, line=None, **kwargs):
        super().__init__(marker=marker, line=line, **kwargs)

    @property
    def marker(self):
        """Markers class with 'color', 'symbol', 'size' properties"""
        return self._marker

    @marker.setter
    def marker(self, val):
        self._marker = validate_property_class(val, 'marker', Marker, self)

    @property
    def line(self):
        """Line class with 'color', 'type', 'width' properties"""
        return self._line

    @line.setter
    def line(self, val):
        self._line = validate_property_class(val, 'line', Line, self)


class Line(BaseProperties):
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
        assert val is None or val in LINESTYLES_MATPLOTLIB_TO_PLOTLY, (
            f"the `style` property of {type(self).__name__} must be one of"
            f"{list(LINESTYLES_MATPLOTLIB_TO_PLOTLY.keys())}"
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
    base: dict, Base, default=None
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
        # self.reset()

    def reset(self):
        """Resets all nested properties to their hard coded default values"""
        self.update(get_defaults_dict("display.styles"), _match_properties=False)
        return self

    @property
    def base(self):
        """base properties common to all families"""
        return self._base

    @base.setter
    def base(self, val):
        self._base = validate_property_class(val, 'base', Base, self)

    @property
    def magnets(self):
        """MagnetStyle class"""
        return self._magnets

    @magnets.setter
    def magnets(self, val):
        self._magnets = validate_property_class(val, 'magnets', Magnets, self)

    @property
    def currents(self):
        """Currents class"""
        return self._currents

    @currents.setter
    def currents(self, val):
        self._currents = validate_property_class(val, 'currents', Currents, self)

    @property
    def dipoles(self):
        """Dipoles class"""
        return self._dipoles

    @dipoles.setter
    def dipoles(self, val):
        self._dipoles = validate_property_class(val, 'dipoles', Dipoles, self)

    @property
    def sensors(self):
        """Sensors"""
        return self._sensors

    @sensors.setter
    def sensors(self, val):
        self._sensors = validate_property_class(val, 'sensors', Sensors, self)

    @property
    def markers(self):
        """Markers class"""
        return self._markers

    @markers.setter
    def markers(self, val):
        self._markers = validate_property_class(val, 'markers', Markers, self)


STYLE_CLASSES = {
    "magnets": MagnetStyle,
    "currents": CurrentStyle,
    "dipoles": DipoleStyle,
    "sensors": SensorStyle,
}

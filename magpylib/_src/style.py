"""Collection of classes for display styling"""
# pylint: disable=C0302
import numpy as np

from magpylib._src.defaults.defaults_utility import (
    MagicProperties,
    validate_property_class,
    color_validator,
    get_defaults_dict,
    validate_style_keys,
    SYMBOLS_MATPLOTLIB_TO_PLOTLY,
    LINESTYLES_MATPLOTLIB_TO_PLOTLY,
    MAGPYLIB_FAMILIES,
    SUPPORTED_PLOTTING_BACKENDS,
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
    - style from defaults
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

    styles_by_family = default_settings.display.style.as_dict()

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
    style_kwargs = validate_style_keys(style_kwargs)
    # create style class instance and update based on precedence
    obj_style = getattr(obj, "style", None)
    style = obj_style.copy() if obj_style is not None else BaseStyle()
    style_kwargs_specific = {
        k: v for k, v in style_kwargs.items() if k.split("_")[0] in style.as_dict()
    }
    style.update(**style_kwargs_specific, _match_properties=True)
    style.update(
        **obj_style_default_dict, _match_properties=False, _replace_None_only=True
    )

    return style


class BaseStyle(MagicProperties):
    """
    Base class for display styling options of `BaseGeo` objects

    Parameters
    ----------
    label: str, default=None
        label of the class instance, can be any string.

    description: dict or Description, default=None
        object description properties

    color: str, default=None
        a valid css color. Can also be one of `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`

    opacity: float, default=None
        object opacity between 0 and 1, where 1 is fully opaque and 0 is fully transparent.

    path: dict or Path, default=None
        an instance of `Path` or dictionary of equivalent key/value pairs, defining the object
        path marker and path line properties.

    model3d: list of Trace3d objects, default=None
        a list of traces where each is an instance of `Trace3d` or dictionary of equivalent
        key/value pairs. Defines properties for an additional user-defined model3d object which is
        positioned relatively to the main object to be displayed and moved automatically with it.
        This feature also allows the user to replace the original 3d representation of the object.
    """

    def __init__(
        self,
        label=None,
        description=None,
        color=None,
        opacity=None,
        path=None,
        model3d=None,
        **kwargs,
    ):
        super().__init__(
            label=label,
            description=description,
            color=color,
            opacity=opacity,
            path=path,
            model3d=model3d,
            **kwargs,
        )

    @property
    def label(self):
        """label of the class instance, can be any string"""
        return self._label

    @label.setter
    def label(self, val):
        self._label = val if val is None else str(val)

    @property
    def description(self):
        """Description class with 'text' and 'show' properties"""
        return self._description

    @description.setter
    def description(self, val):
        self._description = validate_property_class(
            val, "description", Description, self
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
        assert val is None or (isinstance(val, (float, int)) and 0 <= val <= 1), (
            "opacity must be a value betwen 0 and 1\n"
            f"but received {repr(val)} instead"
        )
        self._opacity = val

    @property
    def path(self):
        """an instance of `Path` or dictionary of equivalent key/value pairs, defining the
        object path marker and path line properties"""
        return self._path

    @path.setter
    def path(self, val):
        self._path = validate_property_class(val, "path", Path, self)

    @property
    def model3d(self):
        """3d object representation properties"""
        return self._model3d

    @model3d.setter
    def model3d(self, val):
        self._model3d = validate_property_class(val, "model3d", Model3d, self)


class Description(MagicProperties):
    """
    Defines properties for a description object

    Parameters
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
        """description text"""
        return self._text

    @text.setter
    def text(self, val):
        assert val is None or isinstance(val, str), (
            f"the `show` property of {type(self).__name__} must be a string\n"
            f"but received {repr(val)} instead"
        )
        self._text = val

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


class Model3d(MagicProperties):
    """
    Defines properties for the 3d model representation of the magpylib objects

    Parameters
    ----------
    showdefault: bool, default=True
        Shows/hides default 3D-model

    data: dict or list of dicts, default=None
        a trace or list of traces where each is an instance of `Trace3d` or dictionary of equivalent
        key/value pairs. Defines properties for an additional user-defined model3d object which is
        positioned relatively to the main object to be displayed and moved automatically with it.
        This feature also allows the user to replace the original 3d representation of the object.
    """

    def __init__(self, showdefault=True, data=None, **kwargs):
        super().__init__(showdefault=showdefault, data=data, **kwargs)

    @property
    def showdefault(self):
        """shows/hides main model3d object representation"""
        return self._showdefault

    @showdefault.setter
    def showdefault(self, val):
        assert isinstance(val, bool), (
            f"the `showdefault` property of {type(self).__name__} must be "
            f"one of `[True, False]`"
            f" but received {repr(val)} instead"
        )
        self._showdefault = val

    @property
    def data(self):
        """data 3d object representation (trace or list of traces)"""
        return self._data

    @data.setter
    def data(self, val):
        self._data = self._validate_data(val)

    def _validate_data(self, val):
        if val is None:
            val = []
        elif not isinstance(val, (list, tuple)):
            val = [val]
        m3 = []
        for v in val:
            v = validate_property_class(v, "data", Trace3d, self)
            m3.append(v)
        return m3

    def add_trace(
        self, trace, show=True, backend="matplotlib", coordsargs=None, scale=1,
    ):
        """creates an additional user-defined 3d model object which is positioned relatively
        to the main object to be displayed and moved automatically with it. This feature also allows
        the user to replace the original 3d representation of the object

        Properties
        ----------
        show : bool, default=None
            shows/hides model3d object based on provided trace:

        trace: dict or callable, default=None
            dictionary containing the `x,y,z,i,j,k` keys/values pairs for a model3d object

        backend:
            plotting backend corresponding to the trace.
            Can be one of `['matplotlib', 'plotly']`

        coordsargs: dict
            tells magpylib the name of the coordinate arrays to be moved or rotated.
            by default: `{"x": "x", "y": "y", "z": "z"}`
            if False, object is not rotated

        scale: float, default=1
            scaling factor by which the trace vertices coordinates should be multiplied by. Be aware
            that if the object is not centered at the global CS origin, its position will
            also be scaled.
        """

        new_trace = Trace3d(
            trace=trace, show=show, scale=scale, backend=backend, coordsargs=coordsargs,
        )
        self._data += self._validate_data(new_trace)
        return self


class Trace3d(MagicProperties):
    """
    Defines properties for an additional user-defined 3d model object which is positioned relatively
    to the main object to be displayed and moved automatically with it. This feature also allows
    the user to replace the original 3d representation of the object

    Parameters
    ----------
    show : bool, default=None
        shows/hides model3d object based on provided trace:

    trace: dict or callable, default=None
        dictionary containing the `x,y,z,i,j,k` keys/values pairs for a model3d object

    scale: float, default=1
        scaling factor by which the trace vertices coordinates should be multiplied by. Be aware
        that if the object is not centered at the global CS origin, its position will
        also be scaled.

    backend:
        plotting backend corresponding to the trace.
        Can be one of `['matplotlib', 'plotly']`

    coordsargs: dict
        tells magpylib the name of the coordinate arrays to be moved or rotated.
        by default: `{"x": "x", "y": "y", "z": "z"}`
        if False, object is not rotated
    """

    def __init__(
        self,
        trace=None,
        scale=1,
        show=True,
        backend="matplotlib",
        coordsargs=None,
        **kwargs,
    ):
        super().__init__(
            trace=trace,
            scale=scale,
            show=show,
            backend=backend,
            coordsargs=coordsargs,
            **kwargs,
        )

    @property
    def show(self):
        """
        shows/hides model3d object based on provided trace
        """
        return self._show

    @show.setter
    def show(self, val):
        assert val is None or isinstance(val, bool), (
            f"the `show` property of {type(self).__name__} must be "
            f"one of `[True, False]`"
            f" but received {repr(val)} instead"
        )
        self._show = val

    @property
    def scale(self):
        """
        scaling factor by which the trace vertices coordinates should be multiplied by. Be aware
        that if the object is not centered at the global CS origin, its position will
        also be scaled.
        """
        return self._scale

    @scale.setter
    def scale(self, val):
        assert isinstance(val, (int, float)) and val > 0, (
            f"the `scale` property of {type(self).__name__} must be a strictly positive number"
            f" but received {repr(val)} instead"
        )
        self._scale = val

    @property
    def coordsargs(self):
        """tells magpylib the name of the coordinate arrays to be moved or rotated.
        by default: `{"x": "x", "y": "y", "z": "z"}`
        if False, object is not rotated"""
        return self._coordsargs

    @coordsargs.setter
    def coordsargs(self, val):
        if val is None:
            val = {"x": "x", "y": "y", "z": "z"}
        assert isinstance(val, dict) and all(key in val for key in "xyz"), (
            f"the `coordsargs` property of {type(self).__name__} must be "
            f"a dictionary with `'x', 'y', 'z'` keys"
            f" but received {repr(val)} instead"
        )
        self._coordsargs = val

    @property
    def trace(self):
        """dictionary keys/values pairs for a model3d object"""
        return self._trace

    @trace.setter
    def trace(self, val):
        if val is not None:
            assert isinstance(val, dict) or callable(val), (
                "trace must be a dictionary or a callable returning a dictionary"
                f" but received {type(val).__name__} instead"
            )
            test_val = val
            if callable(val):
                test_val = val()
            assert "type" in test_val, "explicit trace `type` must be defined"
        self._trace = val

    @property
    def backend(self):
        """plotting backend corresponding to the trace.
        Can be one of `['matplotlib', 'plotly']`"""
        return self._backend

    @backend.setter
    def backend(self, val):
        assert val is None or val in SUPPORTED_PLOTTING_BACKENDS, (
            f"the `backend` property of {type(self).__name__} must be one of"
            f"{SUPPORTED_PLOTTING_BACKENDS}"
            f" but received {repr(val)} instead"
        )
        self._backend = val


class Magnetization(MagicProperties):
    """
    Defines magnetization styling properties

    Parameters
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
        self._color = validate_property_class(val, "color", MagnetizationColor, self)


class MagnetizationColor(MagicProperties):
    """
    Defines the magnetization direction color styling properties.

    Note: This feature is only relevant for the plotly backend.

    Parameters
    ----------
    north: str, default=None
        defines the color of the magnetic north pole

    south: str, default=None
        defines the color of the magnetic south pole

    middle: str, default=None
        defines the color between the magnetic poles

    transition: float, default=None
        sets the transition smoothness between poles colors.
        - `transition=0`: discrete transition
        - `transition=1`: smoothest transition
        - can be any value in-between 0 and 1"

    mode: str, default=None
        sets the coloring mode for the magnetization.
        - `'bicolor'`: only north and south poles are shown, middle color is hidden.
        - `'tricolor'`: both pole colors and middle color are shown.
        - `'tricycle'`: both pole colors are shown and middle color is replaced by a color cycling
            through the color sequence.
    """

    _allowed_modes = ("bicolor", "tricolor", "tricycle")

    def __init__(
        self, north=None, south=None, middle=None, transition=None, mode=None, **kwargs
    ):
        super().__init__(
            north=north,
            middle=middle,
            south=south,
            transition=transition,
            mode=mode,
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
        """the color between the magnetic poles"""
        return self._middle

    @middle.setter
    def middle(self, val):
        self._middle = color_validator(val)

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

    @property
    def mode(self):
        """sets the coloring mode for the magnetization.
        - `'bicolor'`: only north and south poles are shown, middle color is hidden.
        - `'tricolor'`: both pole colors and middle color are shown.
        - `'tricycle'`: both pole colors are shown and middle color is replaced by a color cycling
            through the color sequence.
        """
        return self._mode

    @mode.setter
    def mode(self, val):
        assert val is None or val in self._allowed_modes, (
            f"the `mode` property of {type(self).__name__} must be one of"
            f"{list(self._allowed_modes)}"
            f" but received {repr(val)} instead"
        )
        self._mode = val


class MagnetProperties:
    """
    Defines the specific styling properties of objects of the `magnet` family

    Parameters
    ----------
    magnetization: dict or Magnetization, default=None
        Magnetization styling with `'show'`, `'size'`, `'color'` properties
        or a dictionary with equivalent key/value pairs
    """

    @property
    def magnetization(self):
        """
        Magnetization styling with `'show'`, `'size'`, `'color'` properties
        or a dictionary with equivalent key/value pairs
        """
        return self._magnetization

    @magnetization.setter
    def magnetization(self, val):
        self._magnetization = validate_property_class(
            val, "magnetization", Magnetization, self
        )


class Magnet(MagicProperties, MagnetProperties):
    """
    Defines the specific styling properties of objects of the `magnet` family

    Parameters
    ----------
    magnetization: dict or Magnetization, default=None

    """

    def __init__(self, magnetization=None, **kwargs):
        super().__init__(magnetization=magnetization, **kwargs)


class MagnetStyle(BaseStyle, MagnetProperties):
    """Defines the styling properties of objects of the `magnet` family with base properties

    Parameters
    ----------
    label: str, default=None
        label of the class instance, can be any string.

    description: dict or Description, default=None
        object description properties

    color: str, default=None
        a valid css color. Can also be one of `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`

    opacity: float, default=None
        object opacity between 0 and 1, where 1 is fully opaque and 0 is fully transparent.

    path: dict or Path, default=None
        an instance of `Path` or dictionary of equivalent key/value pairs, defining the object
        path marker and path line properties.

    model3d: list of Trace3d objects, default=None
        a list of traces where each is an instance of `Trace3d` or dictionary of equivalent
        key/value pairs. Defines properties for an additional user-defined model3d object which is
        positioned relatively to the main object to be displayed and moved automatically with it.
        This feature also allows the user to replace the original 3d representation of the object.

    magnetization: dict or Magnetization, default=None
        Magnetization styling with `'show'`, `'size'`, `'color'` properties
        or a dictionary with equivalent key/value pairs
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ArrowCS(MagicProperties):
    """Triple coordinate system arrow properties

    Parameters
    ----------
    x: dict or ArrowSingle, default=None
        x-direction `Arrowsingle` class or dict with equivalent key/value pairs
        (e.g. `color`, `show`)

    y: dict or ArrowSingle, default=None
        y-direction `Arrowsingle` class or dict with equivalent key/value pairs
        (e.g. `color`, `show`)

    z: dict or ArrowSingle, default=None
        z-direction `Arrowsingle` class or dict with equivalent key/value pairs
        (e.g. `color`, `show`)
    """

    def __init__(self, x=None, y=None, z=None):
        super().__init__(x=x, y=y, z=z)

    @property
    def x(self):
        """`Arrowsingle` class or dict with equivalent key/value pairs (e.g. `color`, `show`)"""
        return self._x

    @x.setter
    def x(self, val):
        self._x = validate_property_class(val, "x", ArrowSingle, self)

    @property
    def y(self):
        """`Arrowsingle` class or dict with equivalent key/value pairs (e.g. `color`, `show`)"""
        return self._y

    @y.setter
    def y(self, val):
        self._y = validate_property_class(val, "y", ArrowSingle, self)

    @property
    def z(self):
        """`Arrowsingle` class or dict with equivalent key/value pairs (e.g. `color`, `show`)"""
        return self._z

    @z.setter
    def z(self, val):
        self._z = validate_property_class(val, "z", ArrowSingle, self)


class ArrowSingle(MagicProperties):
    """Single coordinate system arrow properties

    Parameters
    ----------
    show: bool, default=True
        show/hide arrow

    color: color, default=None
        valid css color. Can also be one of `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`
    """
    def __init__(self, show=True, color=None):
        super().__init__(show=show, color=color)

    @property
    def show(self):
        """show/hide arrow"""
        return self._show

    @show.setter
    def show(self, val):
        assert val is None or isinstance(val, bool), (
            f"the `show` property of {type(self).__name__} must be either `True` or `False`"
            f" but received {repr(val)} instead"
        )
        self._show = val

    @property
    def color(self):
        """a valid css color. Can also be one of `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`"""
        return self._color

    @color.setter
    def color(self, val):
        self._color = color_validator(val, parent_name=f"{type(self).__name__}")


class SensorProperties:
    """
    Defines the specific styling properties of objects of the `sensor` family

    Parameters
    ----------
    size: float, default=None
        positive float for relative sensor to canvas size

    pixel: dict, Pixel, default=None
        `Pixel` class or dict with equivalent key/value pairs (e.g. `color`, `size`)

    arrows: dict, ArrowCS, default=None
        `ArrowCS` class or dict with equivalent key/value pairs (e.g. `color`, `size`)
    """

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
        self._pixel = validate_property_class(val, "pixel", Pixel, self)

    @property
    def arrows(self):
        """`ArrowCS` class or dict with equivalent key/value pairs (e.g. `color`, `size`)"""
        return self._arrows

    @arrows.setter
    def arrows(self, val):
        self._arrows = validate_property_class(val, "arrows", ArrowCS, self)


class Sensor(MagicProperties, SensorProperties):
    """
    Defines the specific styling properties of objects of the `sensor` family

    Parameters
    ----------
    size: float, default=None
        positive float for relative sensor to canvas size

    pixel: dict, Pixel, default=None
        `Pixel` class or dict with equivalent key/value pairs (e.g. `color`, `size`)

    arrows: dict, ArrowCS, default=None
        `ArrowCS` class or dict with equivalent key/value pairs (e.g. `color`, `size`)
    """

    def __init__(self, size=None, pixel=None, arrows=None, **kwargs):
        super().__init__(size=size, pixel=pixel, arrows=arrows, **kwargs)


class SensorStyle(BaseStyle, SensorProperties):
    """Defines the styling properties of objects of the `sensor` family with base properties

    Parameters
    ----------
    label: str, default=None
        label of the class instance, can be any string.

    description: dict or Description, default=None
        object description properties

    color: str, default=None
        a valid css color. Can also be one of `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`

    opacity: float, default=None
        object opacity between 0 and 1, where 1 is fully opaque and 0 is fully transparent.

    path: dict or Path, default=None
        an instance of `Path` or dictionary of equivalent key/value pairs, defining the object
        path marker and path line properties.

    model3d: list of Trace3d objects, default=None
        a list of traces where each is an instance of `Trace3d` or dictionary of equivalent
        key/value pairs. Defines properties for an additional user-defined model3d object which is
        positioned relatively to the main object to be displayed and moved automatically with it.
        This feature also allows the user to replace the original 3d representation of the object.

    size: float, default=None
        positive float for relative sensor to canvas size

    pixel: dict, Pixel, default=None
        `Pixel` class or dict with equivalent key/value pairs (e.g. `color`, `size`)

    arrows: dict, ArrowCS, default=None
        `ArrowCS` class or dict with equivalent key/value pairs (e.g. `color`, `size`)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Pixel(MagicProperties):
    """
    Defines the styling properties of sensor pixels

    Parameters
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


class CurrentProperties:
    """
    Defines the specific styling properties of objects of the `current` family

    Parameters
    ----------
    arrow: dict or Arrow, default=None
        Arrow class or dict with `'show'`, `'size'` properties/keys
    """

    @property
    def arrow(self):
        """Arrow class or dict with `'show'`, `'size'` properties/keys"""
        return self._arrow

    @arrow.setter
    def arrow(self, val):
        self._arrow = validate_property_class(val, "current", Arrow, self)


class Current(MagicProperties, CurrentProperties):
    """
    Defines the specific styling properties of objects of the `current` family

    Parameters
    ----------
    arrow: dict or Arrow, default=None
        Arrow class or dict with 'show', 'size' properties/keys
    """

    def __init__(self, arrow=None, **kwargs):
        super().__init__(arrow=arrow, **kwargs)


class CurrentStyle(BaseStyle, CurrentProperties):
    """Defines the styling properties of objects of the `current` family and base properties

    Parameters
    ----------
    label: str, default=None
        label of the class instance, can be any string.

    description: dict or Description, default=None
        object description properties

    color: str, default=None
        a valid css color. Can also be one of `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`

    opacity: float, default=None
        object opacity between 0 and 1, where 1 is fully opaque and 0 is fully transparent.

    path: dict or Path, default=None
        an instance of `Path` or dictionary of equivalent key/value pairs, defining the object
        path marker and path line properties.

    model3d: list of Trace3d objects, default=None
        a list of traces where each is an instance of `Trace3d` or dictionary of equivalent
        key/value pairs. Defines properties for an additional user-defined model3d object which is
        positioned relatively to the main object to be displayed and moved automatically with it.
        This feature also allows the user to replace the original 3d representation of the object.

    arrow: dict or Arrow, default=None
        Arrow class or dict with `'show'`, `'size'` properties/keys
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Arrow(MagicProperties):
    """
    Defines the styling properties of current arrows

    Parameters
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


class Marker(MagicProperties):
    """
    Defines the styling properties of plot markers

    Parameters
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

    Parameters
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
        self._marker = validate_property_class(val, "marker", Marker, self)


class DipoleProperties:
    """
    Defines the specific styling properties of the objects of the `dipole` family

    Parameters
    ----------
    size: float, default=None
        positive float for relative dipole to size to canvas size

    pivot: str, default=None
        the part of the arrow that is anchored to the X, Y grid.
        The arrow rotates about this point. Can be one of `['tail', 'middle', 'tip']`
    """

    _allowed_pivots = ("tail", "middle", "tip")

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


class Dipole(MagicProperties, DipoleProperties):
    """
    Defines the specific styling properties of the objects of the `dipole` family

    Parameters
    ----------
    size: float, default=None
        positive float for relative dipole to size to canvas size

    pivot: str, default=None
        the part of the arrow that is anchored to the X, Y grid.
        The arrow rotates about this point. Can be one of `['tail', 'middle', 'tip']`
    """

    def __init__(self, size=None, pivot=None, **kwargs):
        super().__init__(size=size, pivot=pivot, **kwargs)


class DipoleStyle(BaseStyle, DipoleProperties):
    """Defines the styling properties of the objects of the `dipole` family and base properties

    Parameters
    ----------
    label: str, default=None
        label of the class instance, can be any string.

    description: dict or Description, default=None
        object description properties

    color: str, default=None
        a valid css color. Can also be one of `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`

    opacity: float, default=None
        object opacity between 0 and 1, where 1 is fully opaque and 0 is fully transparent.

    path: dict or Path, default=None
        an instance of `Path` or dictionary of equivalent key/value pairs, defining the object
        path marker and path line properties.

    model3d: list of Trace3d objects, default=None
        a list of traces where each is an instance of `Trace3d` or dictionary of equivalent
        key/value pairs. Defines properties for an additional user-defined model3d object which is
        positioned relatively to the main object to be displayed and moved automatically with it.
        This feature also allows the user to replace the original 3d representation of the object.

    size: float, default=None
        positive float for relative dipole to size to canvas size

    pivot: str, default=None
        the part of the arrow that is anchored to the X, Y grid.
        The arrow rotates about this point. Can be one of `['tail', 'middle', 'tip']`
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Path(MagicProperties):
    """
    Defines the styling properties of an object's path

    Parameters
    ----------
    marker: dict, Markers, default=None
        Markers class with 'color', 'symbol', 'size' properties, or dictionary with equivalent
        key/value pairs

    line: dict, Line, default=None
        Line class with 'color', 'symbol', 'size' properties, or dictionary with equivalent
        key/value pairs

    show: bool, default=None
        show/hide path
        - False: shows object(s) at final path position and hides paths lines and markers.
        - True: shows object(s) shows object paths depending on `line`, `marker` and `frames`
        parameters.

    frames: int or 1D-array_like
        show copies of the 3D-model along the given path indices.
        - integer i: displays the object(s) at every i'th path position.
        - array_like shape (n,) of integers: describes certain path indices.

    numbering: bool, default=False
        show/hide numbering on path positions. Only applies if show=True.
    """

    def __init__(
        self, marker=None, line=None, frames=None, show=None, numbering=None, **kwargs
    ):
        super().__init__(
            marker=marker,
            line=line,
            frames=frames,
            show=show,
            numbering=numbering,
            **kwargs,
        )

    @property
    def marker(self):
        """Markers class with 'color', 'symbol', 'size' properties"""
        return self._marker

    @marker.setter
    def marker(self, val):
        self._marker = validate_property_class(val, "marker", Marker, self)

    @property
    def line(self):
        """Line class with 'color', 'type', 'width' properties"""
        return self._line

    @line.setter
    def line(self, val):
        self._line = validate_property_class(val, "line", Line, self)

    @property
    def show(self):
        """show/hide path
        - False: shows object(s) at final path position and hides paths lines and markers.
        - True: shows object(s) shows object paths depending on `line`, `marker` and `frames`
        parameters.
        """
        return self._show

    @show.setter
    def show(self, val):
        assert val is None or isinstance(val, bool), (
            f"the `show` property of {type(self).__name__} must be either "
            "`True` or `False`"
            f"\nbut received {repr(val)} instead"
        )
        self._show = val

    @property
    def frames(self):
        """show copies of the 3D-model along the given path indices.
        - integer i: displays the object(s) at every i'th path position.
        - array_like shape (n,) of integers: describes certain path indices.
        """
        return self._frames

    @frames.setter
    def frames(self, val):
        is_valid_path = True
        if hasattr(val, "__iter__") and not isinstance(val, str):
            val = tuple(val)
            if not all(np.issubdtype(type(v), int) for v in val):
                is_valid_path = False
        elif not (val is None or np.issubdtype(type(val), int)):
            is_valid_path = False
        assert is_valid_path, f"""the `frames` property of {type(self).__name__} must be either:
- integer i: displays the object(s) at every i'th path position.
- array_like shape (n,) of integers: describes certain path indices.
but received {repr(val)} instead"""
        self._frames = val

    @property
    def numbering(self):
        """show/hide numbering on path positions. Only applies if show=True."""
        return self._numbering

    @numbering.setter
    def numbering(self, val):
        assert val is None or isinstance(val, bool), (
            f"the `numbering` property of {type(self).__name__} must be one of (True, False),"
            f" but received {repr(val)} instead"
        )
        self._numbering = val


class Line(MagicProperties):
    """
    Defines Line styling properties

    Parameters
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


class DisplayStyle(MagicProperties):
    """
    Base class containing styling properties for all object families. The properties of the
    sub-classes get set to hard coded defaults at class instantiation

    Parameters
    ----------
    base: dict, Base, default=None
        base properties common to all families

    magnet: dict, Magnet, default=None
        magnet properties

    current: dict, Current, default=None
        current properties

    dipole: dict, Dipole, default=None
        dipole properties

    sensor: dict, Sensor, default=None
        sensor properties

    markers: dict, Markers, default=None
        markers properties
    """

    def __init__(
        self,
        base=None,
        magnet=None,
        current=None,
        dipole=None,
        sensor=None,
        markers=None,
        **kwargs,
    ):
        super().__init__(
            base=base,
            magnet=magnet,
            current=current,
            dipole=dipole,
            sensor=sensor,
            markers=markers,
            **kwargs,
        )
        # self.reset()

    def reset(self):
        """Resets all nested properties to their hard coded default values"""
        self.update(get_defaults_dict("display.style"), _match_properties=False)
        return self

    @property
    def base(self):
        """base properties common to all families"""
        return self._base

    @base.setter
    def base(self, val):
        self._base = validate_property_class(val, "base", BaseStyle, self)

    @property
    def magnet(self):
        """Magnet class"""
        return self._magnet

    @magnet.setter
    def magnet(self, val):
        self._magnet = validate_property_class(val, "magnet", Magnet, self)

    @property
    def current(self):
        """Current class"""
        return self._current

    @current.setter
    def current(self, val):
        self._current = validate_property_class(val, "current", Current, self)

    @property
    def dipole(self):
        """Dipole class"""
        return self._dipole

    @dipole.setter
    def dipole(self, val):
        self._dipole = validate_property_class(val, "dipole", Dipole, self)

    @property
    def sensor(self):
        """Sensor"""
        return self._sensor

    @sensor.setter
    def sensor(self, val):
        self._sensor = validate_property_class(val, "sensor", Sensor, self)

    @property
    def markers(self):
        """Markers class"""
        return self._markers

    @markers.setter
    def markers(self, val):
        self._markers = validate_property_class(val, "markers", Markers, self)


STYLE_CLASSES = {
    "magnet": MagnetStyle,
    "current": CurrentStyle,
    "dipole": DipoleStyle,
    "sensor": SensorStyle,
}

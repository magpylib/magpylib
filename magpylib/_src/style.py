"""Collection of classes for display styling."""
# pylint: disable=C0302
# pylint: disable=too-many-instance-attributes
import numpy as np

from magpylib._src.defaults.defaults_utility import color_validator
from magpylib._src.defaults.defaults_utility import get_defaults_dict
from magpylib._src.defaults.defaults_utility import LINESTYLES_MATPLOTLIB_TO_PLOTLY
from magpylib._src.defaults.defaults_utility import MagicProperties
from magpylib._src.defaults.defaults_utility import MAGPYLIB_FAMILIES
from magpylib._src.defaults.defaults_utility import SUPPORTED_PLOTTING_BACKENDS
from magpylib._src.defaults.defaults_utility import SYMBOLS_MATPLOTLIB_TO_PLOTLY
from magpylib._src.defaults.defaults_utility import validate_property_class
from magpylib._src.defaults.defaults_utility import validate_style_keys


def get_style_class(obj):
    """Returns style instance based on object type. If object has no attribute `_object_type` or is
    not found in `MAGPYLIB_FAMILIES` returns `BaseStyle` instance.
    """
    obj_type = getattr(obj, "_object_type", None)
    style_fam = MAGPYLIB_FAMILIES.get(obj_type, None)
    if isinstance(style_fam, (list, tuple)):
        style_fam = style_fam[0]
    return STYLE_CLASSES.get(style_fam, BaseStyle)


def get_style(obj, default_settings, **kwargs):
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
    """Base class for display styling options of `BaseGeo` objects.

    Parameters
    ----------
    label: str, default=None
        Label of the class instance, e.g. to be displayed in the legend.

    description: dict or `Description` object, default=None
        Object description properties.

    color: str, default=None
        A valid css color. Can also be one of `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`.

    opacity: float, default=None
        object opacity between 0 and 1, where 1 is fully opaque and 0 is fully transparent.

    path: dict or `Path` object, default=None
        An instance of `Path` or dictionary of equivalent key/value pairs, defining the object
        path marker and path line properties.

    model3d: list of `Trace3d` objects, default=None
        A list of traces where each is an instance of `Trace3d` or dictionary of equivalent
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
        """Label of the class instance, e.g. to be displayed in the legend."""
        return self._label

    @label.setter
    def label(self, val):
        self._label = val if val is None else str(val)

    @property
    def description(self):
        """Description with 'text' and 'show' properties."""
        return self._description

    @description.setter
    def description(self, val):
        self._description = validate_property_class(
            val, "description", Description, self
        )

    @property
    def color(self):
        """A valid css color. Can also be one of `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`."""
        return self._color

    @color.setter
    def color(self, val):
        self._color = color_validator(val, parent_name=f"{type(self).__name__}")

    @property
    def opacity(self):
        """Object opacity between 0 and 1, where 1 is fully opaque and 0 is fully transparent."""
        return self._opacity

    @opacity.setter
    def opacity(self, val):
        assert val is None or (isinstance(val, (float, int)) and 0 <= val <= 1), (
            "The `opacity` property must be a value betwen 0 and 1,\n"
            f"but received {repr(val)} instead."
        )
        self._opacity = val

    @property
    def path(self):
        """An instance of `Path` or dictionary of equivalent key/value pairs, defining the
        object path marker and path line properties."""
        return self._path

    @path.setter
    def path(self, val):
        self._path = validate_property_class(val, "path", Path, self)

    @property
    def model3d(self):
        """3d object representation properties."""
        return self._model3d

    @model3d.setter
    def model3d(self, val):
        self._model3d = validate_property_class(val, "model3d", Model3d, self)


class Description(MagicProperties):
    """Defines properties for a description object.

    Parameters
    ----------
    text: str, default=None
        Object description text.

    show: bool, default=None
        If True, adds legend entry suffix based on value.
    """

    def __init__(self, text=None, show=None, **kwargs):
        super().__init__(text=text, show=show, **kwargs)

    @property
    def text(self):
        """Description text."""
        return self._text

    @text.setter
    def text(self, val):
        assert val is None or isinstance(val, str), (
            f"The `show` property of {type(self).__name__} must be a string,\n"
            f"but received {repr(val)} instead."
        )
        self._text = val

    @property
    def show(self):
        """If True, adds legend entry suffix based on value."""
        return self._show

    @show.setter
    def show(self, val):
        assert val is None or isinstance(val, bool), (
            f"The `show` property of {type(self).__name__} must be either True or False,\n"
            f"but received {repr(val)} instead."
        )
        self._show = val


class Model3d(MagicProperties):
    """Defines properties for the 3d model representation of magpylib objects.

    Parameters
    ----------
    showdefault: bool, default=True
        Shows/hides default 3d-model.

    data: dict or list of dicts, default=None
        A trace or list of traces where each is an instance of `Trace3d` or dictionary of equivalent
        key/value pairs. Defines properties for an additional user-defined model3d object which is
        positioned relatively to the main object to be displayed and moved automatically with it.
        This feature also allows the user to replace the original 3d representation of the object.
    """

    def __init__(self, showdefault=True, data=None, **kwargs):
        super().__init__(showdefault=showdefault, data=data, **kwargs)

    @property
    def showdefault(self):
        """If True, show default model3d object representation, else hide it."""
        return self._showdefault

    @showdefault.setter
    def showdefault(self, val):
        assert isinstance(val, bool), (
            f"The `showdefault` property of {type(self).__name__} must be "
            f"one of `[True, False]`,\n"
            f"but received {repr(val)} instead."
        )
        self._showdefault = val

    @property
    def data(self):
        """Data of 3d object representation (trace or list of traces)."""
        return self._data

    @data.setter
    def data(self, val):
        self._data = self._validate_data(val)

    def _validate_data(self, traces, **kwargs):
        if traces is None:
            traces = []
        elif not isinstance(traces, (list, tuple)):
            traces = [traces]
        new_traces = []
        for trace in traces:
            updatefunc = None
            if not isinstance(trace, Trace3d) and callable(trace):
                updatefunc = trace
                trace = Trace3d()
            if not isinstance(trace, Trace3d):
                trace = validate_property_class(trace, "data", Trace3d, self)
            if updatefunc is not None:
                trace.updatefunc = updatefunc
            trace = trace.update(kwargs)
            new_traces.append(trace)
        return new_traces

    def add_trace(self, trace=None, **kwargs):
        """Adds user-defined 3d model object which is positioned relatively to the main object to be
        displayed and moved automatically with it. This feature also allows the user to replace the
        original 3d representation of the object.

        trace: Trace3d instance, dict or callable
            Trace object. Can be a `Trace3d` instance or an dictionary with equivalent key/values
            pairs, or a callable returning the equivalent dictionary.

        backend: str
            Plotting backend corresponding to the trace. Can be one of `['matplotlib', 'plotly']`.

        constructor: str
            Model constructor function or method to be called to build a 3D-model object
            (e.g. 'plot_trisurf', 'Mesh3d). Must be in accordance with the given plotting backend.

        args: tuple, default=None
            Tuple or callable returning a tuple containing positional arguments for building a
            3D-model object.

        kwargs: dict or callable, default=None
            Dictionary or callable returning a dictionary containing the keys/values pairs for
            building a 3D-model object.

        coordsargs: dict, default=None
            Tells magpylib the name of the coordinate arrays to be moved or rotated,
                by default: `{"x": "x", "y": "y", "z": "z"}`, if False, object is not rotated.

        show: bool, default=None
            Shows/hides model3d object based on provided trace.

        scale: float, default=1
            Scaling factor by which the trace vertices coordinates are multiplied.

        updatefunc: callable, default=None
            Callable object with no arguments. Should return a dictionary with keys from the
            trace parameters. If provided, the function is called at `show` time and updates the
            trace parameters with the output dictionary. This allows to update a trace dynamically
            depending on class attributes, and postpone the trace construction to when the object is
            displayed.
        """
        self._data += self._validate_data([trace], **kwargs)
        return self


class Trace3d(MagicProperties):
    """Defines properties for an additional user-defined 3d model object which is positioned
    relatively to the main object to be displayed and moved automatically with it. This feature
    also allows the user to replace the original 3d representation of the object.

    Parameters
    ----------
    backend: str
        Plotting backend corresponding to the trace. Can be one of `['matplotlib', 'plotly']`.

    constructor: str
        Model constructor function or method to be called to build a 3D-model object
        (e.g. 'plot_trisurf', 'Mesh3d). Must be in accordance with the given plotting backend.

    args: tuple, default=None
        Tuple or callable returning a tuple containing positional arguments for building a
        3D-model object.

    kwargs: dict or callable, default=None
        Dictionary or callable returning a dictionary containing the keys/values pairs for
        building a 3D-model object.

    coordsargs: dict, default=None
        Tells magpylib the name of the coordinate arrays to be moved or rotated,
            by default: `{"x": "x", "y": "y", "z": "z"}`, if False, object is not rotated.

    show: bool, default=True
        Shows/hides model3d object based on provided trace.

    scale: float, default=1
        Scaling factor by which the trace vertices coordinates are multiplied.

    updatefunc: callable, default=None
        Callable object with no arguments. Should return a dictionary with keys from the
        trace parameters. If provided, the function is called at `show` time and updates the
        trace parameters with the output dictionary. This allows to update a trace dynamically
        depending on class attributes, and postpone the trace construction to when the object is
        displayed.
    """

    def __init__(
        self,
        backend=None,
        constructor=None,
        args=None,
        kwargs=None,
        coordsargs=None,
        show=True,
        scale=1,
        updatefunc=None,
        **params,
    ):
        super().__init__(
            backend=backend,
            constructor=constructor,
            args=args,
            kwargs=kwargs,
            coordsargs=coordsargs,
            show=show,
            scale=scale,
            updatefunc=updatefunc,
            **params,
        )

    @property
    def args(self):
        """Tuple or callable returning a tuple containing positional arguments for building a
        3D-model object."""
        return self._args

    @args.setter
    def args(self, val):
        if val is not None:
            test_val = val
            if callable(val):
                test_val = val()
            assert isinstance(test_val, tuple), (
                "The `trace` input must be a dictionary or a callable returning a dictionary,\n"
                f"but received {type(val).__name__} instead."
            )
        self._args = val

    @property
    def kwargs(self):
        """Dictionary or callable returning a dictionary containing the keys/values pairs for
        building a 3D-model object."""
        return self._kwargs

    @kwargs.setter
    def kwargs(self, val):
        if val is not None:
            test_val = val
            if callable(val):
                test_val = val()
            assert isinstance(test_val, dict), (
                "The `kwargs` input must be a dictionary or a callable returning a dictionary,\n"
                f"but received {type(val).__name__} instead."
            )
        self._kwargs = val

    @property
    def constructor(self):
        """Model constructor function or method to be called to build a 3D-model object
        (e.g. 'plot_trisurf', 'Mesh3d). Must be in accordance with the given plotting backend."""
        return self._constructor

    @constructor.setter
    def constructor(self, val):
        assert val is None or isinstance(val, str), (
            f"The `constructor` property of {type(self).__name__} must be a string,"
            f"\nbut received {repr(val)} instead."
        )
        self._constructor = val

    @property
    def show(self):
        """If True, show default model3d object representation, else hide it."""
        return self._show

    @show.setter
    def show(self, val):
        assert isinstance(val, bool), (
            f"The `show` property of {type(self).__name__} must be "
            f"one of `[True, False]`,\n"
            f"but received {repr(val)} instead."
        )
        self._show = val

    @property
    def scale(self):
        """Scaling factor by which the trace vertices coordinates are multiplied."""
        return self._scale

    @scale.setter
    def scale(self, val):
        assert isinstance(val, (int, float)) and val > 0, (
            f"The `scale` property of {type(self).__name__} must be a strictly positive number,\n"
            f"but received {repr(val)} instead."
        )
        self._scale = val

    @property
    def coordsargs(self):
        """Tells magpylib the name of the coordinate arrays to be moved or rotated,
        by default: `{"x": "x", "y": "y", "z": "z"}`, if False, object is not rotated.
        """
        return self._coordsargs

    @coordsargs.setter
    def coordsargs(self, val):
        assert val is None or (
            isinstance(val, dict) and all(key in val for key in "xyz")
        ), (
            f"The `coordsargs` property of {type(self).__name__} must be "
            f"a dictionary with `'x', 'y', 'z'` keys,\n"
            f"but received {repr(val)} instead."
        )
        self._coordsargs = val

    @property
    def backend(self):
        """Plotting backend corresponding to the trace. Can be one of `['matplotlib', 'plotly']`."""
        return self._backend

    @backend.setter
    def backend(self, val):
        assert val is None or val in SUPPORTED_PLOTTING_BACKENDS, (
            f"The `backend` property of {type(self).__name__} must be one of"
            f"{SUPPORTED_PLOTTING_BACKENDS},\n"
            f"but received {repr(val)} instead."
        )
        self._backend = val

    @property
    def updatefunc(self):
        """Callable object with no arguments. Should return a dictionary with keys from the
        trace parameters. If provided, the function is called at `show` time and updates the
        trace parameters with the output dictionary. This allows to update a trace dynamically
        depending on class attributes, and postpone the trace construction to when the object is
        displayed."""
        return self._updatefunc

    @updatefunc.setter
    def updatefunc(self, val):
        if val is None:
            val = lambda: {}
        msg = ""
        valid_props = list(self._property_names_generator())
        if not callable(val):
            msg = f"Instead received {type(val)}"
        else:
            test_val = val()
            if not isinstance(test_val, dict):
                msg = f"but callable returned type {type(test_val)}."
            else:
                bad_keys = [k for k in test_val.keys() if k not in valid_props]
                if bad_keys:
                    msg = f"but invalid output dictionary keys received: {bad_keys}."

        assert msg == "", (
            f"The `updatefunc` property of {type(self).__name__} must be a callable returning a "
            f"dictionary with a subset of following keys: {valid_props} keys.\n"
            f"{msg}"
        )
        self._updatefunc = val


class Magnetization(MagicProperties):
    """Defines magnetization styling properties.

    Parameters
    ----------
    show : bool, default=None
        If True show magnetization direction.

    size: float, default=None
        Arrow size of the magnetization direction (for the matplotlib backend)
        only applies if `show=True`.

    color: dict or MagnetizationColor object, default=None
        Color properties showing the magnetization direction (for the plotly backend).
        Only applies if `show=True`.
    """

    def __init__(self, show=None, size=None, color=None, **kwargs):
        super().__init__(show=show, size=size, color=color, **kwargs)

    @property
    def show(self):
        """If True, show magnetization direction."""
        return self._show

    @show.setter
    def show(self, val):
        assert val is None or isinstance(val, bool), (
            "The `show` input must be either True or False,\n"
            f"but received {repr(val)} instead."
        )
        self._show = val

    @property
    def size(self):
        """Positive float for ratio of arrow size to magnet size."""
        return self._size

    @size.setter
    def size(self, val):
        assert val is None or isinstance(val, (int, float)) and val >= 0, (
            f"The `size` property of {type(self).__name__} must be a positive number,\n"
            f"but received {repr(val)} instead."
        )
        self._size = val

    @property
    def color(self):
        """Color properties showing the magnetization direction (for the plotly backend).
        Applies only if `show=True`.
        """
        return self._color

    @color.setter
    def color(self, val):
        self._color = validate_property_class(val, "color", MagnetizationColor, self)


class MagnetizationColor(MagicProperties):
    """Defines the magnetization direction color styling properties. (Only relevant for
    the plotly backend)

    Parameters
    ----------
    north: str, default=None
        Defines the color of the magnetic north pole.

    south: str, default=None
        Defines the color of the magnetic south pole.

    middle: str, default=None
        Defines the color between the magnetic poles.

    transition: float, default=None
        Sets the transition smoothness between poles colors. Can be any value
        in-between 0 (discrete) and 1(smooth).

    mode: str, default=None
        Sets the coloring mode for the magnetization.
        - `'bicolor'`: Only north and south pole colors are shown.
        - `'tricolor'`: Both pole colors and middle color are shown.
        - `'tricycle'`: Both pole colors are shown and middle color is replaced by a color cycling
            through the default color sequence.
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
        """Color of the magnetic north pole."""
        return self._north

    @north.setter
    def north(self, val):
        self._north = color_validator(val)

    @property
    def south(self):
        """Color of the magnetic south pole."""
        return self._south

    @south.setter
    def south(self, val):
        self._south = color_validator(val)

    @property
    def middle(self):
        """Color between the magnetic poles."""
        return self._middle

    @middle.setter
    def middle(self, val):
        self._middle = color_validator(val)

    @property
    def transition(self):
        """Sets the transition smoothness between poles colors. Can be any value
        in-between 0 (discrete) and 1(smooth).
        """
        return self._transition

    @transition.setter
    def transition(self, val):
        assert (
            val is None or isinstance(val, (float, int)) and 0 <= val <= 1
        ), "color transition must be a value betwen 0 and 1"
        self._transition = val

    @property
    def mode(self):
        """Sets the coloring mode for the magnetization.
        - `'bicolor'`: Only north and south pole colors are shown.
        - `'tricolor'`: Both pole colors and middle color are shown.
        - `'tricycle'`: Both pole colors are shown and middle color is replaced by a color cycling
            through the default color sequence.
        """
        return self._mode

    @mode.setter
    def mode(self, val):
        assert val is None or val in self._allowed_modes, (
            f"The `mode` property of {type(self).__name__} must be one of"
            f"{list(self._allowed_modes)},\n"
            f"but received {repr(val)} instead."
        )
        self._mode = val


class MagnetProperties:
    """Defines styling properties of homogeneous magnet classes.

    Parameters
    ----------
    magnetization: dict or `Magnetization` object, default=None
        `Magnetization` instance with `'show'`, `'size'`, `'color'` properties
        or a dictionary with equivalent key/value pairs.
    """

    @property
    def magnetization(self):
        """`Magnetization` instance with `'show'`, `'size'`, `'color'` properties
        or a dictionary with equivalent key/value pairs.
        """
        return self._magnetization

    @magnetization.setter
    def magnetization(self, val):
        self._magnetization = validate_property_class(
            val, "magnetization", Magnetization, self
        )


class Magnet(MagicProperties, MagnetProperties):
    """Defines styling properties of homogeneous magnet classes.

    Parameters
    ----------
    magnetization: dict or Magnetization, default=None
    """

    def __init__(self, magnetization=None, **kwargs):
        super().__init__(magnetization=magnetization, **kwargs)


class MagnetStyle(BaseStyle, MagnetProperties):
    """Defines styling properties of homogeneous magnet classes.

    Parameters
    ----------
    label: str, default=None
        Label of the class instance, e.g. to be displayed in the legend.

    description: dict or `Description` object, default=None
        Object description properties.

    color: str, default=None
        A valid css color. Can also be one of `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`.

    opacity: float, default=None
        Object opacity between 0 and 1, where 1 is fully opaque and 0 is fully transparent.

    path: dict or `Path` object, default=None
        An instance of `Path` or dictionary of equivalent key/value pairs, defining the object
        path marker and path line properties.

    model3d: list of `Trace3d` objects, default=None
        A list of traces where each is an instance of `Trace3d` or dictionary of equivalent
        key/value pairs. Defines properties for an additional user-defined model3d object which is
        positioned relatively to the main object to be displayed and moved automatically with it.
        This feature also allows the user to replace the original 3d representation of the object.

    magnetization: dict or Magnetization, default=None
        Magnetization styling with `'show'`, `'size'`, `'color'` properties
        or a dictionary with equivalent key/value pairs.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ArrowCS(MagicProperties):
    """Defines triple coordinate system arrow properties.

    Parameters
    ----------
    x: dict or `ArrowSingle` object, default=None
        x-direction `Arrowsingle` object or dict with equivalent key/value pairs
        (e.g. `color`, `show`).

    y: dict or `ArrowSingle` object, default=None
        y-direction `Arrowsingle` object or dict with equivalent key/value pairs
        (e.g. `color`, `show`).

    z: dict or `ArrowSingle` object, default=None
        z-direction `Arrowsingle` object or dict with equivalent key/value pairs
        (e.g. `color`, `show`).
    """

    def __init__(self, x=None, y=None, z=None):
        super().__init__(x=x, y=y, z=z)

    @property
    def x(self):
        """
        `ArrowSingle` object or dict with equivalent key/value pairs (e.g. `color`, `show`).
        """
        return self._x

    @x.setter
    def x(self, val):
        self._x = validate_property_class(val, "x", ArrowSingle, self)

    @property
    def y(self):
        """
        `ArrowSingle` object or dict with equivalent key/value pairs (e.g. `color`, `show`).
        """
        return self._y

    @y.setter
    def y(self, val):
        self._y = validate_property_class(val, "y", ArrowSingle, self)

    @property
    def z(self):
        """
        `ArrowSingle` object or dict with equivalent key/value pairs (e.g. `color`, `show`).
        """
        return self._z

    @z.setter
    def z(self, val):
        self._z = validate_property_class(val, "z", ArrowSingle, self)


class ArrowSingle(MagicProperties):
    """Single coordinate system arrow properties.

    Parameters
    ----------
    show: bool, default=True
        Show/hide arrow.

    color: color, default=None
        Valid css color. Can also be one of `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`.
    """

    def __init__(self, show=True, color=None):
        super().__init__(show=show, color=color)

    @property
    def show(self):
        """Show/hide arrow."""
        return self._show

    @show.setter
    def show(self, val):
        assert val is None or isinstance(val, bool), (
            f"The `show` property of {type(self).__name__} must be either True or False,\n"
            f"but received {repr(val)} instead."
        )
        self._show = val

    @property
    def color(self):
        """A valid css color. Can also be one of `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`."""
        return self._color

    @color.setter
    def color(self, val):
        self._color = color_validator(val, parent_name=f"{type(self).__name__}")


class SensorProperties:
    """Defines the specific styling properties of the Sensor class.

    Parameters
    ----------
    size: float, default=None
        Positive float for ratio of sensor to canvas size.

    pixel: dict, Pixel, default=None
        `Pixel` object or dict with equivalent key/value pairs (e.g. `color`, `size`).

    arrows: dict, ArrowCS, default=None
        `ArrowCS` object or dict with equivalent key/value pairs (e.g. `color`, `size`).
    """

    @property
    def size(self):
        """Positive float for ratio of sensor to canvas size."""
        return self._size

    @size.setter
    def size(self, val):
        assert val is None or isinstance(val, (int, float)) and val >= 0, (
            f"The `size` property of {type(self).__name__} must be a positive number,\n"
            f"but received {repr(val)} instead."
        )
        self._size = val

    @property
    def pixel(self):
        """`Pixel` object or dict with equivalent key/value pairs (e.g. `color`, `size`)."""
        return self._pixel

    @pixel.setter
    def pixel(self, val):
        self._pixel = validate_property_class(val, "pixel", Pixel, self)

    @property
    def arrows(self):
        """`ArrowCS` object or dict with equivalent key/value pairs (e.g. `color`, `size`)."""
        return self._arrows

    @arrows.setter
    def arrows(self, val):
        self._arrows = validate_property_class(val, "arrows", ArrowCS, self)


class Sensor(MagicProperties, SensorProperties):
    """Defines styling properties of the Sensor class.

    Parameters
    ----------
    size: float, default=None
        Positive float for ratio of sensor to canvas size.

    pixel: dict, Pixel, default=None
        `Pixel` object or dict with equivalent key/value pairs (e.g. `color`, `size`).

    arrows: dict, ArrowCS, default=None
        `ArrowCS` object or dict with equivalent key/value pairs (e.g. `color`, `size`).
    """

    def __init__(self, size=None, pixel=None, arrows=None, **kwargs):
        super().__init__(size=size, pixel=pixel, arrows=arrows, **kwargs)


class SensorStyle(BaseStyle, SensorProperties):
    """Defines the styling properties of the Sensor class.

    Parameters
    ----------
    label: str, default=None
        Label of the class instance, e.g. to be displayed in the legend.

    description: dict or `Description` object, default=None
        Object description properties.

    color: str, default=None
        A valid css color. Can also be one of `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`.

    opacity: float, default=None
        Object opacity between 0 and 1, where 1 is fully opaque and 0 is fully transparent.

    path: dict or `Path` object, default=None
        An instance of `Path` or dictionary of equivalent key/value pairs, defining the object
        path marker and path line properties.

    model3d: list of `Trace3d` objects, default=None
        A list of traces where each is an instance of `Trace3d` or dictionary of equivalent
        key/value pairs. Defines properties for an additional user-defined model3d object which is
        positioned relatively to the main object to be displayed and moved automatically with it.
        This feature also allows the user to replace the original 3d representation of the object.

    size: float, default=None
        Positive float for ratio of sensor size to canvas size.

    pixel: dict, Pixel, default=None
        `Pixel` object or dict with equivalent key/value pairs (e.g. `color`, `size`).

    arrows: dict, ArrowCS, default=None
        `ArrowCS` object or dict with equivalent key/value pairs (e.g. `color`, `size`).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Pixel(MagicProperties):
    """Defines the styling properties of sensor pixels.

    Parameters
    ----------
    size: float, default=1
        Positive float for relative pixel size.
        - matplotlib backend: Pixel size is the marker size.
        - plotly backend: Relative distance to nearest neighbor pixel.

    color: str, default=None
        Defines the pixel color@property.

    symbol: str, default=None
        Pixel symbol. Can be one of `['.', 'o', '+', 'D', 'd', 's', 'x']`.
        Only applies for matplotlib plotting backend.
    """

    def __init__(self, size=1, color=None, symbol=None, **kwargs):
        super().__init__(size=size, color=color, symbol=symbol, **kwargs)

    @property
    def size(self):
        """Positive float for relative pixel size.
        - matplotlib backend: Pixel size is the marker size.
        - plotly backend: Relative distance to nearest neighbor pixel."""
        return self._size

    @size.setter
    def size(self, val):
        assert val is None or isinstance(val, (int, float)) and val >= 0, (
            f"the `size` property of {type(self).__name__} must be a positive number"
            f"but received {repr(val)} instead."
        )
        self._size = val

    @property
    def color(self):
        """Pixel color."""
        return self._color

    @color.setter
    def color(self, val):
        self._color = color_validator(val, parent_name=f"{type(self).__name__}")

    @property
    def symbol(self):
        """Pixel symbol. Can be one of `['.', 'o', '+', 'D', 'd', 's', 'x']`."""
        return self._symbol

    @symbol.setter
    def symbol(self, val):
        assert val is None or val in SYMBOLS_MATPLOTLIB_TO_PLOTLY, (
            f"The `symbol` property of {type(self).__name__} must be one of"
            f"{list(SYMBOLS_MATPLOTLIB_TO_PLOTLY.keys())},\n"
            f"but received {repr(val)} instead."
        )
        self._symbol = val


class CurrentProperties:
    """Defines styling properties of line current classes.

    Parameters
    ----------
    arrow: dict or `Arrow` object, default=None
        `Arrow` object or dict with `'show'`, `'size'` properties/keys.
    """

    @property
    def arrow(self):
        """`Arrow` object or dict with `'show'`, `'size'` properties/keys."""
        return self._arrow

    @arrow.setter
    def arrow(self, val):
        self._arrow = validate_property_class(val, "current", Arrow, self)


class Current(MagicProperties, CurrentProperties):
    """Defines the specific styling properties of line current classes.

    Parameters
    ----------
    arrow: dict or `Arrow`object, default=None
        `Arrow` object or dict with 'show', 'size' properties/keys.
    """

    def __init__(self, arrow=None, **kwargs):
        super().__init__(arrow=arrow, **kwargs)


class CurrentStyle(BaseStyle, CurrentProperties):
    """Defines styling properties of line current classes.

    Parameters
    ----------
    label: str, default=None
        Label of the class instance, e.g. to be displayed in the legend.

    description: dict or `Description` object, default=None
        Object description properties.

    color: str, default=None
        A valid css color. Can also be one of `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`.

    opacity: float, default=None
        Object opacity between 0 and 1, where 1 is fully opaque and 0 is fully transparent.

    path: dict or `Path` object, default=None
        An instance of `Path` or dictionary of equivalent key/value pairs, defining the object
        path marker and path line properties.

    model3d: list of `Trace3d` objects, default=None
        A list of traces where each is an instance of `Trace3d` or dictionary of equivalent
        key/value pairs. Defines properties for an additional user-defined model3d object which is
        positioned relatively to the main object to be displayed and moved automatically with it.
        This feature also allows the user to replace the original 3d representation of the object.

    arrow: dict or `Arrow` object, default=None
        `Arrow` object or dict with `'show'`, `'size'` properties/keys.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Arrow(MagicProperties):
    """Defines styling properties of current arrows.

    Parameters
    ----------
    show: bool, default=None
        If True current direction is shown with an arrow.

    size: float, default=None
        Positive number defining the size of the arrows.

    width: float, default=None
        Positive number that defines the arrow line width.
    """

    def __init__(self, show=None, size=None, **kwargs):
        super().__init__(show=show, size=size, **kwargs)

    @property
    def show(self):
        """Show/hide arrow showing current direction."""
        return self._show

    @show.setter
    def show(self, val):
        assert val is None or isinstance(val, bool), (
            f"The `show` property of {type(self).__name__} must be either True or False,"
            f"but received {repr(val)} instead."
        )
        self._show = val

    @property
    def size(self):
        """Positive number defining the size of the arrows."""
        return self._size

    @size.setter
    def size(self, val):
        assert val is None or isinstance(val, (int, float)) and val >= 0, (
            f"The `size` property of {type(self).__name__} must be a positive number,\n"
            f"but received {repr(val)} instead."
        )
        self._size = val

    @property
    def width(self):
        """Positive number that defines the arrow line width."""
        return self._width

    @width.setter
    def width(self, val):
        assert val is None or isinstance(val, (int, float)) and val >= 0, (
            f"The `width` property of {type(self).__name__} must be a positive number,\n"
            f"but received {repr(val)} instead."
        )
        self._width = val


class Marker(MagicProperties):
    """Defines styling properties of plot markers.

    Parameters
    ----------
    size: float, default=None
        Marker size.
    color: str, default=None
        Marker color.
    symbol: str, default=None
        Marker symbol. Can be one of `['.', 'o', '+', 'D', 'd', 's', 'x']`.
    """

    def __init__(self, size=None, color=None, symbol=None, **kwargs):
        super().__init__(size=size, color=color, symbol=symbol, **kwargs)

    @property
    def size(self):
        """Marker size."""
        return self._size

    @size.setter
    def size(self, val):
        assert val is None or isinstance(val, (int, float)) and val >= 0, (
            f"The `size` property of {type(self).__name__} must be a positive number,\n"
            f"but received {repr(val)} instead."
        )
        self._size = val

    @property
    def color(self):
        """Marker color."""
        return self._color

    @color.setter
    def color(self, val):
        self._color = color_validator(val)

    @property
    def symbol(self):
        """Marker symbol. Can be one of `['.', 'o', '+', 'D', 'd', 's', 'x']`."""
        return self._symbol

    @symbol.setter
    def symbol(self, val):
        assert val is None or val in SYMBOLS_MATPLOTLIB_TO_PLOTLY, (
            f"The `symbol` property of {type(self).__name__} must be one of"
            f"{list(SYMBOLS_MATPLOTLIB_TO_PLOTLY.keys())},\n"
            f"but received {repr(val)} instead."
        )
        self._symbol = val


class Markers(BaseStyle):
    """Defines styling properties of the markers trace.

    Parameters
    ----------
    marker: dict or `Markers` object, default=None
        `Markers` object with 'color', 'symbol', 'size' properties, or dictionary with equivalent
        key/value pairs.
    """

    def __init__(self, marker=None, **kwargs):
        super().__init__(marker=marker, **kwargs)

    @property
    def marker(self):
        """`Markers` object with 'color', 'symbol', 'size' properties."""
        return self._marker

    @marker.setter
    def marker(self, val):
        self._marker = validate_property_class(val, "marker", Marker, self)


class DipoleProperties:
    """Defines styling properties of dipoles.

    Parameters
    ----------
    size: float
        Positive value for ratio of dipole size to canvas size.

    pivot: str
        The part of the arrow that is anchored to the X, Y grid.
        The arrow rotates about this point. Can be one of `['tail', 'middle', 'tip']`.
    """

    _allowed_pivots = ("tail", "middle", "tip")

    @property
    def size(self):
        """Positive value for ratio of dipole size to canvas size."""
        return self._size

    @size.setter
    def size(self, val):
        assert val is None or isinstance(val, (int, float)) and val >= 0, (
            f"The `size` property of {type(self).__name__} must be a positive number,\n"
            f"but received {repr(val)} instead."
        )
        self._size = val

    @property
    def pivot(self):
        """The part of the arrow that is anchored to the X, Y grid.
        The arrow rotates about this point. Can be one of `['tail', 'middle', 'tip']`.
        """
        return self._pivot

    @pivot.setter
    def pivot(self, val):
        assert val is None or val in (self._allowed_pivots), (
            f"The `pivot` property of {type(self).__name__} must be one of "
            f"{self._allowed_pivots},\n"
            f"but received {repr(val)} instead."
        )
        self._pivot = val


class Dipole(MagicProperties, DipoleProperties):
    """
    Defines styling properties of dipoles.

    Parameters
    ----------
    size: float, default=None
        Positive float for ratio of dipole size to canvas size.

    pivot: str, default=None
        The part of the arrow that is anchored to the X, Y grid.
        The arrow rotates about this point. Can be one of `['tail', 'middle', 'tip']`.
    """

    def __init__(self, size=None, pivot=None, **kwargs):
        super().__init__(size=size, pivot=pivot, **kwargs)


class DipoleStyle(BaseStyle, DipoleProperties):
    """Defines the styling properties of dipole objects.

    Parameters
    ----------
    label: str, default=None
        Label of the class instance, e.g. to be displayed in the legend.

    description: dict or `Description` object, default=None
        Object description properties.

    color: str, default=None
        A valid css color. Can also be one of `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`.

    opacity: float, default=None
        Object opacity between 0 and 1, where 1 is fully opaque and 0 is fully transparent.

    path: dict or `Path` object, default=None
        An instance of `Path` or dictionary of equivalent key/value pairs, defining the object
        path marker and path line properties.

    model3d: list of `Trace3d` objects, default=None
        A list of traces where each is an instance of `Trace3d` or dictionary of equivalent
        key/value pairs. Defines properties for an additional user-defined model3d object which is
        positioned relatively to the main object to be displayed and moved automatically with it.
        This feature also allows the user to replace the original 3d representation of the object.

    size: float, default=None
        Positive float for ratio of dipole size to canvas size.

    pivot: str, default=None
        The part of the arrow that is anchored to the X, Y grid.
        The arrow rotates about this point. Can be one of `['tail', 'middle', 'tip']`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Path(MagicProperties):
    """Defines styling properties of an object's path.

    Parameters
    ----------
    marker: dict or `Markers` object, default=None
        `Markers` object with 'color', 'symbol', 'size' properties, or dictionary with equivalent
        key/value pairs.

    line: dict or `Line` object, default=None
        `Line` object with 'color', 'symbol', 'size' properties, or dictionary with equivalent
        key/value pairs.

    show: bool, default=None
        Show/hide path.
        - False: Shows object(s) at final path position and hides paths lines and markers.
        - True: Shows object(s) shows object paths depending on `line`, `marker` and `frames`
        parameters.

    frames: int or array_like, shape (n,), default=None
        Show copies of the 3D-model along the given path indices.
        - integer i: Displays the object(s) at every i'th path position.
        - array_like, shape (n,), dtype=int: Displays object(s) at given path indices.

    numbering: bool, default=False
        Show/hide numbering on path positions.
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
        """`Markers` object with 'color', 'symbol', 'size' properties."""
        return self._marker

    @marker.setter
    def marker(self, val):
        self._marker = validate_property_class(val, "marker", Marker, self)

    @property
    def line(self):
        """`Line` object with 'color', 'type', 'width' properties."""
        return self._line

    @line.setter
    def line(self, val):
        self._line = validate_property_class(val, "line", Line, self)

    @property
    def show(self):
        """Show/hide path.
        - False: Shows object(s) at final path position and hides paths lines and markers.
        - True: Shows object(s) shows object paths depending on `line`, `marker` and `frames`
        parameters.
        """
        return self._show

    @show.setter
    def show(self, val):
        assert val is None or isinstance(val, bool), (
            f"The `show` property of {type(self).__name__} must be either True or False,\n"
            f"but received {repr(val)} instead."
        )
        self._show = val

    @property
    def frames(self):
        """Show copies of the 3D-model along the given path indices.
        - integer i: Displays the object(s) at every i'th path position.
        - array_like shape (n,) of integers: Displays object(s) at given path indices.
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
        assert is_valid_path, f"""The `frames` property of {type(self).__name__} must be either:
- integer i: Displays the object(s) at every i'th path position.
- array_like, shape (n,), dtype=int: Displays object(s) at given path indices.
but received {repr(val)} instead"""
        self._frames = val

    @property
    def numbering(self):
        """Show/hide numbering on path positions. Only applies if show=True."""
        return self._numbering

    @numbering.setter
    def numbering(self, val):
        assert val is None or isinstance(val, bool), (
            f"The `numbering` property of {type(self).__name__} must be one of (True, False),\n"
            f"but received {repr(val)} instead."
        )
        self._numbering = val


class Line(MagicProperties):
    """Defines line styling properties.

    Parameters
    ----------
    style: str, default=None
        Can be one of:
        `['solid', '-', 'dashed', '--', 'dashdot', '-.', 'dotted', '.', (0, (1, 1)),
        'loosely dotted', 'loosely dashdotted']`

    color: str, default=None
        Line color.

    width: float, default=None
        Positive number that defines the line width.
    """

    def __init__(self, style=None, color=None, width=None, **kwargs):
        super().__init__(style=style, color=color, width=width, **kwargs)

    @property
    def style(self):
        """Line style."""
        return self._style

    @style.setter
    def style(self, val):
        assert val is None or val in LINESTYLES_MATPLOTLIB_TO_PLOTLY, (
            f"The `style` property of {type(self).__name__} must be one of "
            f"{list(LINESTYLES_MATPLOTLIB_TO_PLOTLY.keys())},\n"
            f"but received {repr(val)} instead."
        )
        self._style = val

    @property
    def color(self):
        """Line color."""
        return self._color

    @color.setter
    def color(self, val):
        self._color = color_validator(val)

    @property
    def width(self):
        """Positive number that defines the line width."""
        return self._width

    @width.setter
    def width(self, val):
        assert val is None or isinstance(val, (int, float)) and val >= 0, (
            f"The `width` property of {type(self).__name__} must be a positive number,\n"
            f"but received {repr(val)} instead."
        )
        self._width = val


class DisplayStyle(MagicProperties):
    """Base class containing styling properties for all object families. The properties of the
    sub-classes are set to hard coded defaults at class instantiation.

    Parameters
    ----------
    base: dict or `Base` object, default=None
        Base properties common to all families.

    magnet: dict or `Magnet` object, default=None
        Magnet properties.

    current: dict or `Current` object, default=None
        Current properties.

    dipole: dict or `Dipole` object, default=None
        Dipole properties.

    sensor: dict or `Sensor` object, default=None
        Sensor properties.

    markers: dict or `Markers` object, default=None
        Markers properties.
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
        """Resets all nested properties to their hard coded default values."""
        self.update(get_defaults_dict("display.style"), _match_properties=False)
        return self

    @property
    def base(self):
        """Base properties common to all families."""
        return self._base

    @base.setter
    def base(self, val):
        self._base = validate_property_class(val, "base", BaseStyle, self)

    @property
    def magnet(self):
        """Magnet class."""
        return self._magnet

    @magnet.setter
    def magnet(self, val):
        self._magnet = validate_property_class(val, "magnet", Magnet, self)

    @property
    def current(self):
        """Current class."""
        return self._current

    @current.setter
    def current(self, val):
        self._current = validate_property_class(val, "current", Current, self)

    @property
    def dipole(self):
        """Dipole class."""
        return self._dipole

    @dipole.setter
    def dipole(self, val):
        self._dipole = validate_property_class(val, "dipole", Dipole, self)

    @property
    def sensor(self):
        """Sensor class."""
        return self._sensor

    @sensor.setter
    def sensor(self, val):
        self._sensor = validate_property_class(val, "sensor", Sensor, self)

    @property
    def markers(self):
        """Markers class."""
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

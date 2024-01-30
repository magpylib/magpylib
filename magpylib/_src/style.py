"""Collection of classes for display styling."""
# pylint: disable=C0302
# pylint: disable=too-many-instance-attributes
# pylint: disable=cyclic-import
import numpy as np
import param

from magpylib._src.defaults.defaults_utility import ALLOWED_LINESTYLES
from magpylib._src.defaults.defaults_utility import ALLOWED_SYMBOLS
from magpylib._src.defaults.defaults_utility import color_validator
from magpylib._src.defaults.defaults_utility import get_defaults_dict
from magpylib._src.defaults.defaults_utility import MagicParameterized
from magpylib._src.defaults.defaults_utility import SUPPORTED_PLOTTING_BACKENDS
from magpylib._src.defaults.defaults_utility import validate_style_keys

ALLOWED_SIZEMODES = ("scaled", "absolute")

# pylint: disable=missing-class-docstring


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


def get_style(obj, default_settings, **kwargs):
    """Returns default style based on increasing priority:
    - style from defaults
    - style from object
    - style from kwargs arguments
    """
    obj_families = get_families(obj)
    # parse kwargs into style an non-style arguments
    style_kwargs = kwargs.get("style", {})
    style_kwargs.update(
        {k[6:]: v for k, v in kwargs.items() if k.startswith("style") and k != "style"}
    )

    # retrieve default style dictionary, local import to avoid circular import
    # pylint: disable=import-outside-toplevel

    default_style = default_settings.display.style
    base_style_flat = default_style.base.as_dict(flatten=True, separator="_")

    # construct object specific dictionary base on style family and default style
    for obj_family in obj_families:
        family_style = getattr(default_style, obj_family, {})
        if family_style:
            family_dict = family_style.as_dict(flatten=True, separator="_")
            base_style_flat.update(
                {k: v for k, v in family_dict.items() if v is not None}
            )
    style_kwargs = validate_style_keys(style_kwargs)

    # create style class instance and update based on precedence
    style = obj.style.copy()
    style_kwargs_specific = {
        k: v for k, v in style_kwargs.items() if k.split("_")[0] in style.as_dict()
    }
    style.update(**style_kwargs_specific, _match_properties=True)
    style.update(**base_style_flat, _match_properties=False, _replace_None_only=True)

    return style


class Description(MagicParameterized):
    show = param.Boolean(
        default=True,
        doc="if `True`, adds legend entry suffix based on value",
    )
    text = param.String(doc="Object description text")


class Legend(MagicParameterized):
    show = param.Boolean(
        default=True,
        doc="if `True`, overrides complete legend text",
    )
    text = param.String(doc="Object legend text")


class Marker(MagicParameterized):
    """Defines the styling properties of plot markers"""

    color = param.Color(
        default=None,
        allow_None=True,
        doc="""
        The marker color. Must be a valid css color or one of
        `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`.""",
    )

    size = param.Number(
        default=1,
        bounds=(0, None),
        inclusive_bounds=(True, True),
        softbounds=(1, 5),
        doc="""Marker size""",
    )

    symbol = param.Selector(
        objects=list(ALLOWED_SYMBOLS),
        doc=f"""Marker symbol. Can be one of: {ALLOWED_SYMBOLS}""",
    )


class Line(MagicParameterized):
    color = param.Color(
        default=None,
        allow_None=True,
        doc="""A valid css color""",
    )

    width = param.Number(
        default=1,
        bounds=(0, 20),
        inclusive_bounds=(True, True),
        softbounds=(1, 5),
        doc="""Line width""",
    )

    style = param.Selector(
        default=ALLOWED_LINESTYLES[0],
        objects=ALLOWED_LINESTYLES,
        doc=f"""Line style. Can be one of: {ALLOWED_LINESTYLES}""",
    )


class Arrow(Line):
    show = param.Boolean(
        default=True,
        doc="Show/hide Arrow",
    )

    size = param.Number(
        default=1,
        bounds=(0, None),
        inclusive_bounds=(False, True),
        softbounds=(1, 5),
        doc="""Arrow size""",
    )

    sizemode = param.Selector(
        default=ALLOWED_SIZEMODES[0],
        objects=ALLOWED_SIZEMODES,
        doc=f"""The way the object size gets defined. Can be one of  `{ALLOWED_SIZEMODES}`""",
    )

    offset = param.Magnitude(
        bounds=(0, 1),
        inclusive_bounds=(True, True),
        doc="""Defines the arrow offset. `offset=0` results in the arrow head to be coincident with
        the start of the line, and `offset=1` with the end.""",
    )


class Frames(MagicParameterized):
    indices = param.List(
        default=[],
        item_type=int,
        doc="""Array_like shape (n,) of integers: describes certain path indices.""",
    )

    step = param.Integer(
        default=1,
        bounds=(1, None),
        softbounds=(0, 10),
        doc="""Displays the object(s) at every i'th path position""",
    )

    mode = param.Selector(
        default="indices",
        objects=["indices", "step"],
        doc="""
        The object path frames mode.
        - step: integer i: displays the object(s) at every i'th path position.
        - indices: array_like shape (n,) of integers: describes certain path indices.""",
    )

    @param.depends("indices", watch=True)
    def _update_indices(self):
        self.mode = "indices"

    @param.depends("step", watch=True)
    def _update_step(self):
        self.mode = "step"


class Path(MagicParameterized):
    def __setattr__(self, name, value):
        if name == "frames":
            if isinstance(value, (tuple, list, np.ndarray)):
                self.frames.indices = [int(v) for v in value]
            elif (
                isinstance(value, (int, np.integer))
                and value is not False
                and value is not True
            ):
                self.frames.step = value
            else:
                super().__setattr__(name, value)
            return
        super().__setattr__(name, value)

    show = param.Boolean(
        default=True,
        doc="""
        Show/hide path
        - False: shows object(s) at final path position and hides paths lines and markers.
        - True: shows object(s) shows object paths depending on `line`, `marker` and `frames`
                parameters.""",
    )

    marker = param.ClassSelector(
        class_=Marker,
        default=Marker(),
        doc="""
        Marker class with `'color'``, 'symbol'`, `'size'` properties, or dictionary with equivalent
        key/value pairs""",
    )

    line = param.ClassSelector(
        class_=Line,
        default=Line(),
        doc="""
        Line class with `'color'``, 'width'`, `'style'` properties, or dictionary with equivalent
        key/value pairs""",
    )

    numbering = param.Boolean(
        doc="""Show/hide numbering on path positions. Only applies if show=True.""",
    )

    frames = param.ClassSelector(
        class_=Frames,
        default=Frames(),
        doc="""
        Show copies of the 3D-model along the given path indices.
        - mode: either `step` or `indices`.
        - step: integer i: displays the object(s) at every i'th path position.
        - indices: array_like shape (n,) of integers: describes certain path indices.""",
    )


class Trace3d(MagicParameterized):
    def __setattr__(self, name, value):
        validation_func = getattr(self, f"_validate_{name}", None)
        if validation_func is not None:
            value = validation_func(value)
        return super().__setattr__(name, value)

    backend = param.Selector(
        default="generic",
        objects=list(SUPPORTED_PLOTTING_BACKENDS) + ["generic"],
        doc=f"""
        Plotting backend corresponding to the trace. Can be one of
        {list(SUPPORTED_PLOTTING_BACKENDS) + ['generic']}""",
    )

    constructor = param.String(
        doc="""
        Model constructor function or method to be called to build a 3D-model object
        (e.g. 'plot_trisurf', 'Mesh3d). Must be in accordance with the given plotting backend."""
    )

    args = param.Parameter(
        default=(),
        doc="""
        Tuple or callable returning a tuple containing positional arguments for building a
        3D-model object.""",
    )

    kwargs = param.Parameter(
        default={},
        doc="""
        Dictionary or callable returning a dictionary containing the keys/values pairs for
        building a 3D-model object.""",
    )

    coordsargs = param.Dict(
        default=None,
        doc="""
        Tells Magpylib the name of the coordinate arrays to be moved or rotated.
        by default:
            - plotly backend: `{"x": "x", "y": "y", "z": "z"}`
            - matplotlib backend: `{"x": "args[0]", "y": "args[1]", "z": "args[2]}"`""",
    )

    show = param.Boolean(
        default=True,
        doc="""Show/hide model3d object based on provided trace.""",
    )

    scale = param.Number(
        default=1,
        bounds=(0, None),
        inclusive_bounds=(True, False),
        softbounds=(0.1, 5),
        doc="""
        Scaling factor by which the trace vertices coordinates should be multiplied by.
        Be aware that if the object is not centered at the global CS origin, its position will also
        be affected by scaling.""",
    )

    updatefunc = param.Callable(
        doc="""
        Callable object with no arguments. Should return a dictionary with keys from the
        trace parameters. If provided, the function is called at `show` time and updates the
        trace parameters with the output dictionary. This allows to update a trace dynamically
        depending on class attributes, and postpone the trace construction to when the object is
        displayed."""
    )

    def _validate_coordsargs(self, value):
        assert isinstance(value, dict) and all(key in value for key in "xyz"), (
            f"the `coordsargs` property of {type(self).__name__} must be "
            f"a dictionary with `'x', 'y', 'z'` keys"
            f" but received {repr(value)} instead"
        )
        return value

    def _validate_updatefunc(self, val):
        """Validate updatefunc."""
        if val is None:

            def val():
                return {}

        msg = ""
        valid_keys = self.param.values().keys()
        if not callable(val):
            msg = f"Instead received {type(val)}"
        else:
            test_val = val()
            if not isinstance(test_val, dict):
                msg = f"but callable returned type {type(test_val)}."
            else:
                bad_keys = [k for k in test_val.keys() if k not in valid_keys]
                if bad_keys:
                    msg = f"but invalid output dictionary keys received: {bad_keys}."

        assert msg == "", (
            f"The `updatefunc` property of {type(self).__name__} must be a callable returning a "
            f"dictionary with a subset of following keys: {list(valid_keys)}.\n"
            f"{msg}"
        )
        return val


class Model3d(MagicParameterized):
    def __setattr__(self, name, value):
        if name == "data":
            value = self._validate_data(value)
        return super().__setattr__(name, value)

    showdefault = param.Boolean(
        default=True,
        doc="""Show/hide default 3D-model.""",
    )

    data = param.List(
        item_type=Trace3d,
        doc="""
        A trace or list of traces where each is an instance of `Trace3d` or dictionary of equivalent
        key/value pairs. Defines properties for an additional user-defined model3d object which is
        positioned relatively to the main object to be displayed and moved automatically with it.
        This feature also allows the user to replace the original 3d representation of the object.
        """,
    )

    @staticmethod
    def _validate_trace(trace, **kwargs):
        updatefunc = None
        if trace is None:
            trace = Trace3d()
        if not isinstance(trace, Trace3d) and callable(trace):
            updatefunc = trace
            trace = Trace3d()
        if isinstance(trace, dict):
            trace = Trace3d(**trace)
        if isinstance(trace, Trace3d):
            trace.updatefunc = updatefunc
            if kwargs:
                trace.update(**kwargs)
            trace.update(trace.updatefunc())
        return trace

    def _validate_data(self, traces):
        if traces is None:
            traces = []
        elif not isinstance(traces, (list, tuple)):
            traces = [traces]
        new_traces = []
        for trace in traces:
            new_traces.append(self._validate_trace(trace))
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
            Show/hide model3d object based on provided trace.

        scale: float, default=1
            Scaling factor by which the trace vertices coordinates are multiplied.

        updatefunc: callable, default=None
            Callable object with no arguments. Should return a dictionary with keys from the
            trace parameters. If provided, the function is called at `show` time and updates the
            trace parameters with the output dictionary. This allows to update a trace dynamically
            depending on class attributes, and postpone the trace construction to when the object is
            displayed.
        """
        self.data = list(self.data) + [self._validate_trace(trace, **kwargs)]
        return self


class BaseStyle(MagicParameterized):
    label = param.String(doc="Label of the class instance, can be any string.")

    description = param.ClassSelector(
        class_=Description,
        default=Description(),
        doc="Object description properties such as `text` and `show`.",
    )

    legend = param.ClassSelector(
        class_=Legend,
        default=Legend(),
        doc="Object description properties such as `text` and `show`.",
    )

    color = param.Color(
        default=None,
        allow_None=True,
        doc="A valid css color. Can also be one of `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`.",
    )

    opacity = param.Number(
        default=1,
        bounds=(0, 1),
        doc="Object opacity between 0 and 1, where 1 is fully opaque and 0 is fully transparent.",
    )

    path = param.ClassSelector(
        class_=Path,
        default=Path(),
        doc="""
        An instance of `Path` or dictionary of equivalent key/value pairs, defining the object path
        marker and path line properties.""",
    )

    model3d = param.ClassSelector(
        class_=Model3d,
        default=Model3d(),
        doc="""
        A list of traces where each is an instance of `Trace3d` or dictionary of equivalent
        key/value pairs. Defines properties for an additional user-defined model3d object which is
        positioned relatively to the main object to be displayed and moved automatically with it.
        This feature also allows the user to replace the original 3d representation of the object.
        """,
    )


class MagnetizationColor(MagicParameterized):
    _allowed_modes = ("tricolor", "bicolor", "tricycle")

    north = param.Color(
        default="red",
        doc="""
        The color of the magnetic north pole. Must be a valid css color or one of
        `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`.""",
    )

    south = param.Color(
        default="green",
        doc="""
        The color of the magnetic south pole. Must be a valid css color or one of
        `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`.""",
    )

    middle = param.Color(
        default="grey",
        doc="""
        The color between the magnetic poles. Must be a valid css color or one of
        `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`.""",
    )

    transition = param.Number(
        default=0.2,
        bounds=(0, 1),
        inclusive_bounds=(True, True),
        doc="""
        Sets the transition smoothness between poles colors. Must be between 0 and 1.
        - `transition=0`: discrete transition
        - `transition=1`: smoothest transition
        """,
    )

    mode = param.Selector(
        default=_allowed_modes[0],
        objects=_allowed_modes,
        doc="""
        Sets the coloring mode for the magnetization.
        - `'bicolor'`: only north and south poles are shown, middle color is hidden.
        - `'tricolor'`: both pole colors and middle color are shown.
        - `'tricycle'`: both pole colors are shown and middle color is replaced by a color cycling
        through the color sequence.""",
    )


class Magnetization(MagicParameterized):
    _allowed_modes = ("auto", "arrow", "color", "arrow+color", "color+arrow")

    show = param.Boolean(
        default=True,
        doc="""Show/hide magnetization based on active plotting backend""",
    )

    size = param.Number(
        default=1,
        bounds=(0, None),
        inclusive_bounds=(True, True),
        softbounds=(1, 5),
        doc="""
        Arrow size of the magnetization direction (for the matplotlib backend only), only applies if
        `show=True`""",
    )

    color = param.ClassSelector(
        class_=MagnetizationColor,
        default=MagnetizationColor(),
        doc="""
        Color properties showing the magnetization direction, only applies if `show=True`
        and `mode` contains 'color'""",
    )

    arrow = param.ClassSelector(
        class_=Arrow,
        default=Arrow(),
        doc="""
        Arrow properties showing the magnetization direction, only applies if `show=True`
        and `mode` contains 'arrow'""",
    )

    mode = param.Selector(
        default=_allowed_modes[0],
        objects=_allowed_modes,
        doc="""
        One of {"auto", "arrow", "color", "arrow+color"}, default="auto"
        Magnetization can be displayed via arrows, color or both. By default `mode='auto'` means
        that the chosen backend determines which mode is applied by its capability. If the backend
        can display both and `auto` is chosen, the priority is given to `color`.""",
    )


class MagnetSpecific(MagicParameterized):
    magnetization = param.ClassSelector(
        class_=Magnetization,
        default=Magnetization(),
        doc="""
        Magnetization styling with `'show'`, `'size'`, `'color'` properties or a dictionary with
        equivalent key/value pairs""",
    )


class ArrowCoordSysSingle(MagicParameterized):
    show = param.Boolean(
        default=True,
        doc="""Show/hide single CS arrow""",
    )

    color = param.Color(
        default=None,
        allow_None=True,
        doc="""
        The color of a single CS arrow. Must be a valid css color or one of
        `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`.""",
    )


class ArrowCoordSys(MagicParameterized):
    x = param.ClassSelector(
        class_=ArrowCoordSysSingle,
        default=ArrowCoordSysSingle(),
        doc="""
        `Arrowsingle` class or dict with equivalent key/value pairs for x-direction
        (e.g. `color`, `show`)""",
    )

    y = param.ClassSelector(
        class_=ArrowCoordSysSingle,
        default=ArrowCoordSysSingle(),
        doc="""
        `Arrowsingle` class or dict with equivalent key/value pairs for y-direction
        (e.g. `color`, `show`)""",
    )

    z = param.ClassSelector(
        class_=ArrowCoordSysSingle,
        default=ArrowCoordSysSingle(),
        doc="""
        `Arrowsingle` class or dict with equivalent key/value pairs for z-direction
        (e.g. `color`, `show`)""",
    )


class Pixel(MagicParameterized):
    size = param.Number(
        default=1,
        bounds=(0, None),
        inclusive_bounds=(True, None),
        softbounds=(0.5, 2),
        doc="""
        The relative pixel size.
        - matplotlib backend: pixel size is the marker size
        - plotly backend:  relative size to the distance of nearest neighboring pixel""",
    )

    sizemode = param.Selector(
        default=ALLOWED_SIZEMODES[0],
        objects=ALLOWED_SIZEMODES,
        doc=f"""The way the object size gets defined. Can be one of  `{ALLOWED_SIZEMODES}`""",
    )

    color = param.Color(
        default=None,
        allow_None=True,
        doc="""
        The color of sensor pixel. Must be a valid css color or one of
        `['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w']`.""",
    )

    symbol = param.Selector(
        default="o",
        objects=list(ALLOWED_SYMBOLS),
        doc=f"""
        Marker symbol. Can be one of:
        {ALLOWED_SYMBOLS}""",
    )


class SensorSpecific(MagicParameterized):
    size = param.Number(
        default=1,
        bounds=(0, None),
        inclusive_bounds=(True, True),
        softbounds=(1, 5),
        doc="""Sensor size relative to the canvas size.""",
    )
    sizemode = param.Selector(
        default=ALLOWED_SIZEMODES[0],
        objects=ALLOWED_SIZEMODES,
        doc=f"""The way the object size gets defined. Can be one of  `{ALLOWED_SIZEMODES}`""",
    )
    arrows = param.ClassSelector(
        class_=ArrowCoordSys,
        default=ArrowCoordSys(),
        doc="""`ArrowCS` class or dict with equivalent key/value pairs (e.g. `color`, `size`)""",
    )

    pixel = param.ClassSelector(
        class_=Pixel,
        default=Pixel(),
        doc="""`Pixel` class or dict with equivalent key/value pairs (e.g. `color`, `size`)""",
    )


class CurrentLine(Line):
    show = param.Boolean(
        default=True,
        doc="""Show/hide current direction arrow""",
    )


class CurrentSpecific(MagicParameterized):
    arrow = param.ClassSelector(
        class_=Arrow,
        default=Arrow(),
        doc="""
        `Arrow` class or dict with equivalent key/value pairs""",
    )

    line = param.ClassSelector(
        class_=CurrentLine,
        default=CurrentLine(),
        doc="""
        Line class with `'color'``, 'width'`, `'style'` properties, or dictionary with equivalent
        key/value pairs""",
    )


class DipoleSpecific(MagicParameterized):
    _allowed_pivots = ("tail", "middle", "tip")

    size = param.Number(
        default=1,
        bounds=(0, None),
        inclusive_bounds=(True, True),
        softbounds=(0.5, 5),
        doc="""The dipole arrow size relative to the canvas size""",
    )

    sizemode = param.Selector(
        default=ALLOWED_SIZEMODES[0],
        objects=ALLOWED_SIZEMODES,
        doc=f"""The way the object size gets defined. Can be one of  `{ALLOWED_SIZEMODES}`""",
    )

    pivot = param.Selector(
        default="middle",
        objects=_allowed_pivots,
        doc="""The part of the arrow that is anchored to the X, Y grid. The arrow rotates about
        this point. Can be one of `['tail', 'middle', 'tip']`""",
    )


class MarkerLineSpecific(MagicParameterized):
    show = param.Boolean(
        default=True,
        doc="""
        Show/hide path
        - False: shows object(s) at final path position and hides paths lines and markers.
        - True: shows object(s) shows object paths depending on `line`, `marker` and `frames`
                parameters.""",
    )

    marker = param.ClassSelector(
        class_=Marker,
        default=Marker(),
        doc="""
        Marker class with `'color'``, 'symbol'`, `'size'` properties, or dictionary with equivalent
        key/value pairs""",
    )

    line = param.ClassSelector(
        class_=Line,
        default=Line(),
        doc="""
        Line class with `'color'``, 'width'`, `'style'` properties, or dictionary with equivalent
        key/value pairs""",
    )


class GridMesh(MarkerLineSpecific):
    ...


class OpenMesh(MarkerLineSpecific):
    ...


class SelfIntersectingMesh(MarkerLineSpecific):
    ...


class DisconnectedMesh(MarkerLineSpecific):
    def __setattr__(self, name, value):
        if name == "colorsequence":
            value = [
                color_validator(v, allow_None=False, parent_name="Colorsequence")
                for v in value
            ]
        return super().__setattr__(name, value)

    colorsequence = param.List(
        doc="""
        An iterable of color values used to cycle trough for every disconnected part of
        disconnected triangular mesh object. A color may be specified by
            - a hex string (e.g. '#ff0000')
            - an rgb/rgba string (e.g. 'rgb(255,0,0)')
            - an hsl/hsla string (e.g. 'hsl(0,100%,50%)')
            - an hsv/hsva string (e.g. 'hsv(0,100%,100%)')
            - a named CSS color""",
    )


class Mesh(MagicParameterized):
    grid = param.ClassSelector(
        class_=GridMesh,
        default=GridMesh(),
        doc="""`GridMesh` properties.""",
    )

    open = param.ClassSelector(
        class_=OpenMesh,
        default=OpenMesh(),
        doc="""`OpenMesh` properties.""",
    )

    disconnected = param.ClassSelector(
        class_=DisconnectedMesh,
        default=DisconnectedMesh(),
        doc="""`DisconnectedMesh` properties.""",
    )

    selfintersecting = param.ClassSelector(
        class_=SelfIntersectingMesh,
        default=SelfIntersectingMesh(),
        doc="""`SelfIntersectingMesh` properties.""",
    )


class Orientation(MagicParameterized):
    _allowed_symbols = ("cone", "arrow3d")

    show = param.Boolean(
        default=True,
        doc="Show/hide orientation symbol.",
    )

    size = param.Number(
        default=1,
        bounds=(0, None),
        inclusive_bounds=(True, True),
        softbounds=(1, 5),
        doc="""Size of the orientation symbol""",
    )

    color = param.Color(
        default=None,
        allow_None=True,
        doc="""A valid css color""",
    )

    offset = param.Magnitude(
        bounds=(0, 1),
        inclusive_bounds=(True, True),
        doc="""
            Defines the orientation symbol offset, normal to the triangle surface. `offset=0` results
        in the cone/arrow head to be coincident to the triangle surface and `offset=1` with the
        base""",
    )
    symbol = param.Selector(
        objects=_allowed_symbols,
        doc=f"""Orientation symbol for the triangular faces. Can be one of: {_allowed_symbols}""",
    )


class TriangleSpecific(MagnetSpecific):
    orientation = param.ClassSelector(
        class_=Orientation,
        default=Orientation(),
        doc="""`Orientation` properties.""",
    )


class TriangularMeshSpecific(TriangleSpecific):
    mesh = param.ClassSelector(
        class_=Mesh,
        default=Mesh(),
        doc="""`Mesh` properties.""",
    )


class MagnetStyle(BaseStyle, MagnetSpecific):
    ...


class CurrentStyle(BaseStyle, CurrentSpecific):
    ...


class DipoleStyle(BaseStyle, DipoleSpecific):
    ...


class SensorStyle(BaseStyle, SensorSpecific):
    ...


class TriangleStyle(BaseStyle, TriangleSpecific):
    ...


class TriangularMeshStyle(BaseStyle, TriangularMeshSpecific):
    ...


class MarkersStyleSpecific(MagicParameterized):
    marker = param.ClassSelector(
        class_=Marker,
        default=Marker(),
        doc="""
        Marker class with `'color'``, 'symbol'`, `'size'` properties, or dictionary with equivalent
        key/value pairs""",
    )


class DisplayStyle(MagicParameterized):
    """
    Base class containing styling properties for all object families. The properties of the
    sub-classes get set to hard coded defaults at class instantiation.
    """

    def reset(self):
        """Resets all nested properties to their hard coded default values"""
        self.update(get_defaults_dict("display.style"), _match_properties=False)
        return self

    base = param.ClassSelector(
        class_=BaseStyle,
        default=BaseStyle(),
        doc="""Base properties common to all families""",
    )

    magnet = param.ClassSelector(
        class_=MagnetSpecific,
        default=MagnetSpecific(),
        doc="""Magnet properties""",
    )

    current = param.ClassSelector(
        class_=CurrentSpecific,
        default=CurrentSpecific(),
        doc="""Current properties""",
    )

    triangularmesh = param.ClassSelector(
        class_=TriangularMeshSpecific,
        default=TriangularMeshSpecific(),
        doc="""Triangularmesh properties""",
    )

    triangle = param.ClassSelector(
        class_=TriangleSpecific,
        default=TriangleSpecific(),
        doc="""Triangularmesh properties""",
    )

    dipole = param.ClassSelector(
        class_=DipoleSpecific,
        default=DipoleSpecific(),
        doc="""Dipole properties""",
    )

    sensor = param.ClassSelector(
        class_=SensorSpecific,
        default=SensorSpecific(),
        doc="""Sensor properties""",
    )

    markers = param.ClassSelector(
        class_=MarkersStyleSpecific,
        default=MarkersStyleSpecific(),
        doc="""Markers properties""",
    )

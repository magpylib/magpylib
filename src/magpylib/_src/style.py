"""Collection of classes for display styling.

All classes are declarative `PropertyNode` trees: each field is declared once
as a typed descriptor and gains validation, magic underscore access, dict
round-tripping, schema introspection, change observation and layered
resolution generically (see ``magpylib._src.defaults.property_tree``).
"""

# pylint: disable=too-many-lines

import re

import numpy as np

from magpylib._src.defaults.defaults_utility import (
    ALLOWED_LINESTYLES,
    ALLOWED_SYMBOLS,
    SUPPORTED_PLOTTING_BACKENDS,
    get_defaults_dict,
    validate_style_keys,
)
from magpylib._src.defaults.property_tree import (
    Boolean,
    Choice,
    Color,
    ColorSequence,
    Nested,
    Number,
    Property,
    PropertyNode,
    String,
)

ALLOWED_SIZEMODES = ("scaled", "absolute")


def get_families(obj):
    """Return the style family names of an object, generic first.

    Families are declared as a ``_style_family`` class attribute and name the
    sections of the default style (``magpy.defaults.display.style``) that apply
    to the object, e.g. ``("magnet", "triangularmesh")`` for a TriangularMesh.
    """
    return getattr(type(obj), "_style_family", ())


def get_style(obj, default_settings, **kwargs):
    """Return the resolved style of an object with increasing priority:

    - hardcoded/library defaults (base, then object family sections of
      ``default_settings.display.style``)
    - style explicitly set on the object
    - style from (show) kwargs arguments
    """
    # parse kwargs into style and non-style arguments
    style_kwargs = kwargs.get("style") or {}
    style_kwargs.update(
        {k[6:]: v for k, v in kwargs.items() if k.startswith("style") and k != "style"}
    )
    style_kwargs = validate_style_keys(style_kwargs)

    # apply object style and kwargs (highest priority)
    style = obj.style.copy()
    style_kwargs_specific = {
        k: v for k, v in style_kwargs.items() if k.split("_")[0] in style._fields
    }
    style.update(style_kwargs_specific)

    # fill unset properties from the defaults layers, most specific first
    default_style = default_settings.display.style
    family_layers = [getattr(default_style, fam) for fam in reversed(get_families(obj))]
    return style.merged(*family_layers, default_style.base)


# field descriptors specific to style classes ---------------------------


class Frames(Property):
    """Path frames: an integer interval or an iterable of path indices."""

    kind = "frames"

    def validate(self, obj, value):
        is_valid = True
        if hasattr(value, "__iter__") and not isinstance(value, str):
            value = tuple(value)
            if not all(
                np.issubdtype(type(v), np.integer) and not isinstance(v, bool)
                for v in value
            ):
                is_valid = False
        elif not (
            np.issubdtype(type(value), np.integer) and not isinstance(value, bool)
        ):
            is_valid = False
        if not is_valid:
            msg = (
                f"Input frames of {type(obj).__name__} must be either: "
                "integer i (displays the objects at every i'th path position) or "
                "array-like, shape (n,), dtype=int (displays objects at given path "
                f"indices; instead received {value!r}."
            )
            raise ValueError(msg)
        return value


class TupleOrCallable(Property):
    """A tuple, or a callable returning one."""

    def validate(self, obj, value):
        test_value = value() if callable(value) else value
        if not isinstance(test_value, tuple):
            self._fail(obj, value, "a tuple or a callable returning a tuple")
        return value


class DictOrCallable(Property):
    """A dict, or a callable returning one."""

    def validate(self, obj, value):
        test_value = value() if callable(value) else value
        if not isinstance(test_value, dict):
            self._fail(obj, value, "a dictionary or a callable returning a dictionary")
        return value


class CoordsArgs(Property):
    """A dict with 'x', 'y', 'z' keys naming coordinate arrays."""

    def validate(self, obj, value):
        if not (isinstance(value, dict) and all(key in value for key in "xyz")):
            self._fail(obj, value, "a dictionary with 'x', 'y', 'z' keys")
        return value


def _empty_updatefunc():
    return {}


class UpdateFunc(Property):
    """A callable with no arguments returning a dict of Trace3d parameters."""

    def validate(self, obj, value):
        msg = ""
        valid_props = list(type(obj)._fields)
        if not callable(value):
            msg = f"; instead received {type(value).__name__}."
        else:
            test_value = value()
            if not isinstance(test_value, dict):
                msg = f"; callable returned type {type(test_value).__name__}."
            else:
                bad_keys = [k for k in test_value if k not in valid_props]
                if bad_keys:
                    msg = f"; invalid output dictionary keys received {bad_keys}."
        if msg:
            full_msg = (
                f"Input updatefunc of {type(obj).__name__} must be a callable returning "
                f"a dictionary with a subset of these keys: {valid_props}{msg}"
            )
            raise ValueError(full_msg)
        return value


class Regex(Property):
    """A string matching a regex pattern."""

    def __init__(self, pattern, default=None, doc=""):
        super().__init__(default, doc)
        self.pattern = pattern

    def validate(self, obj, value):
        if not re.match(self.pattern, str(value)):
            self._fail(obj, value, f"matching the regex pattern {self.pattern}")
        return value

    def schema(self):
        return {**super().schema(), "type": ["string", "null"], "pattern": self.pattern}


class PixelSource(Property):
    """Pixel field source, e.g. 'Bx', 'Hxy', 'J', a tuple of those, or False."""

    _allowed_vectors = ("B", "H", "M", "J")

    def validate(self, obj, value):
        valid = True
        if value is not False:
            field_str, *coords_str = value
            if not coords_str:
                coords_str = list("xyz")
            if field_str not in self._allowed_vectors and set(coords_str).difference(
                set("xyz")
            ):
                valid = False
        if not valid:
            msg = (
                f"Input source of {type(obj).__name__} must be None or False or start"
                f" with either {self._allowed_vectors} and be followed by a combination"
                f" of 'x', 'y', 'z' (e.g. 'Bxy' or ('Bxy', 'Bz') );"
                f" instead received {value!r}."
            )
            raise ValueError(msg)
        return value


# style classes ----------------------------------------------------------


class Line(PropertyNode):
    """Defines line styling properties.

    Parameters
    ----------
    style: str, default=None
        Line style, one of ALLOWED_LINESTYLES.
    color: str, default=None
        Line color.
    width: float, default=None
        Positive number that defines the line width.
    """

    style = Choice(*ALLOWED_LINESTYLES, doc="Line style.")
    color = Color(doc="Line color.")
    width = Number(minimum=0, doc="Positive number that defines the line width.")


class Arrow(Line):
    """Defines styling properties of current and magnetization arrows.

    Parameters
    ----------
    show: bool, default=None
        Show/hide arrow.
    size: float, default=None
        Positive number defining the arrow size.
    sizemode: {'scaled', 'absolute'}, default=None
        Scale reference for the arrow size.
    offset: float, default=None
        Arrow offset. ``offset=0`` puts the arrow head coincident to the start
        of the line, ``offset=1`` to the end.
    style, color, width:
        See `Line`.
    """

    show = Boolean(doc="Show/hide arrow.")
    size = Number(minimum=0, doc="Positive number defining the size of the arrows.")
    sizemode = Choice(*ALLOWED_SIZEMODES, doc="Sizemode of the arrows.")
    offset = Number(minimum=0, maximum=1, doc="Arrow offset between 0 and 1.")


class CurrentLine(Line):
    """Defines styling properties of current lines.

    Parameters
    ----------
    show: bool, default=None
        Show/hide current line.
    style, color, width:
        See `Line`.
    """

    show = Boolean(doc="Show/hide current line.")


class Marker(PropertyNode):
    """Defines styling properties of plot markers.

    Parameters
    ----------
    size: float, default=None
        Marker size.
    color: str, default=None
        Marker color.
    symbol: str, default=None
        Marker symbol, one of ALLOWED_SYMBOLS.
    """

    size = Number(minimum=0, doc="Marker size.")
    color = Color(doc="Marker color.")
    symbol = Choice(*ALLOWED_SYMBOLS, doc="Marker symbol.")


class Description(PropertyNode):
    """Defines properties for a description object.

    Parameters
    ----------
    text: str, default=None
        Object description text.
    show: bool, default=None
        If True, adds legend entry suffix based on value.
    """

    text = String(doc="Description text.")
    show = Boolean(doc="If True, adds legend entry suffix based on value.")


class Legend(PropertyNode):
    """Defines properties for a legend object.

    Parameters
    ----------
    text: str, default=None
        Legend text.
    show: bool, default=None
        If True, adds legend entry based on value.
    """

    text = String(doc="Legend text.")
    show = Boolean(doc="If True, adds legend entry based on value.")


class Trace3d(PropertyNode):
    """User-defined 3D model trace that moves/rotates with its parent object.

    Parameters
    ----------
    backend : {'generic', 'matplotlib', 'plotly'}, default 'generic'
        Plotting backend for this trace.
    constructor : str | None, default None
        Name of the constructor function/method to build the 3D model (e.g.,
        ``'plot_trisurf'``, ``'Mesh3d'``). Must match the selected backend.
    args : tuple | callable | None, default None
        Positional arguments for the constructor, or a callable returning them.
    kwargs : dict | callable | None, default None
        Keyword arguments for the constructor, or a callable returning them.
    coordsargs : dict | None, default None
        Names of coordinate arrays to be transformed; by default
        ``{"x": "x", "y": "y", "z": "z"}``.
    show : bool, default True
        Show or hide the resulting model3d object.
    scale : float, default 1
        Multiplier applied to the trace vertex coordinates.
    updatefunc : callable | None, default None
        Callable with no arguments returning a dictionary of trace parameters
        to update at show time.
    """

    backend = Choice(
        "generic",
        *SUPPORTED_PLOTTING_BACKENDS,
        default="generic",
        doc="Plotting backend for this trace.",
    )
    constructor = String(doc="Constructor name to build the 3D model.")
    args = TupleOrCallable(doc="Positional arguments for the constructor.")
    kwargs = DictOrCallable(doc="Keyword arguments for the constructor.")
    coordsargs = CoordsArgs(doc="Names of coordinate arrays to be transformed.")
    show = Boolean(default=True, doc="Show or hide the model3d trace.")
    scale = Number(
        exclusive_minimum=0,
        default=1,
        doc="Multiplier applied to the trace vertex coordinates.",
    )
    updatefunc = UpdateFunc(
        default=_empty_updatefunc,
        doc="Callable updating the trace parameters at show time.",
    )


class TraceData(Property):
    """List of Trace3d objects; accepts single/list of Trace3d, dict or callable."""

    default = ()

    def __init__(self, doc=""):
        super().__init__((), doc)

    def validate(self, obj, value):
        return obj._validate_data(value)


class Model3d(PropertyNode):
    """Defines properties for the 3d model representation of magpylib objects.

    Parameters
    ----------
    showdefault: bool, default=True
        Shows/hides default 3d-model.
    data: dict or list of dicts, default=None
        A trace or list of traces where each is an instance of `Trace3d` or
        dictionary of equivalent key/value pairs.
    """

    showdefault = Boolean(default=True, doc="Shows/hides default 3d-model.")
    data = TraceData(doc="Data of 3d object representation (trace or list of traces).")

    @staticmethod
    def _validate_data(traces, **kwargs):
        if traces is None:
            traces = []
        elif not isinstance(traces, list | tuple):
            traces = [traces]
        new_traces = []
        for trace_item in traces:
            trace = trace_item
            updatefunc = None
            if not isinstance(trace, Trace3d) and callable(trace):
                updatefunc = trace
                trace = Trace3d()
            if trace is None:
                trace = Trace3d()
            elif isinstance(trace, dict):
                trace = Trace3d(**trace)
            if not isinstance(trace, Trace3d):
                msg = (
                    "The data property of Model3d must be an instance of Trace3d or "
                    "a dictionary with equivalent key/value pairs; "
                    f"instead received {trace!r}."
                )
                raise TypeError(msg)
            if updatefunc is not None:
                trace.updatefunc = updatefunc
            trace = trace.update(kwargs)
            new_traces.append(trace)
        return new_traces

    def add_trace(self, trace=None, **kwargs):
        """Add a user-defined 3D trace that moves/rotates with this object.

        Parameters
        ----------
        trace : Trace3d | dict | callable | None, default None
            A trace, a dict with equivalent key/value pairs, or a callable
            returning such a dict. If a callable is given, it is used as
            ``updatefunc`` and a default ``Trace3d`` is created.
        **kwargs
            Trace3d properties, see `Trace3d`.

        Returns
        -------
        Model3d
        """
        self.data = [*self.data, *self._validate_data([trace], **kwargs)]
        return self


class Path(PropertyNode):
    """Defines styling properties of an object's path.

    Parameters
    ----------
    show: bool, default=None
        Show/hide path.
    marker: dict or `Marker` object, default=None
        Path marker properties.
    line: dict or `Line` object, default=None
        Path line properties.
    frames: int or array-like, shape (n,), default=None
        Show copies of the 3D-model along the given path indices.
        - integer i: Displays the object(s) at every i'th path position.
        - array-like, shape (n,), dtype=int: Displays object(s) at given path
          indices.
    numbering: bool, default=False
        Show/hide numbering on path positions.
    """

    show = Boolean(doc="Show/hide path.")
    marker = Nested(Marker, doc="Path marker properties.")
    line = Nested(Line, doc="Path line properties.")
    frames = Frames(doc="Show copies of the 3D-model along the given path indices.")
    numbering = Boolean(doc="Show/hide numbering on path positions.")


class BaseStyle(PropertyNode):
    """Base class for display styling options of `_BaseGeo` objects.

    Parameters
    ----------
    label: str, default=None
        Label of the class instance, e.g. to be displayed in the legend.
    description: dict or `Description` object, default=None
        Object description properties.
    legend: dict or `Legend` object, default=None
        Object legend properties. Legend has the `{label} ({description})`
        format.
    color: str, default=None
        A valid css color. Can also be one of `['r', 'g', 'b', 'y', 'm', 'c',
        'k', 'w']`.
    opacity: float, default=None
        Object opacity between 0 and 1, where 1 is fully opaque and 0 is fully
        transparent.
    path: dict or `Path` object, default=None
        Object path marker and path line properties.
    model3d: dict or `Model3d` object, default=None
        Properties for an additional user-defined 3d model object which is
        positioned relatively to the main object to be displayed and moved
        automatically with it. Can also replace the original 3d representation.
    """

    label = String(coerce=True, doc="Label of the class instance.")
    description = Nested(
        Description, from_str="text", doc="Object description properties."
    )
    legend = Nested(Legend, from_str="text", doc="Object legend properties.")
    color = Color(doc="A valid css color.")
    opacity = Number(minimum=0, maximum=1, doc="Object opacity between 0 and 1.")
    path = Nested(Path, doc="Object path marker and path line properties.")
    model3d = Nested(Model3d, doc="3d object representation properties.")


class MagnetizationColor(PropertyNode):
    """Defines the magnetization direction color styling properties. (Only
    relevant for the plotly backend)

    Parameters
    ----------
    north: str, default=None
        Color of the magnetic north pole.
    south: str, default=None
        Color of the magnetic south pole.
    middle: str, default=None
        Color between the magnetic poles.
    transition: float, default=None
        Transition smoothness between pole colors, between 0 (discrete) and
        1 (smooth).
    mode: {'bicolor', 'tricolor', 'tricycle'}, default=None
        Coloring mode for the magnetization.
        - `'bicolor'`: Only north and south pole colors are shown.
        - `'tricolor'`: Both pole colors and middle color are shown.
        - `'tricycle'`: Both pole colors are shown and middle color is
          replaced by a color cycling through the default color sequence.
    """

    north = Color(doc="Color of the magnetic north pole.")
    south = Color(doc="Color of the magnetic south pole.")
    middle = Color(doc="Color between the magnetic poles.")
    transition = Number(
        minimum=0, maximum=1, doc="Transition smoothness between pole colors."
    )
    mode = Choice(
        "bicolor", "tricolor", "tricycle", doc="Coloring mode for the magnetization."
    )


class Magnetization(PropertyNode):
    """Defines magnetization styling properties.

    Parameters
    ----------
    show : bool, default=None
        If ``True`` show magnetization direction.
    arrow: dict or `Arrow` object, default=None
        Arrow properties. Only applies if `mode='arrow'`.
    color: dict or `MagnetizationColor` object, default=None
        Color properties showing the magnetization direction (for the plotly
        backend). Only applies if `show=True`.
    mode: {'auto', 'arrow', 'color', 'arrow+color'}, default=None
        Magnetization can be displayed via arrows, color or both. With
        `mode='auto'` the chosen backend determines the mode by capability.
    """

    show = Boolean(doc="If True, show magnetization direction.")
    arrow = Nested(Arrow, doc="Magnetization arrow properties.")
    color = Nested(MagnetizationColor, doc="Magnetization color properties.")
    mode = Choice(
        "auto",
        "arrow",
        "color",
        "arrow+color",
        "color+arrow",
        doc="Magnetization display mode.",
    )

    @property
    def size(self):
        """Deprecated (please use arrow.size): Arrow size property."""
        return self.arrow.size

    @size.setter
    def size(self, val):
        if val is not None:
            self.arrow.size = val


class MagnetProperties(PropertyNode):
    """Defines styling properties of homogeneous magnet classes."""

    magnetization = Nested(Magnetization, doc="Magnetization styling properties.")


class DefaultMagnet(MagnetProperties):
    """Defines styling properties of homogeneous magnet classes.

    Parameters
    ----------
    magnetization: dict or `Magnetization` object, default=None
    """


class MagnetStyle(BaseStyle, MagnetProperties):
    """Styling properties for homogeneous magnet classes.

    Parameters
    ----------
    magnetization: dict or `Magnetization` object, default=None
        Magnetization styling properties.
    label, description, legend, color, opacity, path, model3d:
        See `BaseStyle`.
    """


class MarkerLineProperties(PropertyNode):
    """Defines styling properties of Markers and Lines."""

    show = Boolean(doc="Show/hide lines and markers.")
    marker = Nested(Marker, doc="Marker properties.")
    line = Nested(Line, doc="Line properties.")


class GridMesh(MarkerLineProperties):
    """Defines styling properties of GridMesh objects."""


class OpenMesh(MarkerLineProperties):
    """Defines styling properties of OpenMesh objects."""


class DisconnectedMesh(MarkerLineProperties):
    """Defines styling properties of DisconnectedMesh objects.

    Parameters
    ----------
    show: bool, default=None
        Show/hide Lines and Markers.
    marker: dict or `Marker` object, default=None
    line: dict or `Line` object, default=None
    colorsequence: iterable, default=None
        An iterable of color values used to cycle through for every
        disconnected part of disconnected triangular mesh object.
    """

    colorsequence = ColorSequence(
        doc="Colors cycled through for every disconnected mesh part."
    )


class SelfIntersectingMesh(MarkerLineProperties):
    """Defines styling properties of SelfIntersectingMesh objects."""


class TriMesh(PropertyNode):
    """Defines TriMesh mesh properties.

    Parameters
    ----------
    grid: dict or `GridMesh` object, default=None
        All mesh vertices and edges of a TriangularMesh object.
    open: dict or `OpenMesh` object, default=None
        Shows open mesh vertices and edges of a TriangularMesh object, if any.
    disconnected: dict or `DisconnectedMesh` object, default=None
        Shows disconnected bodies of a TriangularMesh object, if any.
    selfintersecting: dict or `SelfIntersectingMesh` object, default=None
        Shows self-intersecting triangles of a TriangularMesh object, if any.
    """

    grid = Nested(GridMesh, doc="All mesh vertices and edges.")
    open = Nested(OpenMesh, doc="Open mesh vertices and edges, if any.")
    disconnected = Nested(DisconnectedMesh, doc="Disconnected mesh bodies, if any.")
    selfintersecting = Nested(
        SelfIntersectingMesh, doc="Self-intersecting triangles, if any."
    )


class Orientation(PropertyNode):
    """Defines Triangle orientation properties.

    Parameters
    ----------
    show: bool, default=None
        Show/hide orientation symbol.
    size: float, default=None
        Size of the orientation symbol.
    color: str, default=None
        A valid css color.
    symbol: {'cone', 'arrow3d'}, default=None
        Orientation symbol for the triangular faces.
    offset: float, default=None
        Orientation symbol offset, normal to the triangle surface. ``offset=0``
        results in the cone/arrow head being coincident to the triangle surface
        and ``offset=1`` with the base.
    """

    show = Boolean(doc="Show/hide orientation symbol.")
    size = Number(minimum=0, doc="Size of the orientation symbol.")
    color = Color(doc="Orientation symbol color.")
    symbol = Choice("cone", "arrow3d", doc="Orientation symbol.")
    offset = Number(doc="Orientation symbol offset, normal to the triangle surface.")


class TriangleProperties(PropertyNode):
    """Defines Triangle properties."""

    orientation = Nested(Orientation, doc="Orientation styling of triangles.")


class DefaultTriangle(MagnetProperties, TriangleProperties):
    """Defines styling properties of the Triangle class.

    Parameters
    ----------
    magnetization: dict or `Magnetization` object, default=None
    orientation: dict or `Orientation` object, default=None
    """


class TriangleStyle(MagnetStyle, TriangleProperties):
    """Defines styling properties of the Triangle class.

    Parameters
    ----------
    magnetization: dict or `Magnetization` object, default=None
    orientation: dict or `Orientation` object, default=None
    label, description, legend, color, opacity, path, model3d:
        See `BaseStyle`.
    """


class TriangularMeshProperties(PropertyNode):
    """Defines TriangularMesh properties."""

    mesh = Nested(TriMesh, doc="TriMesh styling properties.")


class DefaultTriangularMesh(
    MagnetProperties, TriangleProperties, TriangularMeshProperties
):
    """Defines styling properties of homogeneous TriangularMesh magnet classes.

    Parameters
    ----------
    magnetization: dict or `Magnetization` object, default=None
    orientation: dict or `Orientation` object, default=None
    mesh: dict or `TriMesh` object, default=None
    """


class TriangularMeshStyle(MagnetStyle, TriangleProperties, TriangularMeshProperties):
    """Defines styling properties of the TriangularMesh magnet class.

    Parameters
    ----------
    magnetization: dict or `Magnetization` object, default=None
    orientation: dict or `Orientation` object, default=None
    mesh: dict or `TriMesh` object, default=None
    label, description, legend, color, opacity, path, model3d:
        See `BaseStyle`.
    """


class CurrentGridMesh(MarkerLineProperties):
    """Defines styling properties of CurrentGridMesh objects."""


class CurrentMesh(PropertyNode):
    """Defines CurrentSheet mesh properties.

    Parameters
    ----------
    grid: dict or `CurrentGridMesh` object, default=None
        All mesh vertices and edges of a CurrentSheet object.
    """

    grid = Nested(CurrentGridMesh, doc="All mesh vertices and edges.")


class CurrentDirection(PropertyNode):
    """Defines CurrentSheet direction properties.

    Parameters
    ----------
    show: bool, default=None
        Show/hide direction symbol.
    size: float, default=None
        Size of the direction symbol.
    color: str, default=None
        A valid css color.
    symbol: {'cone', 'arrow3d'}, default=None
        Current direction symbol for the triangular faces.
    """

    show = Boolean(doc="Show/hide direction symbol.")
    size = Number(minimum=0, doc="Size of the direction symbol.")
    color = Color(doc="Direction symbol color.")
    symbol = Choice("cone", "arrow3d", doc="Current direction symbol.")


class CurrentSheetProperties(PropertyNode):
    """Defines CurrentSheet properties."""

    direction = Nested(CurrentDirection, doc="Current direction styling.")
    mesh = Nested(CurrentMesh, doc="CurrentMesh styling properties.")


class DefaultCurrentSheet(CurrentSheetProperties):
    """Defines styling properties of the CurrentSheet classes.

    Parameters
    ----------
    direction: dict or `CurrentDirection` object, default=None
    mesh: dict or `CurrentMesh` object, default=None
    """


class CurrentSheetStyle(BaseStyle, CurrentSheetProperties):
    """Defines styling properties of the CurrentSheet classes.

    Parameters
    ----------
    direction: dict or `CurrentDirection` object, default=None
        Current direction styling.
    mesh: dict or `CurrentMesh` object, default=None
        CurrentMesh styling properties.
    label, description, legend, color, opacity, path, model3d:
        See `BaseStyle`.
    """


class ArrowSingle(PropertyNode):
    """Single coordinate system arrow properties.

    Parameters
    ----------
    show: bool, default=True
        Show/hide arrow.
    color: str, default=None
        A valid css color.
    """

    show = Boolean(default=True, doc="Show/hide arrow.")
    color = Color(doc="Arrow color.")


class ArrowCS(PropertyNode):
    """Defines triple coordinate system arrow properties.

    Parameters
    ----------
    x: dict or `ArrowSingle` object, default=None
        x-direction arrow properties (e.g. `color`, `show`).
    y: dict or `ArrowSingle` object, default=None
        y-direction arrow properties (e.g. `color`, `show`).
    z: dict or `ArrowSingle` object, default=None
        z-direction arrow properties (e.g. `color`, `show`).
    """

    x = Nested(ArrowSingle, doc="x-direction arrow properties.")
    y = Nested(ArrowSingle, doc="y-direction arrow properties.")
    z = Nested(ArrowSingle, doc="z-direction arrow properties.")


class PixelField(PropertyNode):
    """Defines the styling properties of sensor pixel fields.

    Parameters
    ----------
    source: str, default=None
        The pixel color source (e.g. "Bx", "Hxy", "J", etc.).
    colormap: str, default=None
        The colormap used with `source`.
    shownull: bool, default=None
        Show/hide null or invalid field values.
    symbol: {'cone', 'arrow', 'arrow3d', 'none'}, default=None
        Orientation symbol for field vector.
    sizescaling: {'uniform', 'linear', 'log', 'log^[2-9]'}, default=None
        Symbol size scaling relative to the field magnitude.
    sizemin: float, default=None
        Minimum relative size of field symbols (0 to 1).
    colorscaling: {'uniform', 'linear', 'log', 'log^[2-9]'}, default=None
        Color scale scaling relative to the field magnitude.
    """

    _allowed_scalings_pattern = r"^(uniform|linear|(log)+|log\^[2-9])$"

    source = PixelSource(doc="Pixel field source.")
    colormap = Choice(
        "Viridis",
        "Jet",
        "Rainbow",
        "Plasma",
        "Inferno",
        "Magma",
        "Cividis",
        "Greys",
        "Purples",
        "Blues",
        "Greens",
        "Oranges",
        "Reds",
        "YlOrBr",
        "YlOrRd",
        "OrRd",
        "PuRd",
        "RdPu",
        "BuPu",
        "GnBu",
        "PuBu",
        "YlGnBu",
        "PuBuGn",
        "BuGn",
        "YlGn",
        doc="Colormap used with source.",
    )
    shownull = Boolean(doc="Show/hide null or invalid field values.")
    symbol = Choice(
        "cone", "arrow", "arrow3d", "none", doc="Orientation symbol for field vector."
    )
    sizescaling = Regex(
        _allowed_scalings_pattern, doc="Symbol size scaling relative to field."
    )
    sizemin = Number(
        minimum=0, maximum=1, doc="Minimum relative size of field symbols."
    )
    colorscaling = Regex(
        _allowed_scalings_pattern, doc="Color scale scaling relative to field."
    )


class Pixel(PropertyNode):
    """Defines the styling properties of sensor pixels.

    Parameters
    ----------
    size: float, default=1
        Positive float for relative pixel size.
    sizemode: {'scaled', 'absolute'}, default=None
        Scale reference for the pixel size.
    color: str, default=None
        Pixel color.
    symbol: str, default=None
        Pixel symbol, one of `['cube', '.', 'o', '+', 'D', 'd', 's', 'x']`.
    field: dict or `PixelField` object, default=None
        Pixel field styling properties.
    """

    size = Number(minimum=0, default=1, doc="Positive float for relative pixel size.")
    sizemode = Choice(*ALLOWED_SIZEMODES, doc="Sizemode of the pixel.")
    color = Color(doc="Pixel color.")
    symbol = Choice("cube", *ALLOWED_SYMBOLS, doc="Pixel symbol.")
    field = Nested(PixelField, doc="Pixel field styling properties.")


class SensorProperties(PropertyNode):
    """Defines the specific styling properties of the Sensor class."""

    size = Number(minimum=0, doc="Positive float for ratio of sensor to canvas size.")
    sizemode = Choice(*ALLOWED_SIZEMODES, doc="Sizemode of the sensor.")
    pixel = Nested(Pixel, doc="Pixel styling properties.")
    arrows = Nested(ArrowCS, doc="Coordinate system arrows properties.")


class DefaultSensor(SensorProperties):
    """Defines styling properties of the Sensor class.

    Parameters
    ----------
    size: float, default=None
    sizemode: {'scaled', 'absolute'}, default=None
    pixel: dict or `Pixel` object, default=None
    arrows: dict or `ArrowCS` object, default=None
    """


class SensorStyle(BaseStyle, SensorProperties):
    """Styling properties for the Sensor class.

    Parameters
    ----------
    size: float, default=None
        Positive float for ratio of sensor to canvas size.
    sizemode: {'scaled', 'absolute'}, default=None
        Scale reference for the sensor size.
    pixel: dict or `Pixel` object, default=None
        Pixel styling properties.
    arrows: dict or `ArrowCS` object, default=None
        Coordinate system arrows properties.
    label, description, legend, color, opacity, path, model3d:
        See `BaseStyle`.
    """


class CurrentProperties(PropertyNode):
    """Defines styling properties of line current classes."""

    arrow = Nested(Arrow, doc="Current arrow properties.")
    line = Nested(CurrentLine, doc="Current line properties.")


class DefaultCurrent(CurrentProperties):
    """Defines the specific styling properties of line current classes.

    Parameters
    ----------
    arrow: dict or `Arrow` object, default=None
    line: dict or `CurrentLine` object, default=None
    """


class CurrentStyle(BaseStyle, CurrentProperties):
    """Styling properties for line current classes.

    Parameters
    ----------
    arrow: dict or `Arrow` object, default=None
        Current arrow properties.
    line: dict or `CurrentLine` object, default=None
        Current line properties.
    label, description, legend, color, opacity, path, model3d:
        See `BaseStyle`.
    """


class DefaultMarkers(BaseStyle):
    """Defines styling properties of the markers trace.

    Parameters
    ----------
    marker: dict or `Marker` object, default=None
        Marker properties.
    """

    marker = Nested(Marker, doc="Marker properties.")


class DipoleProperties(PropertyNode):
    """Defines styling properties of dipoles."""

    size = Number(minimum=0, doc="Positive float for ratio of dipole to canvas size.")
    sizemode = Choice(*ALLOWED_SIZEMODES, doc="Sizemode of the dipole.")
    pivot = Choice(
        "tail",
        "middle",
        "tip",
        doc="Part of the arrow anchored to the grid, about which it rotates.",
    )


class DefaultDipole(DipoleProperties):
    """Defines styling properties of dipoles.

    Parameters
    ----------
    size: float, default=None
    sizemode: {'scaled', 'absolute'}, default=None
    pivot: {'tail', 'middle', 'tip'}, default=None
    """


class DipoleStyle(BaseStyle, DipoleProperties):
    """Styling properties for dipole objects.

    Parameters
    ----------
    size: float, default=None
        Positive float for ratio of dipole to canvas size.
    sizemode: {'scaled', 'absolute'}, default=None
        Scale reference for the dipole size.
    pivot: {'tail', 'middle', 'tip'}, default=None
        The part of the arrow anchored to the grid about which it rotates.
    label, description, legend, color, opacity, path, model3d:
        See `BaseStyle`.
    """


class DisplayStyle(PropertyNode):
    """Base class containing styling properties for all object families.

    Parameters
    ----------
    base: dict or `BaseStyle` object, default=None
        Base properties common to all families.
    magnet: dict or `DefaultMagnet` object, default=None
        Magnet properties.
    current: dict or `DefaultCurrent` object, default=None
        Current properties.
    currentsheet: dict or `DefaultCurrentSheet` object, default=None
        CurrentSheet properties.
    dipole: dict or `DefaultDipole` object, default=None
        Dipole properties.
    triangle: dict or `DefaultTriangle` object, default=None
        Triangle properties.
    triangularmesh: dict or `DefaultTriangularMesh` object, default=None
        TriangularMesh properties.
    sensor: dict or `DefaultSensor` object, default=None
        Sensor properties.
    markers: dict or `DefaultMarkers` object, default=None
        Markers properties.
    """

    base = Nested(BaseStyle, doc="Base properties common to all families.")
    magnet = Nested(DefaultMagnet, doc="Magnet default style.")
    current = Nested(DefaultCurrent, doc="Current default style.")
    currentsheet = Nested(DefaultCurrentSheet, doc="CurrentSheet default style.")
    dipole = Nested(DefaultDipole, doc="Dipole default style.")
    triangle = Nested(DefaultTriangle, doc="Triangle default style.")
    triangularmesh = Nested(DefaultTriangularMesh, doc="TriangularMesh default style.")
    sensor = Nested(DefaultSensor, doc="Sensor default style.")
    markers = Nested(DefaultMarkers, doc="Markers default style.")

    def reset(self):
        """Resets all nested properties to their hard coded default values."""
        self.update(get_defaults_dict("display.style"), _match_properties=False)
        return self

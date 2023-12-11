"""Polyline current class code"""
import warnings

from magpylib._src.display.traces_core import make_Polyline
from magpylib._src.exceptions import MagpylibDeprecationWarning
from magpylib._src.fields.field_BH_polyline import current_vertices_field
from magpylib._src.input_checks import check_format_input_vertices
from magpylib._src.obj_classes.class_BaseExcitations import BaseCurrent
from magpylib._src.utility import unit_prefix


class Polyline(BaseCurrent):
    """Current flowing in straight lines from vertex to vertex.

    Can be used as `sources` input for magnetic field computation.

    The vertex positions are defined in the local object coordinates (rotate with object).
    When `position=(0,0,0)` and `orientation=None` global and local coordinates coincide.

    Parameters
    ----------
    current: float, default=`None`
        Electrical current in units of A.

    vertices: array_like, shape (n,3), default=`None`
        The current flows along the vertices which are given in units of mm in the
        local object coordinates (move/rotate with object). At least two vertices
        must be given.

    position: array_like, shape (3,) or (m,3), default=`(0,0,0)`
        Object position(s) in the global coordinates in units of mm. For m>1, the
        `position` and `orientation` attributes together represent an object path.

    orientation: scipy `Rotation` object with length 1 or m, default=`None`
        Object orientation(s) in the global coordinates. `None` corresponds to
        a unit-rotation. For m>1, the `position` and `orientation` attributes
        together represent an object path.

    parent: `Collection` object or `None`
        The object is a child of it's parent collection.

    style: dict
        Object style inputs must be in dictionary form, e.g. `{'color':'red'}` or
        using style underscore magic, e.g. `style_color='red'`.

    Returns
    -------
    current source: `Polyline` object

    Examples
    --------
    `Polyline` objects are magnetic field sources. In this example we compute the H-field kA/m
    of a square-shaped line-current with 1 A current at the observer position (1,1,1) given in
    units of mm:

    >>> import magpylib as magpy
    >>> src = magpy.current.Polyline(
    ...     current=1,
    ...     vertices=((1,0,0), (0,1,0), (-1,0,0), (0,-1,0), (1,0,0)),
    ... )
    >>> H = src.getH((1,1,1))
    >>> print(H)
    [0.03160639 0.03160639 0.00766876]

    We rotate the source object, and compute the B-field, this time at a set of observer positions:

    >>> src.rotate_from_angax(45, 'x')
    Polyline(id=...)
    >>> B = src.getB([(1,1,1), (2,2,2), (3,3,3)])
    >>> print(B)
    [[-6.68990257e-18  3.50341393e-02 -3.50341393e-02]
     [-5.94009823e-19  3.62181325e-03 -3.62181325e-03]
     [-2.21112416e-19  1.03643004e-03 -1.03643004e-03]]

    The same result is obtained when the rotated source moves along a path away from an
    observer at position (1,1,1). This time we use a `Sensor` object as observer.

    >>> src.move([(-1,-1,-1), (-2,-2,-2)])
    Polyline(id=...)
    >>> sens = magpy.Sensor(position=(1,1,1))
    >>> B = src.getB(sens)
    >>> print(B)
    [[-6.68990257e-18  3.50341393e-02 -3.50341393e-02]
     [-5.94009823e-19  3.62181325e-03 -3.62181325e-03]
     [-2.21112416e-19  1.03643004e-03 -1.03643004e-03]]
    """

    # pylint: disable=dangerous-default-value
    _field_func = staticmethod(current_vertices_field)
    _field_func_kwargs_ndim = {
        "current": 1,
        "vertices": 3,
        "segment_start": 2,
        "segment_end": 2,
    }
    get_trace = make_Polyline

    def __init__(
        self,
        current=None,
        vertices=None,
        position=(0, 0, 0),
        orientation=None,
        style=None,
        **kwargs,
    ):
        # instance attributes
        self.vertices = vertices

        # init inheritance
        super().__init__(position, orientation, current, style, **kwargs)

    # property getters and setters
    @property
    def vertices(self):
        """
        The current flows along the vertices which are given in units of mm in the
        local object coordinates (move/rotate with object). At least two vertices
        must be given.
        """
        return self._vertices

    @vertices.setter
    def vertices(self, vert):
        """Set Polyline vertices, array_like, mm."""
        self._vertices = check_format_input_vertices(vert)

    @property
    def _default_style_description(self):
        """Default style description text"""
        if self.vertices is None:
            return "no vertices"
        return f"{unit_prefix(self.current)}A" if self.current else "no current"


class Line(Polyline):
    """Line is deprecated, see Polyline"""

    # pylint: disable=method-hidden
    @staticmethod
    def _field_func(*args, **kwargs):
        """Catch Deprecation warning in getBH_dict"""
        _deprecation_warn()
        return current_vertices_field(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        _deprecation_warn()
        super().__init__(*args, **kwargs)


def _deprecation_warn():
    warnings.warn(
        (
            "Line is deprecated and will be removed in a future version, "
            "use Polyline instead."
        ),
        MagpylibDeprecationWarning,
        stacklevel=2,
    )

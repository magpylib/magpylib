# pylint: disable=too-many-positional-arguments

"""Polyline current class code"""

import warnings

from magpylib._src.display.traces_core import make_Polyline
from magpylib._src.exceptions import MagpylibDeprecationWarning
from magpylib._src.fields.field_BH_polyline import current_vertices_field
from magpylib._src.input_checks import check_format_input_vertices
from magpylib._src.obj_classes.class_BaseExcitations import BaseCurrent
from magpylib._src.utility import unit_prefix


class Polyline(BaseCurrent):
    """Line current flowing in straight paths from vertex to vertex.

    Can be used as `sources` input for magnetic field computation.

    The vertex positions are defined in the local object coordinates (rotate with object).
    When `position=(0,0,0)` and `orientation=None` global and local coordinates coincide.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    position: array_like, shape (3,) or (m,3), default=`(0,0,0)`
        Object position(s) in the global coordinates in units of m. For m>1, the
        `position` and `orientation` attributes together represent an object path.

    orientation: scipy `Rotation` object with length 1 or m, default=`None`
        Object orientation(s) in the global coordinates. `None` corresponds to
        a unit-rotation. For m>1, the `position` and `orientation` attributes
        together represent an object path.

    vertices: array_like, shape (n,3), default=`None`
        The current flows along the vertices which are given in units of m in the
        local object coordinates (move/rotate with object). At least two vertices
        must be given.

    current: float, default=`None`
        Electrical current in units of A.

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
    `Polyline` objects are magnetic field sources. In this example we compute the H-field in A/m
    of a square-shaped line-current with 1 A current at the observer position (1,1,1) given in
    units of m:

    >>> import magpylib as magpy
    >>> src = magpy.current.Polyline(
    ...     current=1,
    ...     vertices=((.01,0,0), (0,.01,0), (-.01,0,0), (0,-.01,0), (.01,0,0)),
    ... )
    >>> H = src.getH((.01,.01,.01))
    >>> print(H)
    [3.16063859 3.16063859 0.76687556]

    We rotate the source object, and compute the B-field, this time at a set of observer positions:

    >>> src.rotate_from_angax(90, 'x')
    Polyline(id=...)
    >>> B = src.getB([(.01,.01,.01), (.02,.02,.02), (.03,.03,.03)])
    >>> print(B)
    [[-3.97177559e-06 -9.63684251e-07 -3.97177559e-06]
     [-4.90331150e-07 -3.11039072e-08 -4.90331150e-07]
     [-1.43908549e-07 -4.10438492e-09 -1.43908549e-07]]
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
        The current flows along the vertices which are given in units of m in the
        local object coordinates (move/rotate with object). At least two vertices
        must be given.
        """
        return self._vertices

    @vertices.setter
    def vertices(self, vert):
        """Set Polyline vertices, array_like, meter."""
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

# pylint: disable=too-many-positional-arguments

"""TriangleStrip current class code"""

from magpylib._src.display.traces_core import make_TriangleStrip
from magpylib._src.fields.field_BH_current_sheet import BHJM_current_strip
from magpylib._src.input_checks import check_format_input_vertices
from magpylib._src.obj_classes.class_BaseExcitations import BaseCurrent
from magpylib._src.utility import unit_prefix

class TriangleStrip(BaseCurrent):
    """Current flowing in straight lines along a Ribbon made of adjacent Triangles.

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
        The current flows along a band which consists of Triangles {T1, T2, ...}
        defined by the vertices {V1, V2, V3, V4, ...} as T1=(V1,V2,V3),
        T2=(V2,V3,V4), ... The vertices are given in units of m in the local
        object coordinates (move/rotate with object). At least three vertices
        must be given, which define the first Triangle.

    current: float, default=`None`
        Electrical current in units of A. It is transformed into a homogeneous current
        density which flows along the Triangles in direction: T1: V1->V3, T2: V2->V4, ...

    parent: `Collection` object or `None`
        The object is a child of it's parent collection.

    style: dict
        Object style inputs must be in dictionary form, e.g. `{'color':'red'}` or
        using style underscore magic, e.g. `style_color='red'`.

    Returns
    -------
    current source: `TriangleStrip` object

    Examples
    --------
    `TriangleStrip` objects are magnetic field sources. In this example we compute the H-field in A/m
    of a square current sheet (two triangles) with 1 A current at the observer position (1,1,1) cm:

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> src = magpy.current.TriangleStrip(
    ...    current=1,
    ...    vertices=((0,0,0), (0,1,0), (1,0,0), (1,1,0)),
    ... )
    >>> H = src.getH((.01,.01,.01))
    >>> with np.printoptions(precision=3):
    ...     print(H)
    [ 3.965e-17 -2.905e-01 -3.747e-01]

    """

    # pylint: disable=dangerous-default-value
    _field_func = staticmethod(BHJM_current_strip)
    _field_func_kwargs_ndim = {
        "current": 1,
        "vertices": 3,
    }
    get_trace = make_TriangleStrip

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
        The current flows along the triangles ...
        """
        return self._vertices

    @vertices.setter
    def vertices(self, vert):
        """Set TriangleStrip vertices, array_like, meter."""
        self._vertices = check_format_input_vertices(vert)

    @property
    def _default_style_description(self):
        """Default style description text"""
        if self.vertices is None:
            return "no vertices"
        return f"{unit_prefix(self.current)}A" if self.current else "no current"


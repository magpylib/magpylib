"""Line current class code
DOCSTRINGS V4 READY
"""
from magpylib._src.input_checks import check_format_input_vertices
from magpylib._src.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._src.obj_classes.class_BaseExcitations import BaseCurrent
from magpylib._src.obj_classes.class_BaseGeo import BaseGeo
from magpylib._src.obj_classes.class_BaseGetBH import BaseGetBH


class Line(BaseGeo, BaseDisplayRepr, BaseGetBH, BaseCurrent):
    """Current flowing in straight lines from vertex to vertex.

    Can be used as `sources` input for magnetic field computation.

    The vertex positions are defined in the local object coordinates (rotate with object).
    When `position=(0,0,0)` and `orientation=None` global and local coordinates conincide.

    Parameters
    ----------
    current: float, default=`None`
        Electrical current in units of [A].

    vertices: array_like, shape (n,3), default=`None`
        The current flows along the vertices which are given in units of [mm] in the
        local object coordinates (move/rotate with object). At least two vertices
        must be given.

    position: array_like, shape (3,) or (m,3), default=`(0,0,0)`
        Object position(s) in the global coordinates in units of [mm]. For m>1, the
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
    current source: `Line` object

    Examples
    --------
    `Line` objects are magnetic field sources. In this example we compute the H-field [kA/m]
    of a square-shaped line-current with 1 [A] current at the observer position (1,1,1) given in
    units of [mm]:

    >>> import magpylib as magpy
    >>> src = magpy.current.Line(
    ...     current=1,
    ...     vertices=((1,0,0), (0,1,0), (-1,0,0), (0,-1,0), (1,0,0)),
    ... )
    >>> H = src.getH((1,1,1))
    >>> print(H)
    [0.03160639 0.03160639 0.00766876]

    We rotate the source object, and compute the B-field, this time at a set of observer positions:

    >>> src.rotate_from_angax(45, 'x')
    Line...
    >>> B = src.getB([(1,1,1), (2,2,2), (3,3,3)])
    >>> print(B)
    [[-6.68990257e-18  3.50341393e-02 -3.50341393e-02]
     [-5.94009823e-19  3.62181325e-03 -3.62181325e-03]
     [-2.21112416e-19  1.03643004e-03 -1.03643004e-03]]

    The same result is obtained when the rotated source moves along a path away from an
    observer at position (1,1,1). This time we use a `Sensor` object as observer.

    >>> src.move([(-1,-1,-1), (-2,-2,-2)])
    Line...
    >>> sens = magpy.Sensor(position=(1,1,1))
    >>> B = src.getB(sens)
    >>> print(B)
    [[-6.68990257e-18  3.50341393e-02 -3.50341393e-02]
     [-5.94009823e-19  3.62181325e-03 -3.62181325e-03]
     [-2.21112416e-19  1.03643004e-03 -1.03643004e-03]]
    """

    # pylint: disable=dangerous-default-value

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
        self._object_type = "Line"

        # init inheritance
        BaseGeo.__init__(self, position, orientation, style=style, **kwargs)
        BaseDisplayRepr.__init__(self)
        BaseCurrent.__init__(self, current)

    # property getters and setters
    @property
    def vertices(self):
        """
        The current flows along the vertices which are given in units of [mm] in the
        local object coordinates (move/rotate with object). At least two vertices
        must be given.
        """
        return self._vertices

    @vertices.setter
    def vertices(self, vert):
        """Set Line vertices, array_like, [mm]."""
        self._vertices = check_format_input_vertices(vert)

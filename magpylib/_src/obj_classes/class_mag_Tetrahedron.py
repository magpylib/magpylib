"""Magnet Tetrahedron class code
DOCSTRINGS V4 READY
"""
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._src.obj_classes.class_BaseExcitations import BaseHomMag
from magpylib._src.obj_classes.class_BaseGeo import BaseGeo
from magpylib._src.obj_classes.class_BaseGetBH import BaseGetBH


class Tetrahedron(BaseGeo, BaseDisplayRepr, BaseGetBH, BaseHomMag):
    """Tetrahedron magnet with homogeneous magnetization.

    Can be used as `sources` input for magnetic field computation.

    When `position=(0,0,0)` and `orientation=None` the Tetrahedron sides are parallel
    to the global coordinate basis vectors and the geometric center of the Tetrahedron
    is located in the origin.

    Parameters
    ----------
    magnetization: array_like, shape (3,), default=`None`
        Magnetization vector (mu0*M, remanence field) in units of [mT] given in
        the local object coordinates (rotates with object).

    vertices: ndarray, shape (4,3)
        Vertices (x1,y1,z1), (x2,y2,z2), (x3,y3,z3), (x4,y4,z4), in the relative
        coordinate system of the tetrahedron.

    position: array_like, shape (3,) or (m,3)
        Object position(s) in the global coordinates in units of [mm]. For m>1, the
        `position` and `orientation` attributes together represent an object path.
        When setting vertices, the initial position is set to the barycenter.

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
    magnet source: `Tetrahedron` object

    Examples
    --------
    `Tetrahedron` magnets are magnetic field sources. Below we compute the H-field [kA/m] of a
    cubical magnet with magnetization (100,200,300) in units of [mT] and 1 [mm] sides
    at the observer position (1,1,1) given in units of [mm]:

    >>> import magpylib as magpy
    >>> src = magpy.magnet.Tetrahedron(magnetization=(100,200,300), vertices=(1,1,1))
    >>> H = src.getH((1,1,1))
    >>> print(H)
    [6.21116976 4.9689358  3.72670185]

    We rotate the source object, and compute the B-field, this time at a set of observer positions:

    >>> src.rotate_from_angax(45, 'x')
    Tetrahedron(id=...)
    >>> B = src.getB([(1,1,1), (2,2,2), (3,3,3)])
    >>> print(B)
    [[4.30496934 6.9363475  0.50728577]
     [0.54127889 0.86827283 0.05653357]
     [0.1604214  0.25726266 0.01664045]]

    The same result is obtained when the rotated source moves along a path away from an
    observer at position (1,1,1). Here we use a `Sensor` object as observer.

    >>> sens = magpy.Sensor(position=(1,1,1))
    >>> src.move([(-1,-1,-1), (-2,-2,-2)])
    Tetrahedron(id=...)
    >>> B = src.getB(sens)
    >>> print(B)
    [[4.30496934 6.9363475  0.50728577]
     [0.54127889 0.86827283 0.05653357]
     [0.1604214  0.25726266 0.01664045]]
    """

    def __init__(
        self,
        magnetization=None,
        vertices=None,
        position=(0, 0, 0),
        orientation=None,
        style=None,
        **kwargs,
    ):

        # instance attributes
        self.vertices = vertices
        self._object_type = "Tetrahedron"

        # init inheritance
        BaseGeo.__init__(self, position, orientation, style=style, **kwargs)
        BaseDisplayRepr.__init__(self)
        BaseHomMag.__init__(self, magnetization)

    # property getters and setters
    @property
    def vertices(self):
        """Length of the Tetrahedron sides [a,b,c] in units of [mm]."""
        return self._vertices

    @vertices.setter
    def vertices(self, dim):
        """Set Tetrahedron vertices (a,b,c), shape (3,), [mm]."""
        # TODO check len=4
        self._vertices = check_format_input_vector(
            dim,
            dims=(2,),
            shape_m1=3,
            sig_name="Tetrahedron.vertices",
            sig_type="array_like (list, tuple, ndarray) of shape (4,3) with positive values",
            allow_None=True,
        )

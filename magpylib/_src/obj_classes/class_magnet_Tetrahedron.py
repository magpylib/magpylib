"""Magnet Tetrahedron class code"""
import numpy as np

from magpylib._src.display.traces_core import make_Tetrahedron
from magpylib._src.fields.field_BH_tetrahedron import magnet_tetrahedron_field
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet


class Tetrahedron(BaseMagnet):
    """Tetrahedron magnet with homogeneous magnetization.

    Can be used as `sources` input for magnetic field computation.

    When `position=(0,0,0)` and `orientation=None` the Tetrahedron vertices coordinates
    are the same as in the global coordinate system. The geometric center of the Tetrahedron
    is determined by its vertices and. It is not necessarily located in the origin an can
    be computed with the barycenter property.

    Parameters
    ----------
    magnetization: array_like, shape (3,), default=`None`
        Magnetization vector (mu0*M, remanence field) in units of mT given in
        the local object coordinates (rotates with object).

    vertices: ndarray, shape (4,3)
        Vertices [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3), (x4,y4,z4)], in the relative
        coordinate system of the tetrahedron.

    position: array_like, shape (3,) or (m,3)
        Object position(s) in the global coordinates in units of mm. For m>1, the
        `position` and `orientation` attributes together represent an object path.
        When setting vertices, the initial position is set to the barycenter.

    barycenter: array_like, shape (3,)
        Read only property that returns the geometric barycenter (=center of mass)
        of the object.

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
    `Tetrahedron` magnets are magnetic field sources. Below we compute the H-field in kA/m of a
    tetrahedron magnet with magnetization (100,200,300) in units of mT dimensions defined
    through the vertices (0,0,0), (1,0,0), (0,1,0) and (0,0,1) in units of mm at the
    observer position (1,1,1) given in units of mm:

    >>> import magpylib as magpy
    >>> verts = [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]
    >>> src = magpy.magnet.Tetrahedron(magnetization=(100,200,300), vertices=verts)
    >>> H = src.getH((1,1,1))
    >>> print(H)
    [2.07089783 1.65671826 1.2425387 ]

    We rotate the source object, and compute the B-field, this time at a set of observer positions:

    src.rotate_from_angax(45, 'x')
    Tetrahedron(id=...)
    B = src.getB([(1,1,1), (2,2,2), (3,3,3)])
    print(B)
    [[ 8.68006559e-01  2.00895792e+00 -5.03469140e-01]
    [ 1.01357229e-01  1.93731796e-01 -1.59677364e-02]
    [ 2.90426931e-02  5.22556994e-02 -1.70596096e-03]]

    The same result is obtained when the rotated source moves along a path away from an
    observer at position (1,1,1). Here we use a `Sensor` object as observer.

    sens = magpy.Sensor(position=(1,1,1))
    src.move([(-1,-1,-1), (-2,-2,-2)])
    Sensor(id=...)
    B = src.getB(sens)
    print(B)
    [[ 8.68006559e-01  2.00895792e+00 -5.03469140e-01]
    [ 1.01357229e-01  1.93731796e-01 -1.59677364e-02]
    [ 2.90426931e-02  5.22556994e-02 -1.70596096e-03]]
    """

    _field_func = staticmethod(magnet_tetrahedron_field)
    _field_func_kwargs_ndim = {"magnetization": 1, "vertices": 3}
    get_trace = make_Tetrahedron

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
        super().__init__(position, orientation, magnetization, style, **kwargs)

    # property getters and setters
    @property
    def vertices(self):
        """Length of the Tetrahedron sides [a,b,c] in units of mm."""
        return self._vertices

    @vertices.setter
    def vertices(self, dim):
        """Set Tetrahedron vertices (a,b,c), shape (3,), (mm)."""
        self._vertices = check_format_input_vector(
            dim,
            dims=(2,),
            shape_m1=3,
            length=4,
            sig_name="Tetrahedron.vertices",
            sig_type="array_like (list, tuple, ndarray) of shape (4,3)",
            allow_None=True,
        )

    @property
    def _barycenter(self):
        """Object barycenter."""
        return self._get_barycenter(self._position, self._orientation, self.vertices)

    @property
    def barycenter(self):
        """Object barycenter."""
        return np.squeeze(self._barycenter)

    @staticmethod
    def _get_barycenter(position, orientation, vertices):
        """Returns the barycenter of a tetrahedron."""
        centroid = (
            np.array([0.0, 0.0, 0.0]) if vertices is None else np.mean(vertices, axis=0)
        )
        barycenter = orientation.apply(centroid) + position
        return barycenter

    @property
    def _default_style_description(self):
        """Default style description text"""
        if self.vertices is None:
            return "no vertices"
        return ""

"""Magnet Facets class code
DOCSTRINGS V4 READY
"""
import numpy as np
from scipy.spatial import ConvexHull  # pylint: disable=no-name-in-module

from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._src.obj_classes.class_BaseExcitations import BaseHomMag
from magpylib._src.obj_classes.class_BaseGeo import BaseGeo
from magpylib._src.obj_classes.class_BaseGetBH import BaseGetBH


class Facets(BaseGeo, BaseDisplayRepr, BaseGetBH, BaseHomMag):
    """Facets magnet with homogeneous magnetization.

    Can be used as `sources` input for magnetic field computation.

    When `position=(0,0,0)` and `orientation=None` the Facets facets coordinates
    are the same as in the global coordinate system. The geometric center of the Facets
    is determined by its vertices and is not necessarily located in the origin.

    Parameters
    ----------
    magnetization: array_like, shape (3,), default=`None`
        Magnetization vector (mu0*M, remanence field) in units of [mT] given in
        the local object coordinates (rotates with object).

    facets: ndarray, shape (n,4,3)
        Facets in the relative coordinate system of the Facets object.

    position: array_like, shape (3,) or (m,3)
        Object position(s) in the global coordinates in units of [mm]. For m>1, the
        `position` and `orientation` attributes together represent an object path.
        When setting facets, the initial position is set to the barycenter.

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
    magnet source: `Facets` object

    Examples
    --------
    `Facets` magnets are magnetic field sources. Below we compute the H-field [kA/m] of a
    tetrahedral magnet with magnetization (100,200,300) in units of [mT] and 1 [mm] sides
    at the observer position (0,0,0) given in units of [mm]:

    >>> import magpylib as magpy
    >>> vertices = [(1,0,-1/2**0.5),(0,1,1/2**0.5),(-1,0,-1/2**0.5),(1,-1,1/2**0.5)]
    >>> src = magpy.magnet.Facets((100,200,300), vertices=vertices)
    >>> H = src.getH((0,0,0))
    >>> print(H)
    [  3.42521345 -40.76504699 -70.06509857]

    We rotate the source object, and compute the B-field, this time at a set of observer positions:

    >>> src.rotate_from_angax(45, 'x')
    Facets(id=...)
    >>> B = src.getB([(1,1,1), (2,2,2), (3,3,3)])
    >>> print(B)
    [[3.2653876  7.77807843 0.41141725]
     [0.49253111 0.930953   0.0763492 ]
     [0.1497206  0.26663798 0.02164654]]
    """

    def __init__(
        self,
        magnetization=None,
        facets=None,
        position=(0, 0, 0),
        orientation=None,
        style=None,
        **kwargs,
    ):

        # instance attributes
        self.facets = facets
        self._object_type = "Facets"
        self._triangles = kwargs.pop("triangles", None)
        self._vertices = kwargs.pop("vertices", None)
        if self._triangles is None and self._vertices is None:
            (
                self._vertices,
                self._triangles,
            ) = self._get_vertices_and_triangles_from_facets(facets)

        # init inheritance
        BaseGeo.__init__(self, position, orientation, style=style, **kwargs)
        BaseDisplayRepr.__init__(self)
        BaseHomMag.__init__(self, magnetization)

    # property getters and setters
    @property
    def facets(self):
        """Facets objects"""
        return self._facets

    @facets.setter
    def facets(self, val):
        """Set Facets facets (a,b,c), shape (3,), [mm]."""
        self._facets = check_format_input_vector(
            val,
            dims=(3,),
            shape_m1=3,
            sig_name="Facets.facets",
            sig_type="array_like (list, tuple, ndarray) of shape (4,3)",
            allow_None=True,
        )

    @property
    def vertices(self):
        """Object vertices"""
        # vertices = np.unique(self.facets.reshape((-1, 3)), axis=0)
        return self._vertices

    @property
    def triangles(self):
        """Object triangles"""
        return self._triangles

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
        centroid = np.mean(vertices, axis=0)
        barycenter = orientation.apply(centroid) + position
        return barycenter

    @staticmethod
    def _get_vertices_and_triangles_from_facets(facets):
        """Return vertices and triangles from facets"""
        vertices, tr = np.unique(facets.reshape((-1, 3)), axis=0, return_inverse=True)
        triangles = tr.reshape((-1, 3))
        return vertices, triangles

    @classmethod
    def from_points(
        cls,
        magnetization=None,
        points=None,
        triangles=None,
        position=(0, 0, 0),
        orientation=None,
        style=None,
        **kwargs,
    ):
        """Facets magnet with homogeneous magnetization.

        Can be used as `sources` input for magnetic field computation.

        When `position=(0,0,0)` and `orientation=None` the Facets vertices coordinates
        are the same as in the global coordinate system. The geometric center of the Facets
        is determined by its vertices and is not necessarily located in the origin.

        In this case the vertices are calculated from points and triangles. If `triangles` is `None`
        a `sciyp.spatial.ConvexHull` infers it."""

        # pylint: disable=protected-access
        if points is None:
            raise ValueError(
                "Points must be defined as an array-like object of shape (n,3)"
            )
        vertices = np.array(points)
        if triangles is None:
            hull = ConvexHull(vertices)
            triangles = hull.simplices
        triangles = np.array(triangles)
        facets = vertices[triangles]
        return cls(
            magnetization=magnetization,
            facets=facets,
            position=position,
            orientation=orientation,
            style=style,
            triangles=triangles,
            vertices=vertices,
            **kwargs,
        )

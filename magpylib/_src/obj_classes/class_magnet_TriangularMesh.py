"""Magnet TriangularMesh class code"""
import numpy as np
from scipy.spatial import ConvexHull  # pylint: disable=no-name-in-module

from magpylib._src.display.traces_generic import make_TriangularMesh
from magpylib._src.fields.field_BH_trimesh import magnet_trimesh_field
from magpylib._src.fields.field_BH_trimesh import mask_inside_trimesh
from magpylib._src.fields.field_BH_trimesh import segments_intersect_triangles
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet


class TriangularMesh(BaseMagnet):
    """Triangular surface mesh magnet with homogeneous magnetization.
    Can be used as `sources` input for magnetic field computation.
    When `position=(0,0,0)` and `orientation=None` the TriangularMesh facets coordinates
    are the same as in the global coordinate system. The geometric center of the TriangularMesh
    is determined by its vertices and is not necessarily located in the origin.
    Parameters
    ----------
    magnetization: array_like, shape (3,), default=`None`
        Magnetization vector (mu0*M, remanence field) in units of [mT] given in
        the local object coordinates (rotates with object).

    vertices: ndarray, shape (n,3)
        Points defining the vertices of the facets in the relative coordinate system of the
        TriangularMesh object.

    triangles: ndarray, shape (n,3)
        Indices corresponding to the the points (vertices) constructing each triangle of the
        TriangularMesh object.

    position: array_like, shape (3,) or (m,3)
        Object position(s) in the global coordinates in units of [mm]. For m>1, the
        `position` and `orientation` attributes together represent an object path.
        When setting facets, the initial position is set to the barycenter.

    orientation: scipy `Rotation` object with length 1 or m, default=`None`
        Object orientation(s) in the global coordinates. `None` corresponds to
        a unit-rotation. For m>1, the `position` and `orientation` attributes
        together represent an object path.

    reorient_facets: bool, optional
        If `False`, no facet orientation check is performed. If `True`, facets pointing inwards are
        fliped in the right direction.

    validate_mesh: bool, optional
        If `True`, the provided set of facets is validated by checking if it forms a closed body and
        if it does not self-intersect. Can be deactivated for perfomance reasons by setting it to
        `False`.

    parent: `Collection` object or `None`
        The object is a child of it's parent collection.

    style: dict
        Object style inputs must be in dictionary form, e.g. `{'color':'red'}` or
        using style underscore magic, e.g. `style_color='red'`.

    Returns
    -------
    magnet source: `TriangularMesh` object

    Examples
    --------
    """

    _field_func = staticmethod(magnet_trimesh_field)
    _field_func_kwargs_ndim = {"magnetization": 2, "facets": 3}
    _draw_func = make_TriangularMesh

    def __init__(
        self,
        magnetization=None,
        vertices=None,
        triangles=None,
        position=(0, 0, 0),
        orientation=None,
        reorient_facets=True,
        validate_mesh=True,
        style=None,
        **kwargs,
    ):

        # instance attributes
        self._triangles, self._vertices = self._validate_facets(
            vertices,
            triangles,
            reorient_facets=reorient_facets,
            validate_mesh=validate_mesh,
        )
        # init inheritance
        super().__init__(position, orientation, magnetization, style, **kwargs)

    # property getters and setters
    @property
    def vertices(self):
        """Facets objects"""
        return self._vertices

    @property
    def triangles(self):
        """Facets objects"""
        return self._triangles

    @property
    def facets(self):
        """Facets objects"""
        return self._vertices[self._triangles]

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

    def _validate_facets(
        self,
        vertices=None,
        triangles=None,
        reorient_facets=True,
        validate_mesh=True,
    ):
        """Validate facet input, reorient if necessary."""
        vertices = check_format_input_vector(
            vertices,
            dims=(2,),
            shape_m1=3,
            sig_name="TriangularMesh.vertices",
            sig_type="array_like (list, tuple, ndarray) of shape (n,3)",
            allow_None=True,
        )
        triangles = check_format_input_vector(
            triangles,
            dims=(2,),
            shape_m1=3,
            sig_name="TriangularMesh.triangles",
            sig_type="array_like (list, tuple, ndarray) of shape (n,3)",
            allow_None=True,
        )
        triangles = triangles.astype(int)
        print(np.max(triangles), vertices.shape[0])
        if np.max(triangles) >= vertices.shape[0]:
            raise ValueError(
                f"The triangles max index ({np.max(triangles)}) must be stricly lower than "
                f"the number of vertices ({vertices.shape[0]})"
            )
        if validate_mesh:
            tr = triangles
            edges = np.concatenate([tr[:, 0:2], tr[:, 1:3], tr[:, ::2]], axis=0)
            # make sure unique edge pairs are found regardless of vertices order
            edges = np.sort(edges, axis=1)
            edges_uniq, edges_counts = np.unique(edges, axis=0, return_counts=True)
            # if closed, each edge belongs to exactly 2 facets
            open_edges_sum = np.sum(edges_counts != 2)
            if open_edges_sum != 0:
                raise ValueError(
                    f"Provided set of facets result in {open_edges_sum} open edges"
                )
            intersecting_edges = segments_intersect_triangles(
                vertices[edges_uniq].swapaxes(0, 1), vertices[triangles].swapaxes(0, 1)
            )
            intersecting_edges_sum = np.sum(intersecting_edges != 0)
            if intersecting_edges_sum != 0:
                raise ValueError(
                    "Provided set of facets result in at least "
                    f"{intersecting_edges_sum} edges intersecting faces"
                )
        if reorient_facets:
            triangles = self._flip_facets_outwards(vertices, triangles)
        return triangles, vertices

    @staticmethod
    def _flip_facets_outwards(vertices, triangles, tol=1e-8):
        """Flip facets pointing inwards"""

        facets = vertices[triangles]

        facet_centers = facets.mean(axis=1)

        # calculate vectors normal to the facets
        a = facets[:, 0, :] - facets[:, 1, :]
        b = facets[:, 1, :] - facets[:, 2, :]
        facet_orient_vec = np.cross(a, b)
        facet_orient_vec_norm = np.linalg.norm(facet_orient_vec, axis=0)

        # move vertices from facet centers towards face orientation
        check_points = facet_centers + facet_orient_vec * tol / facet_orient_vec_norm

        # find points which are now inside
        inside_mask = mask_inside_trimesh(check_points, facets)

        # flip triangles which point inside
        triangles[inside_mask] = triangles[inside_mask][:, [0, 2, 1]]
        return triangles

    @classmethod
    def from_ConvexHull_points(
        cls,
        magnetization=None,
        points=None,
        position=(0, 0, 0),
        orientation=None,
        style=None,
        **kwargs,
    ):
        """Triangular surface mesh magnet with homogeneous magnetization.
        Can be used as `sources` input for magnetic field computation.
        When `position=(0,0,0)` and `orientation=None` the TriangularMesh vertices coordinates
        are the same as in the global coordinate system. The geometric center of the TriangularMesh
        is determined by its vertices and is not necessarily located in the origin.

        Using this class methods allows to construct a TriangularMesh object from a cloud of
        `points` . The `triangles` are constructed via `sciyp.spatial.ConvexHull`.

        Parameters
        ----------
        magnetization: array_like, shape (3,), default=`None`
            Magnetization vector (mu0*M, remanence field) in units of [mT] given in
            the local object coordinates (rotates with object).

        points: ndarray, shape (n,3)
            Points defining the vertices of the facets in the relative coordinate system of the
            TriangularMesh object.

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
        magnet source: `TriangularMesh` object

        Examples
        --------
        """

        # apply facet flip since ConvexHull does not guarantee that the facets are all
        # pointing outwards
        return cls(
            magnetization=magnetization,
            vertices=points,
            triangles=ConvexHull(points).simplices,
            position=position,
            orientation=orientation,
            reorient_facets=True,
            validate_mesh=True,
            style=style,
            **kwargs,
        )

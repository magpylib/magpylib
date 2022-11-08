"""Magnet Facet class code
"""
# from warnings import warn

import numpy as np
# from scipy.spatial import ConvexHull  # pylint: disable=no-name-in-module

from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet
from magpylib._src.fields.field_BH_facet import magnet_facet_field_from_obj
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.display.traces_generic import make_Facet


class Facet(BaseMagnet):
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

    facets: ndarray, shape (n,4,3)
        Facets in the relative coordinate system of the TriangularMesh object.

    position: array_like, shape (3,) or (m,3)
        Object position(s) in the global coordinates in units of [mm]. For m>1, the
        `position` and `orientation` attributes together represent an object path.
        When setting facets, the initial position is set to the barycenter.

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
    magnet source: `TriangularMesh` object

    Examples
    --------
    """

    _field_func = staticmethod(magnet_facet_field_from_obj)
    _field_func_kwargs_ndim = {"magnetization": 2, "facets": 3}
    _draw_func = make_Facet

    def __init__(
        self,
        magnetization=None,
        facets=None,
        position=(0, 0, 0),
        orientation=None,
        #reorient_facets=True,
        #check_euler_charasteristic=True,
        style=None,
        **kwargs,
    ):

        # instance attributes
        #triangles = kwargs.pop("triangles", None)
        #vertices = kwargs.pop("vertices", None)
        #self._facets, self._triangles, self._vertices = self._validate_facets(
        #    facets,
        #    vertices,
        #    triangles,
        #    reorient_facets=reorient_facets,
        #    check_euler_charasteristic=check_euler_charasteristic,
        #)
        
        self.facets = facets
    
        # init inheritance
        super().__init__(position, orientation, magnetization, style, **kwargs)

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
            dims=(2,3),
            shape_m1=3,
            sig_name="Facet.facets",
            sig_type="array_like (list, tuple, ndarray) of shape (n,3,3)",
            reshape=(-1,3,3),
            allow_None=True,
        )
        # self._facets, self._triangles, self._vertices = self._validate_facets(
        #     facets=val,
        #     reorient_facets=True,
        #     check_euler_charasteristic=True,
        # )

    @property
    def _barycenter(self):
        """Object barycenter."""
        return self._get_barycenter(self._position, self._orientation, self._facets)

    @property
    def barycenter(self):
        """Object barycenter."""
        return np.squeeze(self._barycenter)

    @staticmethod
    def _get_barycenter(position, orientation, facets):
        """Returns the barycenter of a tetrahedron."""
        centroid = np.mean(facets, axis=(0,1))
        barycenter = orientation.apply(centroid) + position
        return barycenter

    # def _validate_facets(
    #     self,
    #     facets=None,
    #     vertices=None,
    #     triangles=None,
    #     reorient_facets=True,
    #     check_euler_charasteristic=True,
    # ):
    #     """Validate facet input, reorient if necessary."""
    #     facets = check_format_input_vector(
    #         facets,
    #         dims=(3,),
    #         shape_m1=3,
    #         sig_name="TriangularMesh.facets",
    #         sig_type="array_like (list, tuple, ndarray) of shape (4,3)",
    #         allow_None=True,
    #     )
    #     if facets is None and not reorient_facets:
    #         facets = vertices[triangles]
    #     elif facets is not None:
    #         vertices, triangles = self._get_vertices_and_triangles_from_facets(facets)
    #     if check_euler_charasteristic:
    #         # find unique pairs of vertices for each facet
    #         tr = triangles
    #         edges = np.concatenate([tr[:, 0:2], tr[:, 1:3], tr[:, ::2]], axis=0)
    #         # make sure unique pairs are found regardless of order
    #         edges = np.sort(edges, axis=1)
    #         edges_uniq = np.unique(edges, axis=0)
    #         # check Euler-Poincaré characteristic, if convex and closed: V + F - E = 2.
    #         euler = len(vertices) + len(facets) - len(edges_uniq)
    #         if euler != 2:
    #             warn(
    #                 f"Provided set of vertices result in Euler-Poincaré charasteristic of {euler}"
    #                 "and my not create a closed surface."
    #             )
    #     if reorient_facets:
    #         triangles = self._flip_facets_outwards(vertices, triangles)
    #         facets = vertices[triangles]
    #     return facets, triangles, vertices

    # @property
    # def vertices(self):
    #     """Object vertices"""
    #     # vertices = np.unique(self.facets.reshape((-1, 3)), axis=0)
    #     return self._vertices

    # @property
    # def triangles(self):
    #     """Object triangles"""
    #     return self._triangles


    # @staticmethod
    # def _get_vertices_and_triangles_from_facets(facets):
    #     """Return vertices and triangles from facets"""
    #     vertices, tr = np.unique(facets.reshape((-1, 3)), axis=0, return_inverse=True)
    #     triangles = tr.reshape((-1, 3))
    #     return vertices, triangles

    # @staticmethod
    # def _flip_facets_outwards(vertices, triangles, tol=1e-8):
    #     """Flip facets pointing inwards"""

    #     # pylint: disable=import-outside-toplevel
    #     from magpylib._src.fields.field_BH_facet import mask_inside_trimesh

    #     facets = vertices[triangles]

    #     facet_centers = facets.mean(axis=1)

    #     # calculate vectors normal to the facets
    #     a = facets[:, 0, :] - facets[:, 1, :]
    #     b = facets[:, 1, :] - facets[:, 2, :]
    #     facet_orient_vec = np.cross(a, b)
    #     facet_orient_vec_norm = np.linalg.norm(facet_orient_vec, axis=0)

    #     # move vertices from facet centers towards face orientation
    #     check_points = facet_centers + facet_orient_vec * tol / facet_orient_vec_norm

    #     # find points which are now inside
    #     inside_mask = mask_inside_trimesh(check_points, facets)

    #     # flip triangles which point inside
    #     triangles[inside_mask] = triangles[inside_mask][:, [0, 2, 1]]
    #     return triangles

    # @classmethod
    # def from_points(
    #     cls,
    #     magnetization=None,
    #     points=None,
    #     triangles="ConvexHull",
    #     position=(0, 0, 0),
    #     orientation=None,
    #     style=None,
    #     reorient_facets=True,
    #     check_euler_charasteristic=True,
    #     **kwargs,
    # ):
    #     """Triangular surface mesh magnet with homogeneous magnetization.

    #     Can be used as `sources` input for magnetic field computation.

    #     When `position=(0,0,0)` and `orientation=None` the TriangularMesh vertices coordinates
    #     are the same as in the global coordinate system. The geometric center of the TriangularMesh
    #     is determined by its vertices and is not necessarily located in the origin.

    #     In this case the facets are constructed from `points` and `triangles`. By default,
    #     `triangles` are constructed via sciyp.spatial.ConvexHull`.
    #     """

    #     # pylint: disable=protected-access
    #     if points is None:
    #         raise ValueError(
    #             "Points must be defined as an array-like object of shape (n,3)"
    #         )
    #     vertices = np.array(points)
    #     if isinstance(triangles, str) and triangles.lower() == "convexhull":
    #         hull = ConvexHull(vertices)
    #         triangles = hull.simplices
    #         # apply facet flip since ConvexHull does not guarantee that the facets are all
    #         # pointing outwards
    #         reorient_facets = True
    #     triangles = np.array(triangles)
    #     facets = vertices[triangles]
    #     return cls(
    #         magnetization=magnetization,
    #         facets=facets,
    #         position=position,
    #         orientation=orientation,
    #         style=style,
    #         triangles=triangles,
    #         vertices=vertices,
    #         reorient_facets=reorient_facets,
    #         check_euler_charasteristic=check_euler_charasteristic,
    #         **kwargs,
    #     )

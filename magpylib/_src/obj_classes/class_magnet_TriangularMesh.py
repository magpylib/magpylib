"""Magnet TriangularMesh class code"""
import numpy as np
from scipy.spatial import ConvexHull  # pylint: disable=no-name-in-module

from magpylib._src.display.traces_generic import make_TriangularMesh
from magpylib._src.fields.field_BH_triangularmesh import fix_trimesh_orientation
from magpylib._src.fields.field_BH_triangularmesh import (
    get_disjoint_triangles_subsets,
)
from magpylib._src.fields.field_BH_triangularmesh import magnet_trimesh_field
from magpylib._src.fields.field_BH_triangularmesh import trimesh_is_closed
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet
from magpylib._src.obj_classes.class_misc_Triangle import Triangle
from magpylib._src.style import TriangularMeshStyle


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

    reorient_triangles: bool, optional
        If `False`, no facet orientation check is performed. If `True`, facets pointing inwards are
        fliped in the right direction.

    validate_closed: bool, optional
        If `True`, the provided set of facets is validated by checking if it forms a closed body.
        Can be deactivated for perfomance reasons by setting it to `False`.

    validate_connected: bool, optional
        If `True`, the provided set of facets is validated by checking if it forms a connected body.
        Can be deactivated for perfomance reasons by setting it to `False`.

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
    _style_class = TriangularMeshStyle

    def __init__(
        self,
        magnetization=None,
        vertices=None,
        triangles=None,
        position=(0, 0, 0),
        orientation=None,
        validate_closed=True,
        validate_connected=True,
        reorient_triangles=True,
        style=None,
        **kwargs,
    ):

        self._vertices, self._triangles = self._input_check(vertices, triangles)
        self._is_connected = None
        self._is_closed = None
        self._is_reoriented = False
        self._triangles_subsets = None

        if validate_closed:
            self._validate_closed()

        if validate_connected:
            self._validate_connected()

        if reorient_triangles and self.is_closed:
            # perform only if closed, or inside-outside will fail
            self.reorient_triangles()

        # inherit
        super().__init__(position, orientation, magnetization, style, **kwargs)

    # property getters and setters
    @property
    def vertices(self):
        """Mesh vertices objects"""
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
    def is_closed(self):
        """Is-closed boolean check"""
        if self._is_closed is None:
            self._is_closed = trimesh_is_closed(self._triangles)
        return self._is_closed

    @property
    def is_reoriented(self):
        """Tells if the triangles have been reoriented"""
        return self._is_reoriented

    @property
    def is_connected(self):
        """Is-connected boolean check"""
        if self._is_connected is None:
            self._is_connected = len(self.triangles_subsets) == 1
        return self._is_connected

    @property
    def triangles_subsets(self):
        """return triangles subsets"""
        if self._triangles_subsets is None:
            self._triangles_subsets = get_disjoint_triangles_subsets(self._triangles)
        return self._triangles_subsets

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

    def _input_check(self, vertices, triangles):
        """input checks here ?"""

        # no. vertices must exceed largest triangle index
        # not all vertices can lie in a plane
        # unique vertices ?
        # do validation checks
        verts = check_format_input_vector(
            vertices,
            dims=(2,),
            shape_m1=3,
            sig_name="TriangularMesh.vertices",
            sig_type="array_like (list, tuple, ndarray) of shape (n,3)",
        )

        # check if triangle indices have allowed values
        # triangles must not be duplicates
        # triangles must not be degenerate ()
        # do validation checks
        trias = check_format_input_vector(
            triangles,
            dims=(2,),
            shape_m1=3,
            sig_name="TriangularMesh.triangles",
            sig_type="array_like (list, tuple, ndarray) of shape (n,3)",
        ).astype(int)

        return (verts, trias)

    def _validate_connected(self):
        """
        Check if trimesh consists of multiple disconnecetd parts.
        Raise error if this is the case.
        """
        if not self.is_connected:
            raise ValueError(
                "Bad `triangles` input of TriangularMesh. "
                "Resulting mesh is not connected. "
                "Disable error by setting `validate_connected=False`."
            )

    def _validate_closed(self):
        """
        Check if input mesh is closed
        """
        if not self.is_closed:
            raise ValueError(
                "Bad `triangles` input of TriangularMesh. "
                "Resulting mesh is not closed. "
                "Disable error by setting `validate_closed=False`."
            )

    def reorient_triangles(self):
        """Triangular facets pointing inwards are fliped in the right direction.
        Prior to reorientation, it is checked if the mesh is closed.
        """
        _ = self.is_closed  # perform isclosed check through getter
        self._triangles = fix_trimesh_orientation(self._vertices, self._triangles)
        self._is_reoriented = True

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

        Notes
        -----
        Facets are automatically reoriented since `scipy.spatial.ConvexHull` objects do not
        guarantee that the facets are all pointing outwards. A mesh validation is also performed.

        Returns
        -------
        magnet source: `TriangularMesh` object

        Examples
        --------
        """

        # reorient facets since ConvexHull does not guarantee that the facets are all
        # pointing outwards
        reorient_triangles = kwargs.pop("reorient_triangles", True)
        validate_closed = kwargs.pop("validate_closed", True)
        validate_connected = kwargs.pop("validate_connected", True)
        return cls(
            magnetization=magnetization,
            vertices=points,
            triangles=ConvexHull(points).simplices,
            position=position,
            orientation=orientation,
            reorient_triangles=reorient_triangles,
            validate_closed=validate_closed,
            validate_connected=validate_connected,
            style=style,
            **kwargs,
        )

    @classmethod
    def from_pyvista(
        cls,
        magnetization=None,
        polydata=None,
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

        polydata: pyvista.core.pointset.PolyData object
            A valid pyvista Polydata mesh object. (e.g. `pyvista.Sphere()`)

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

        Notes
        -----
        Facets are automatically reoriented since `pyvista.core.pointset.PolyData` objects do not
        guarantee that the facets are all pointing outwards. A mesh validation is also performed.

        Returns
        -------
        magnet source: `TriangularMesh` object

        Examples
        --------
        """
        # pylint: disable=import-outside-toplevel
        try:
            import pyvista
        except ImportError as missing_module:  # pragma: no cover
            raise ModuleNotFoundError(
                """In order load pyvista Polydata objects, you first need to install pyvista via pip
                or conda, see https://docs.pyvista.org/getting-started/installation.html"""
            ) from missing_module
        if not isinstance(polydata, pyvista.core.pointset.PolyData):
            raise TypeError(
                "The `polydata` parameter must be an instance of `pyvista.core.pointset.PolyData`, "
                f"received {polydata!r} instead"
            )
        polydata = polydata.triangulate()
        vertices = polydata.points
        triangles = polydata.faces.reshape(-1, 4)[:, 1:]
        reorient_triangles = kwargs.pop("reorient_triangles", True)
        validate_closed = kwargs.pop("validate_closed", True)
        validate_connected = kwargs.pop("validate_connected", True)
        return cls(
            magnetization=magnetization,
            vertices=vertices,
            triangles=triangles,
            position=position,
            orientation=orientation,
            reorient_triangles=reorient_triangles,
            validate_closed=validate_closed,
            validate_connected=validate_connected,
            style=style,
            **kwargs,
        )

    @classmethod
    def from_triangular_facets(
        cls,
        magnetization=None,
        facets=None,
        position=(0, 0, 0),
        orientation=None,
        reorient_triangles=True,
        validate_closed=True,
        validate_connected=True,
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

        facets: array_like
            An array_like of valid triangular facets objects. Elements can be one of
                - `magpylib.misc.Triangle` (only vertices are taken, magnetization is ignored)
                - array-like object of shape (3,3)


        position: array_like, shape (3,) or (m,3)
            Object position(s) in the global coordinates in units of [mm]. For m>1, the
            `position` and `orientation` attributes together represent an object path.
            When setting facets, the initial position is set to the barycenter.

        orientation: scipy `Rotation` object with length 1 or m, default=`None`
            Object orientation(s) in the global coordinates. `None` corresponds to
            a unit-rotation. For m>1, the `position` and `orientation` attributes
            together represent an object path.

        reorient_triangles: bool, optional
            If `False`, no facet orientation check is performed. If `True`, facets pointing inwards
            are fliped in the right direction.

        validate_closed: bool, optional
            If `True`, the provided set of facets is validated by checking if it forms a closed
            body. Can be deactivated for perfomance reasons by setting it to `False`.

        validate_connected: bool, optional
            If `True`, the provided set of facets is validated by checking if it forms a connected
            body. Can be deactivated for perfomance reasons by setting it to `False`.

        parent: `Collection` object or `None`
            The object is a child of it's parent collection.

        style: dict
            Object style inputs must be in dictionary form, e.g. `{'color':'red'}` or
            using style underscore magic, e.g. `style_color='red'`.

        Notes
        -----
        Facets are automatically reoriented since `pyvista.core.pointset.PolyData` objects do not
        guarantee that the facets are all pointing outwards. A mesh validation is also performed.

        Returns
        -------
        magnet source: `TriangularMesh` object

        Examples
        --------
        """
        if not isinstance(facets, (np.ndarray, list, tuple)):
            raise TypeError(
                "The `facets` parameter must be array-like, "
                f"\nreceived type {type(facets)} instead"
            )
        if isinstance(facets, np.ndarray):
            if not (facets.ndim == 3 and facets.shape[-2:] == (3, 3)):
                raise ValueError(
                    "The `facets` parameter must be array-like of shape (n,3,3), "
                    "or list like of `magpylib.misc.Triangle`and array-like object of shape (3,3)"
                    f"\nreceived array of shape {facets.shape} instead"
                )
        else:
            facet_list = []
            for facet in facets:
                if isinstance(facet, (np.ndarray, list, tuple)):
                    facet = np.array(facet)
                    if facet.shape != (3, 3):
                        raise ValueError(
                            "A facet object must be a (3,3) array-like or "
                            "a `magpylib.misc.Triangle object"
                            f"\nreceived array of shape {facet.shape} instead"
                        )
                elif isinstance(facet, Triangle):
                    facet = facet.vertices
                else:
                    raise TypeError(
                        "A facet object must be a (3,3) array-like or "
                        "a `magpylib.misc.Triangle object"
                        f"\nreceived type {type(facet)} instead"
                    )
                facet_list.append(facet)
            facets = np.array(facet_list).astype(float)

        vertices, tr = np.unique(facets.reshape((-1, 3)), axis=0, return_inverse=True)
        triangles = tr.reshape((-1, 3))

        return cls(
            magnetization=magnetization,
            vertices=vertices,
            triangles=triangles,
            position=position,
            orientation=orientation,
            reorient_triangles=reorient_triangles,
            validate_closed=validate_closed,
            validate_connected=validate_connected,
            style=style,
            **kwargs,
        )

"""Magnet TriangularMesh class code"""
import numpy as np
from scipy.spatial import ConvexHull  # pylint: disable=no-name-in-module

from magpylib._src.display.traces_generic import make_TriangularMesh
from magpylib._src.exceptions import MagpylibMissingInput
from magpylib._src.fields.field_BH_triangularmesh import calculate_centroid
from magpylib._src.fields.field_BH_triangularmesh import fix_trimesh_orientation
from magpylib._src.fields.field_BH_triangularmesh import (
    get_disjoint_triangles_subsets,
)
from magpylib._src.fields.field_BH_triangularmesh import magnet_trimesh_field
from magpylib._src.fields.field_BH_triangularmesh import trimesh_is_closed
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet
from magpylib._src.obj_classes.class_Collection import Collection
from magpylib._src.obj_classes.class_misc_Triangle import Triangle
from magpylib._src.style import TriangularMeshStyle


class TriangularMesh(BaseMagnet):
    """Magnet with homogeneous magnetization defined by triangular surface mesh.
    Can be used as `sources` input for magnetic field computation.
    When `position=(0,0,0)` and `orientation=None` the TriangularMesh vertices
    are the same as in the global coordinate system.

    Parameters
    ----------
    magnetization: array_like, shape (3,), default=`None`
        Magnetization vector (mu0*M, remanence field) in units of [mT] given in
        the local object coordinates (rotates with object).

    vertices: ndarray, shape (n,3)
        A set of points in units of [mm] in the local object coordinates from which the
        triangles of the mesh are constructed by the additional `triangles`input.

    triangles: ndarray, shape (n,3)
        Indices of vertices. Each triplet represents one triangle of the mesh.

    position: array_like, shape (3,) or (m,3), default=`(0,0,0)`
        Object position(s) in the global coordinates in units of [mm]. For m>1, the
        `position` and `orientation` attributes together represent an object path.

    orientation: scipy `Rotation` object with length 1 or m, default=`None`
        Object orientation(s) in the global coordinates. `None` corresponds to
        a unit-rotation. For m>1, the `position` and `orientation` attributes
        together represent an object path.

    reorient_triangles: bool, default=`True`
        In a properly oriented mesh, all facets must be oriented outwards.
        If `True`, check and fix the orientation of each triangle.

    validate_closed: bool, default=`True`
        Only a closed mesh guarantees a physical magnet.
        If `True`, raise error if mesh is not closed.

    validate_connected: bool, default=`True`
        If `True` raise an error if mesh is not connected.

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
    We compute the B-field in units of [mT] of a triangular mesh (4 vertices, 4 triangles)
    with magnetization (100,200,300) in units of [mT] at the observer position
    (1,1,1) given in units of [mm]:

    >>> import magpylib as magpy
    >>> vv = ((0,0,0), (1,0,0), (0,1,0), (0,0,1))
    >>> tt = ((0,1,2), (0,1,3), (0,2,3), (1,2,3))
    >>> trim = magpy.magnet.TriangularMesh(magnetization=(100,200,300), vertices=vv, triangles=tt)
    >>> print(trim.getB((1,1,1)))
    [2.60236696 2.08189357 1.56142018]
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
        return self._get_barycenter(
            self._position, self._orientation, self._vertices, self._triangles
        )

    @property
    def barycenter(self):
        """Object barycenter."""
        return np.squeeze(self._barycenter)

    @staticmethod
    def _get_barycenter(position, orientation, vertices, triangles):
        """Returns the barycenter of a tetrahedron."""
        centroid = calculate_centroid(vertices, triangles)
        barycenter = orientation.apply(centroid) + position
        return barycenter

    def _input_check(self, vertices, triangles):
        """input checks here ?"""
        # no. vertices must exceed largest triangle index
        # not all vertices can lie in a plane
        # unique vertices ?
        # do validation checks
        if vertices is None:
            raise MagpylibMissingInput(f"Parameter `vertices` of {self} must be set.")
        if triangles is None:
            raise MagpylibMissingInput(f"Parameter `triangles` of {self} must be set.")
        verts = check_format_input_vector(
            vertices,
            dims=(2,),
            shape_m1=3,
            sig_name="TriangularMesh.vertices",
            sig_type="array_like (list, tuple, ndarray) of shape (n,3)",
        )
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

    def to_TrianglesCollection(self):
        """Return a Collection of Triangles objects from the current TriangularMesh"""
        tris = [
            Triangle(magnetization=self.magnetization, vertices=v) for v in self.facets
        ]
        coll = Collection(tris)
        coll.position = self.position
        coll.orientation = self.orientation
        coll.style.update(self.style.as_dict(), _match_properties=False)
        return coll

    @classmethod
    def from_ConvexHull(
        cls,
        magnetization=None,
        points=None,
        position=(0, 0, 0),
        orientation=None,
        validate_closed=True,
        validate_connected=True,
        reorient_triangles=True,
        style=None,
        **kwargs,
    ):
        """Create a TriangularMesh magnet from a point cloud via its convex hull.

        Parameters
        ----------
        magnetization: array_like, shape (3,), default=`None`
            Magnetization vector (mu0*M, remanence field) in units of [mT] given in
            the local object coordinates (rotates with object).

        points: ndarray, shape (n,3)
            Point cloud from which the convex hull is computed.

        position: array_like, shape (3,) or (m,3)
            Object position(s) in the global coordinates in units of [mm]. For m>1, the
            `position` and `orientation` attributes together represent an object path.
            When setting facets, the initial position is set to the barycenter.

        orientation: scipy `Rotation` object with length 1 or m, default=`None`
            Object orientation(s) in the global coordinates. `None` corresponds to
            a unit-rotation. For m>1, the `position` and `orientation` attributes
            together represent an object path.

        reorient_triangles: bool, default=`True`
            In a properly oriented mesh, all facets must be oriented outwards.
            If `True`, check and fix the orientation of each triangle.

        validate_closed: bool, default=`True`
            Only a closed mesh guarantees a physical magnet.
            If `True`, raise error if mesh is not closed.

        validate_connected: bool, default=`True`
            If `True` raise an error if mesh is not connected.

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
        validate_closed=True,
        validate_connected=True,
        reorient_triangles=True,
        style=None,
        **kwargs,
    ):
        """Create a TriangularMesh magnet from a pyvista PolyData mesh object.

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

        reorient_triangles: bool, default=`True`
            In a properly oriented mesh, all facets must be oriented outwards.
            If `True`, check and fix the orientation of each triangle.

        validate_closed: bool, default=`True`
            Only a closed mesh guarantees a physical magnet.
            If `True`, raise error if mesh is not closed.

        validate_connected: bool, default=`True`
            If `True` raise an error if mesh is not connected.

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
    def from_triangles(
        cls,
        magnetization=None,
        triangles=None,
        position=(0, 0, 0),
        orientation=None,
        reorient_triangles=True,
        validate_closed=True,
        validate_connected=True,
        style=None,
        **kwargs,
    ):
        """Create a TriangularMesh magnet from a set of triangles.

        Parameters
        ----------
        magnetization: array_like, shape (3,), default=`None`
            Magnetization vector (mu0*M, remanence field) in units of [mT] given in
            the local object coordinates (rotates with object).

        triangles: array_like
            An array_like of valid triangles. Elements can be one of
                - `magpylib.misc.Triangle` (only vertices are taken, magnetization is ignored)
                - triangle vertices of shape (3,3)

        position: array_like, shape (3,) or (m,3)
            Object position(s) in the global coordinates in units of [mm]. For m>1, the
            `position` and `orientation` attributes together represent an object path.
            When setting facets, the initial position is set to the barycenter.

        orientation: scipy `Rotation` object with length 1 or m, default=`None`
            Object orientation(s) in the global coordinates. `None` corresponds to
            a unit-rotation. For m>1, the `position` and `orientation` attributes
            together represent an object path.

        reorient_triangles: bool, default=`True`
            In a properly oriented mesh, all facets must be oriented outwards.
            If `True`, check and fix the orientation of each triangle.

        validate_closed: bool, default=`True`
            Only a closed mesh guarantees a physical magnet.
            If `True`, raise error if mesh is not closed.

        validate_connected: bool, default=`True`
            If `True` raise an error if mesh is not connected.

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

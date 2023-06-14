"""Magnet TriangularMesh class code"""
import warnings

import numpy as np
from scipy.spatial import ConvexHull  # pylint: disable=no-name-in-module

from magpylib._src.display.traces_generic import make_TriangularMesh
from magpylib._src.exceptions import MagpylibMissingInput
from magpylib._src.fields.field_BH_triangularmesh import calculate_centroid
from magpylib._src.fields.field_BH_triangularmesh import fix_trimesh_orientation
from magpylib._src.fields.field_BH_triangularmesh import (
    get_disjoint_faces_subsets,
)
from magpylib._src.fields.field_BH_triangularmesh import magnet_trimesh_field
from magpylib._src.fields.field_BH_triangularmesh import trimesh_is_closed
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.input_checks import check_format_input_vector2
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
        triangular faces of the mesh are constructed by the additional `faces`input.

    faces: ndarray, shape (n,3)
        Indices of vertices. Each triplet represents one triangle of the mesh.

    position: array_like, shape (3,) or (m,3), default=`(0,0,0)`
        Object position(s) in the global coordinates in units of [mm]. For m>1, the
        `position` and `orientation` attributes together represent an object path.

    orientation: scipy `Rotation` object with length 1 or m, default=`None`
        Object orientation(s) in the global coordinates. `None` corresponds to
        a unit-rotation. For m>1, the `position` and `orientation` attributes
        together represent an object path.

    reorient_faces: bool, default=`True`
        In a properly oriented mesh, all faces must be oriented outwards.
        If `True`, check and fix the orientation of each triangle.

    check_closed: bool, default=`True`
        Only a closed mesh guarantees a physical magnet.
        If `True`, raise error if mesh is not closed.

    check_connected: bool, default=`True`
        If `True` raise an error if mesh is not connected.

    parent: `Collection` object or `None`
        The object is a child of it's parent collection.

    style: dict
        Object style inputs must be in dictionary form, e.g. `{'color':'red'}` or
        using style underscore magic, e.g. `style_color='red'`.

    Notes
    -----
    Faces are automatically reoriented since `scipy.spatial.ConvexHull` objects do not
    guarantee that the faces are all pointing outwards. A mesh validation is also performed.

    Returns
    -------
    magnet source: `TriangularMesh` object

    Examples
    --------
    We compute the B-field in units of [mT] of a triangular mesh (4 vertices, 4 faces)
    with magnetization (100,200,300) in units of [mT] at the observer position
    (1,1,1) given in units of [mm]:

    >>> import magpylib as magpy
    >>> vv = ((0,0,0), (1,0,0), (0,1,0), (0,0,1))
    >>> tt = ((0,1,2), (0,1,3), (0,2,3), (1,2,3))
    >>> trim = magpy.magnet.TriangularMesh(magnetization=(100,200,300), vertices=vv, faces=tt)
    >>> print(trim.getB((1,1,1)))
    [2.60236696 2.08189357 1.56142018]
    """

    _field_func = staticmethod(magnet_trimesh_field)
    _field_func_kwargs_ndim = {"magnetization": 2, "mesh": 3}
    _draw_func = make_TriangularMesh
    _style_class = TriangularMeshStyle

    def __init__(
        self,
        magnetization=None,
        vertices=None,
        faces=None,
        position=(0, 0, 0),
        orientation=None,
        check_closed="warn",
        check_connected="warn",
        reorient_faces="warn",
        style=None,
        **kwargs,
    ):
        self._vertices, self._faces = self._input_check(vertices, faces)
        self._status_connected = None
        self._status_closed = None
        self._status_reoriented = False
        self._status_connected_data = None

        self.check_closed(mode=check_closed)
        self.check_connected(mode=check_connected)
        self.reorient_faces(mode=reorient_faces)

        # inherit
        super().__init__(position, orientation, magnetization, style, **kwargs)

    # property getters and setters
    @property
    def vertices(self):
        """Mesh vertices"""
        return self._vertices

    @property
    def faces(self):
        """Mesh faces"""
        return self._faces

    @property
    def mesh(self):
        """Mesh"""
        return self._vertices[self._faces]

    @property
    def status_closed(self):
        """Return closed satus"""
        return self._status_closed

    @property
    def status_connected(self):
        """Return connected satus"""
        return self._status_connected

    @property
    def status_reoriented(self):
        """Return reoriented satus"""
        return self._status_reoriented

    @staticmethod
    def _validate_mode_arg(arg, arg_name="mode"):
        """Validate mode argument"""
        accepted_arg_vals = (True, False, "warn", "raise", "ignore", "skip")
        if arg not in accepted_arg_vals:
            raise ValueError(
                f"The `{arg_name}` argument must be one of {accepted_arg_vals}, "
                f"instead received {arg!r}."
                f"\nNote that `True` translates to `'warn'` and `False` to `'skip'`"
            )
        arg = "warn" if arg is True else "skip" if arg is False else arg
        return arg

    def check_closed(self, mode="warn"):
        """
        Check whether the mesh is closed.

        This function checks if the mesh is closed. If the mesh is not closed,
        it issues a warning or raises a ValueError, depending on the 'mode' parameter.
        If 'mode' is set to 'ignore', it does not issue a warning or raise an error.

        Parameters
        ----------
        mode : str, optional
            Controls how to handle if the mesh is not closed.
            Accepted values are "warn", "raise", or "ignore".
            If "warn", a warning is issued. If "raise", a ValueError is raised.
            If "ignore", no action is taken. By default "warn".

        Returns
        -------
        bool
            True if the mesh is closed, False otherwise.

        Raises
        ------
        ValueError
            If 'mode' is not one of the accepted values or if 'mode' is "raise" and the mesh
            is not closed.

        Warns
        -----
        UserWarning
            If the mesh is not closed and 'mode' is "warn".
        """
        mode = self._validate_mode_arg(mode, arg_name="check_closed mode")
        if mode != "skip" and self._status_closed is None:
            self._status_closed = trimesh_is_closed(self._faces)
            if not self._status_closed:
                msg = (
                    f"{self!r} is open. Field calculation may deliver erroneous results. "
                    "This check can be disabled at initialization time by setting "
                    "`check_closed='skip'`. Open edges can be displayed by setting "
                    "the `style_mesh_disjoint_show=True` in the `show` function. "
                    "Note that faces reorientation may also fail in that case!"
                )
                if mode == "warn":
                    warnings.warn(msg)
                elif mode == "raise":
                    raise ValueError(msg)
        return self._status_closed

    def check_connected(self, mode="warn"):
        """Check whether the mesh is connected.

        This function checks if the mesh is connected. If the mesh is not connected,
        it issues a warning or raises a ValueError, depending on the 'mode' parameter.
        If 'mode' is set to 'ignore', it does not issue a warning or raise an error.

        Parameters
        ----------
        mode : str, optional
            Controls how to handle if the mesh is not connected.
            Accepted values are "warn", "raise", or "ignore".
            If "warn", a warning is issued. If "raise", a ValueError is raised.
            If "ignore", no action is taken. By default "warn".

        Returns
        -------
        bool
            True if the mesh is connected, False otherwise.

        Raises
        ------
        ValueError
            If 'mode' is not one of the accepted values or if 'mode' is "raise" and the mesh
            is not connected.

        Warns
        -----
        UserWarning
            If the mesh is not connected and 'mode' is "warn".
        """
        mode = self._validate_mode_arg(mode, arg_name="check_connected mode")
        if mode != "skip" and self._status_connected is None:
            self._status_connected = len(self.get_faces_subsets()) == 1
            if not self._status_connected:
                msg = (
                    f"{self!r} is disjoint. "
                    "Mesh parts can be obtained via the `.get_faces_subsets()` method "
                    "and be displayed by setting `style_mesh_disjoint_show=True` "
                    "in the `show` function. "
                    "This check can be disabled at initialization time by setting "
                    "`check_connected='skip'`. "
                    "Note that this should not affect field calculation."
                )
                if mode == "warn":
                    warnings.warn(msg)
                elif mode == "raise":
                    raise ValueError(msg)
        return self._status_connected

    def reorient_faces(self, mode="warn"):
        """Triangular faces pointing inwards are fliped in the right direction.
        Prior to reorientation, it is checked if the mesh is closed.
        """
        mode = self._validate_mode_arg(mode, arg_name="reorient_faces mode")
        if mode != "skip":
            if self._status_closed is None and mode == "warn":
                warnings.warn(
                    f"{self!r} status_closed is None (=unchecked). Now applying check_closed()"
                )
            self.check_closed(mode=mode)
            # if mesh is not closed, inside-outside will not fail but may deliver inconsistent results
            self._faces = fix_trimesh_orientation(self._vertices, self._faces)
            self._status_reoriented = True

    def get_faces_subsets(self):
        """return faces subsets"""
        if self._status_connected_data is None:
            self._status_connected_data = get_disjoint_faces_subsets(self._faces)
        return self._status_connected_data

    @property
    def _barycenter(self):
        """Object barycenter."""
        return self._get_barycenter(
            self._position, self._orientation, self._vertices, self._faces
        )

    @property
    def barycenter(self):
        """Object barycenter."""
        return np.squeeze(self._barycenter)

    @staticmethod
    def _get_barycenter(position, orientation, vertices, faces):
        """Returns the barycenter of a tetrahedron."""
        centroid = calculate_centroid(vertices, faces)
        barycenter = orientation.apply(centroid) + position
        return barycenter

    def _input_check(self, vertices, faces):
        """input checks here ?"""
        # no. vertices must exceed largest triangle index
        # not all vertices can lie in a plane
        # unique vertices ?
        # do validation checks
        if vertices is None:
            raise MagpylibMissingInput(f"Parameter `vertices` of {self} must be set.")
        if faces is None:
            raise MagpylibMissingInput(f"Parameter `faces` of {self} must be set.")
        verts = check_format_input_vector(
            vertices,
            dims=(2,),
            shape_m1=3,
            sig_name="TriangularMesh.vertices",
            sig_type="array_like (list, tuple, ndarray) of shape (n,3)",
        )
        trias = check_format_input_vector(
            faces,
            dims=(2,),
            shape_m1=3,
            sig_name="TriangularMesh.faces",
            sig_type="array_like (list, tuple, ndarray) of shape (n,3)",
        ).astype(int)
        try:
            verts[trias]
        except IndexError as e:
            raise IndexError(
                "Some `faces` indices do not match with `vertices` array"
            ) from e
        return verts, trias

    def to_TriangleCollection(self):
        """Return a Collection of Triangle objects from the current TriangularMesh"""
        tris = [
            Triangle(magnetization=self.magnetization, vertices=v) for v in self.mesh
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
        check_closed="warn",
        check_connected="warn",
        reorient_faces=True,
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

        orientation: scipy `Rotation` object with length 1 or m, default=`None`
            Object orientation(s) in the global coordinates. `None` corresponds to
            a unit-rotation. For m>1, the `position` and `orientation` attributes
            together represent an object path.

        reorient_faces: bool, default=`True`
            In a properly oriented mesh, all faces must be oriented outwards.
            If `True`, check and fix the orientation of each triangle.

        check_closed: {'warn', 'raise', 'ignore'}, default='warn'
            Only a closed mesh guarantees a physical magnet.
            If the mesh is open and "warn", a warning is issued.
            If the mesh is open and "raise", a ValueError is raised.
            If "ignore", no mesh check is perfomed.

        check_connected: {'warn', 'raise', 'ignore'}, default='warn'
            If the mesh is disjoint and "warn", a warning is issued.
            If the mesh is disjoint and "raise", a ValueError is raised.
            If "ignore", no mesh check is perfomed.

        parent: `Collection` object or `None`
            The object is a child of it's parent collection.

        style: dict
            Object style inputs must be in dictionary form, e.g. `{'color':'red'}` or
            using style underscore magic, e.g. `style_color='red'`.

        Notes
        -----
        Faces are automatically reoriented since `scipy.spatial.ConvexHull` objects do not
        guarantee that the faces are all pointing outwards. A mesh validation is also performed.

        Returns
        -------
        magnet source: `TriangularMesh` object

        Examples
        --------
        """
        return cls(
            magnetization=magnetization,
            vertices=points,
            faces=ConvexHull(points).simplices,
            position=position,
            orientation=orientation,
            reorient_faces=reorient_faces,
            check_closed=check_closed,
            check_connected=check_connected,
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
        check_closed="warn",
        check_connected="warn",
        reorient_faces=True,
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

        orientation: scipy `Rotation` object with length 1 or m, default=`None`
            Object orientation(s) in the global coordinates. `None` corresponds to
            a unit-rotation. For m>1, the `position` and `orientation` attributes
            together represent an object path.

        reorient_faces: bool, default=`True`
            In a properly oriented mesh, all faces must be oriented outwards.
            If `True`, check and fix the orientation of each triangle.

        check_closed: {'warn', 'raise', 'ignore'}, default='warn'
            Only a closed mesh guarantees a physical magnet.
            If the mesh is open and "warn", a warning is issued.
            If the mesh is open and "raise", a ValueError is raised.
            If "ignore", no mesh check is perfomed.

        check_connected: {'warn', 'raise', 'ignore'}, default='warn'
            If the mesh is disjoint and "warn", a warning is issued.
            If the mesh is disjoint and "raise", a ValueError is raised.
            If "ignore", no mesh check is perfomed.

        parent: `Collection` object or `None`
            The object is a child of it's parent collection.

        style: dict
            Object style inputs must be in dictionary form, e.g. `{'color':'red'}` or
            using style underscore magic, e.g. `style_color='red'`.

        Notes
        -----
        Faces are automatically reoriented since `pyvista.core.pointset.PolyData` objects do not
        guarantee that the faces are all pointing outwards. A mesh validation is also performed.

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
        faces = polydata.faces.reshape(-1, 4)[:, 1:]

        return cls(
            magnetization=magnetization,
            vertices=vertices,
            faces=faces,
            position=position,
            orientation=orientation,
            reorient_faces=reorient_faces,
            check_closed=check_closed,
            check_connected=check_connected,
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
        reorient_faces=True,
        check_closed="warn",
        check_connected="warn",
        style=None,
        **kwargs,
    ):
        """Create a TriangularMesh magnet from a list or Collection of Triangle objects.

        Parameters
        ----------
        magnetization: array_like, shape (3,), default=`None`
            Magnetization vector (mu0*M, remanence field) in units of [mT] given in
            the local object coordinates (rotates with object).

        triangles: list or Collection of Triangle objects
            Only vertices of Triangle objects are taken, magnetization is ignored.

        position: array_like, shape (3,) or (m,3)
            Object position(s) in the global coordinates in units of [mm]. For m>1, the
            `position` and `orientation` attributes together represent an object path.

        orientation: scipy `Rotation` object with length 1 or m, default=`None`
            Object orientation(s) in the global coordinates. `None` corresponds to
            a unit-rotation. For m>1, the `position` and `orientation` attributes
            together represent an object path.

        reorient_faces: bool, default=`True`
            In a properly oriented mesh, all faces must be oriented outwards.
            If `True`, check and fix the orientation of each triangle.

        check_closed: {'warn', 'raise', 'ignore'}, default='warn'
            Only a closed mesh guarantees a physical magnet.
            If the mesh is open and "warn", a warning is issued.
            If the mesh is open and "raise", a ValueError is raised.
            If "ignore", no mesh check is perfomed.

        check_connected: {'warn', 'raise', 'ignore'}, default='warn'
            If the mesh is disjoint and "warn", a warning is issued.
            If the mesh is disjoint and "raise", a ValueError is raised.
            If "ignore", no mesh check is perfomed.

        parent: `Collection` object or `None`
            The object is a child of it's parent collection.

        style: dict
            Object style inputs must be in dictionary form, e.g. `{'color':'red'}` or
            using style underscore magic, e.g. `style_color='red'`.

        Notes
        -----
        Faces are automatically reoriented since `pyvista.core.pointset.PolyData` objects do not
        guarantee that the faces are all pointing outwards. A mesh validation is also performed.

        Returns
        -------
        magnet source: `TriangularMesh` object

        Examples
        --------
        """
        if not isinstance(triangles, (list, Collection)):
            raise TypeError(
                "The `triangles` parameter must be a list or Collection of `Triangle` objects, "
                f"\nreceived type {type(triangles)} instead"
            )
        for obj in triangles:
            if not isinstance(obj, Triangle):
                raise TypeError(
                    "All elements of `triangles` must be `Triangle` objects, "
                    f"\nreceived type {type(obj)} instead"
                )
        mesh = np.array([tria.vertices for tria in triangles])
        vertices, tr = np.unique(mesh.reshape((-1, 3)), axis=0, return_inverse=True)
        faces = tr.reshape((-1, 3))

        return cls(
            magnetization=magnetization,
            vertices=vertices,
            faces=faces,
            position=position,
            orientation=orientation,
            reorient_faces=reorient_faces,
            check_closed=check_closed,
            check_connected=check_connected,
            style=style,
            **kwargs,
        )

    @classmethod
    def from_mesh(
        cls,
        magnetization=None,
        mesh=None,
        position=(0, 0, 0),
        orientation=None,
        reorient_faces=True,
        check_closed="warn",
        check_connected="warn",
        style=None,
        **kwargs,
    ):
        """Create a TriangularMesh magnet from a mesh input.

        Parameters
        ----------
        magnetization: array_like, shape (3,), default=`None`
            Magnetization vector (mu0*M, remanence field) in units of [mT] given in
            the local object coordinates (rotates with object).

        mesh: array_like, shape (n,3,3)
            An array_like of triangular faces that make up a triangular mesh.

        position: array_like, shape (3,) or (m,3)
            Object position(s) in the global coordinates in units of [mm]. For m>1, the
            `position` and `orientation` attributes together represent an object path.

        orientation: scipy `Rotation` object with length 1 or m, default=`None`
            Object orientation(s) in the global coordinates. `None` corresponds to
            a unit-rotation. For m>1, the `position` and `orientation` attributes
            together represent an object path.

        reorient_faces: bool, default=`True`
            In a properly oriented mesh, all faces must be oriented outwards.
            If `True`, check and fix the orientation of each triangle.

        check_closed: {'warn', 'raise', 'ignore'}, default='warn'
            Only a closed mesh guarantees a physical magnet.
            If the mesh is open and "warn", a warning is issued.
            If the mesh is open and "raise", a ValueError is raised.
            If "ignore", no mesh check is perfomed.

        check_connected: {'warn', 'raise', 'ignore'}, default='warn'
            If the mesh is disjoint and "warn", a warning is issued.
            If the mesh is disjoint and "raise", a ValueError is raised.
            If "ignore", no mesh check is perfomed.

        parent: `Collection` object or `None`
            The object is a child of it's parent collection.

        style: dict
            Object style inputs must be in dictionary form, e.g. `{'color':'red'}` or
            using style underscore magic, e.g. `style_color='red'`.

        Notes
        -----
        Faces are automatically reoriented since `pyvista.core.pointset.PolyData` objects do not
        guarantee that the faces are all pointing outwards. A mesh validation is also performed.

        Returns
        -------
        magnet source: `TriangularMesh` object

        Examples
        --------
        """
        mesh = check_format_input_vector2(
            mesh,
            shape=[None, 3, 3],
            param_name="mesh",
        )
        vertices, tr = np.unique(mesh.reshape((-1, 3)), axis=0, return_inverse=True)
        faces = tr.reshape((-1, 3))

        return cls(
            magnetization=magnetization,
            vertices=vertices,
            faces=faces,
            position=position,
            orientation=orientation,
            reorient_faces=reorient_faces,
            check_closed=check_closed,
            check_connected=check_connected,
            style=style,
            **kwargs,
        )

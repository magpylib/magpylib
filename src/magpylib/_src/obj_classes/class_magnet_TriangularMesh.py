"""Magnet TriangularMesh class code"""

import warnings
from typing import ClassVar

import numpy as np
from scipy.spatial import ConvexHull  # pylint: disable=no-name-in-module

from magpylib._src.display.traces_core import make_TriangularMesh
from magpylib._src.exceptions import MagpylibMissingInput
from magpylib._src.fields.field_BH_triangularmesh import (
    BHJM_magnet_trimesh,
    calculate_centroid,
    fix_trimesh_orientation,
    get_disconnected_faces_subsets,
    get_intersecting_triangles,
    get_open_edges,
)
from magpylib._src.input_checks import (
    check_format_input_vector,
    check_format_input_vector2,
)
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet
from magpylib._src.obj_classes.class_Collection import Collection
from magpylib._src.obj_classes.class_misc_Triangle import Triangle
from magpylib._src.style import TriangularMeshStyle

# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=too-many-positional-arguments


class TriangularMesh(BaseMagnet):
    """Magnet with homogeneous magnetization defined by triangular surface mesh.
    Can be used as `sources` input for magnetic field computation.

    When `position=(0,0,0)` and `orientation=None` the TriangularMesh vertices
    are the same as in the global coordinate system.

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
        A set of points in units of m in the local object coordinates from which the
        triangular faces of the mesh are constructed by the additional `faces` input.

    faces: array_like, shape (n,3), default=`None`
        Indices of vertices. Each triplet represents one triangle of the mesh.

    polarization: array_like, shape (3,), default=`None`
        Magnetic polarization vector J = mu0*M in units of T,
        given in the local object coordinates (rotates with object).

    magnetization: array_like, shape (3,), default=`None`
        Magnetization vector M = J/mu0 in units of A/m,
        given in the local object coordinates (rotates with object).

    volume: float
        Read-only. Object physical volume in units of m^3.

    centroid: np.ndarray, shape (3,) or (m,3)
        Read-only. Object centroid in units of m.

    reorient_faces: bool or string, default=`True`
        In a properly oriented mesh, all faces must be oriented outwards.
        If `True`, check and fix the orientation of each triangle.
        Options are `'skip'`(=`False`), `'warn'`(=`True`), `'raise'`, `'ignore'`.

    check_open: bool or string, default=`True`
        Only a closed mesh guarantees correct B-field computation.
        If `True`, check if mesh is open.
        Options are `'skip'`(=`False`), `'warn'`(=`True`), `'raise'`, `'ignore'`.

    check_disconnected: bool or string, default=`True`
        Individual magnets should be connected bodies to avoid confusion.
        If `True`, check if mesh is disconnected.
        Options are `'skip'`(=`False`), `'warn'`(=`True`), `'raise'`, `'ignore'`.

    check_selfintersecting: bool or string, default=`True`
        a proper body cannot have a self-intersecting mesh.
        If `True`, check if mesh is self-intersecting.
        Options are `'skip'`(=`False`), `'warn'`(=`True`), `'raise'`, `'ignore'`.

    check_selfintersecting: bool, optional
        If `True`, the provided set of facets is validated by checking if the body is not
        self-intersecting. Can be deactivated for performance reasons by setting it to `False`.

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
    We compute the B-field in units of T of a triangular mesh (4 vertices, 4 faces)
    with polarization (0.1,0.2,0.3) in units of T at the observer position
    (0.01,0.01,0.01) given in units of m:

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> vv = ((0,0,0), (.01,0,0), (0,.01,0), (0,0,.01))
    >>> tt = ((0,1,2), (0,1,3), (0,2,3), (1,2,3))
    >>> trim = magpy.magnet.TriangularMesh(polarization=(.1,.2,.3), vertices=vv, faces=tt)
    >>> with np.printoptions(precision=3):
    ...     print(trim.getB((.01,.01,.01))*1000)
    [2.602 2.082 1.561]
    """

    _field_func = staticmethod(BHJM_magnet_trimesh)
    _field_func_kwargs_ndim: ClassVar[dict[str, int]] = {"polarization": 2, "mesh": 3}
    get_trace = make_TriangularMesh
    _style_class = TriangularMeshStyle

    def __init__(
        self,
        position=(0, 0, 0),
        orientation=None,
        vertices=None,
        faces=None,
        polarization=None,
        magnetization=None,
        check_open="warn",
        check_disconnected="warn",
        check_selfintersecting="warn",
        reorient_faces="warn",
        style=None,
        **kwargs,
    ):
        self._vertices, self._faces = self._input_check(vertices, faces)
        self._status_disconnected = None
        self._status_open = None
        self._status_reoriented = False
        self._status_selfintersecting = None
        self._status_disconnected_data = None
        self._status_open_data = None
        self._status_selfintersecting_data = None

        self.check_open(mode=check_open)
        self.check_disconnected(mode=check_disconnected)
        self.reorient_faces(mode=reorient_faces)
        self.check_selfintersecting(mode=check_selfintersecting)

        # inherit
        super().__init__(
            position, orientation, magnetization, polarization, style, **kwargs
        )

    # Properties
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
    def status_open(self):
        """Return open status"""
        return self._status_open

    @property
    def status_disconnected(self):
        """Return disconnected status"""
        return self._status_disconnected

    @property
    def status_reoriented(self):
        """Return reoriented status"""
        return self._status_reoriented

    # Methods
    def _get_volume(self):
        """Volume of object in units of m³.

        Based on algorithm from: https://n-e-r-v-o-u-s.com/blog/?p=4415
        For each triangle, compute the signed volume of tetrahedron from origin
        to triangle using: V = (1/6) * (v1 x v2) . v3
        Sum all signed volumes to get total mesh volume.

        Returns
        -------
        float
            Volume in units of m³.
        """
        if self._vertices is None or self._faces is None:
            return 0.0
        if self.status_open is None:
            self.check_open()
        if self.status_open is True:
            msg = "Open mesh detected. Cannot compute volume."
            raise ValueError(msg)

        # Vectorized calculation: get all triangle vertices at once
        # Shape: (n_faces, 3, 3) where each face has 3 vertices with 3 coordinates
        triangles = self.mesh

        # Extract vertex arrays: v1, v2, v3 for all triangles
        # Shape: (n_faces, 3) for each vertex array
        v1 = triangles[:, 0]  # First vertex of each triangle
        v2 = triangles[:, 1]  # Second vertex of each triangle
        v3 = triangles[:, 2]  # Third vertex of each triangle

        # Vectorized cross product: v1 x v2 for all triangles
        # Shape: (n_faces, 3)
        cross_products = np.cross(v1, v2)

        # Vectorized dot product: (v1 x v2) . v3 for all triangles
        # Shape: (n_faces,)
        dot_products = np.sum(cross_products * v3, axis=1)

        # Calculate signed volumes and sum them
        signed_volumes = dot_products / 6.0
        total_volume = np.sum(signed_volumes)

        # Return absolute value to get positive volume
        return abs(total_volume)

    def _get_centroid(self):
        """Centroid of object in units of m."""
        return self.barycenter

    @staticmethod
    def _validate_mode_arg(arg, arg_name="mode"):
        """Validate mode argument"""
        accepted_arg_vals = (True, False, "warn", "raise", "ignore", "skip")
        if arg not in accepted_arg_vals:
            msg = (
                f"The `{arg_name}` argument must be one of {accepted_arg_vals}, "
                f"instead received {arg!r}."
                f"\nNote that `True` translates to `'warn'` and `False` to `'skip'`"
            )
            raise ValueError(msg)
        return "warn" if arg is True else "skip" if arg is False else arg

    def check_open(self, mode="warn"):
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
            True if the mesh is open, False otherwise.

        Raises
        ------
        ValueError
            If 'mode' is not one of the accepted values or if 'mode' is "raise" and the mesh
            is open.

        Warns
        -----
        UserWarning
            If the mesh is open and 'mode' is "warn".
        """
        mode = self._validate_mode_arg(mode, arg_name="check_open mode")
        if mode != "skip" and self._status_open is None:
            self._status_open = len(self.get_open_edges()) > 0
            if self._status_open:
                msg = (
                    f"Open mesh detected in {self!r}. Intrinsic inside-outside checks may "
                    "give bad results and subsequently getB() and reorient_faces() may give bad "
                    "results as well. "
                    "This check can be disabled at initialization with check_open='skip'. "
                    "Open edges can be displayed in show() with style_mesh_open_show=True."
                    "Open edges are stored in the status_open_data property."
                )
                if mode == "warn":
                    warnings.warn(msg, stacklevel=2)
                elif mode == "raise":
                    raise ValueError(msg)
        return self._status_open

    def check_disconnected(self, mode="warn"):
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
            True if the mesh is disconnected, False otherwise.

        Raises
        ------
        ValueError
            If 'mode' is not one of the accepted values or if 'mode' is "raise" and the mesh
            is disconnected.

        Warns
        -----
        UserWarning
            If the mesh is disconnected and 'mode' is "warn".
        """
        mode = self._validate_mode_arg(mode, arg_name="check_disconnected mode")
        if mode != "skip" and self._status_disconnected is None:
            self._status_disconnected = len(self.get_faces_subsets()) > 1
            if self._status_disconnected:
                msg = (
                    f"Disconnected mesh detected in {self!r}. Magnet consists of multiple "
                    "individual parts. "
                    "This check can be disabled at initialization with check_disconnected='skip'. "
                    "Parts can be displayed in show() with style_mesh_disconnected_show=True. "
                    "Parts are stored in the status_disconnected_data property."
                )
                if mode == "warn":
                    warnings.warn(msg, stacklevel=2)
                elif mode == "raise":
                    raise ValueError(msg)
        return self._status_disconnected

    def check_selfintersecting(self, mode="warn"):
        """Check whether the mesh is self-intersecting.

        This function checks if the mesh is self-intersecting. If the mesh is self-intersecting,
        it issues a warning or raises a ValueError, depending on the 'mode' parameter.
        If 'mode' is set to 'ignore', it does not issue a warning or raise an error.

        Parameters
        ----------
        mode : str, optional
            Controls how to handle if the mesh is self-intersecting.
            Accepted values are "warn", "raise", or "ignore".
            If "warn", a warning is issued. If "raise", a ValueError is raised.
            If "ignore", no action is taken. By default "warn".

        Returns
        -------
        bool
            True if the mesh is self-intersecting, False otherwise.

        Raises
        ------
        ValueError
            If 'mode' is not one of the accepted values or if 'mode' is "raise" and the mesh
            is self-intersecting.

        Warns
        -----
        UserWarning
            If the mesh is self-intersecting and 'mode' is "warn".
        """
        mode = self._validate_mode_arg(mode, arg_name="check_selfintersecting mode")
        if mode != "skip" and self._status_selfintersecting is None:
            self._status_selfintersecting = len(self.get_selfintersecting_faces()) > 1
            if self._status_selfintersecting:
                msg = (
                    f"Self-intersecting mesh detected in {self!r}. "
                    "This check can be disabled at initialization with "
                    "check_selfintersecting='skip'. "
                    "Intersecting faces can be display in show() with "
                    "style_mesh_selfintersecting_show=True. "
                    "Parts are stored in the status_selfintersecting_data property."
                )
                if mode == "warn":
                    warnings.warn(msg, stacklevel=2)
                elif mode == "raise":
                    raise ValueError(msg)
        return self._status_selfintersecting

    def reorient_faces(self, mode="warn"):
        """Correctly reorients the mesh's faces.

        In a properly oriented mesh, all faces must be oriented outwards. This function
        fixes the orientation of each face. It issues a warning or raises a ValueError,
        depending on the 'mode' parameter. If 'mode' is set to 'ignore', it does not issue
        a warning or raise an error. Note that this parameter is passed on the check_closed()
        function as the mesh is only orientable if it is closed.

        Parameters
        ----------
        mode : str, optional
            Controls how to handle if the mesh is open and not orientable.
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
            is open and not orientable.

        Warns
        -----
        UserWarning
            If the mesh is not connected and 'mode' is "warn".
        """
        mode = self._validate_mode_arg(mode, arg_name="reorient_faces mode")
        if mode != "skip":
            if self._status_open is None:
                if mode in ["warn", "raise"]:
                    warnings.warn(
                        f"Unchecked mesh status in {self!r} detected. Now applying check_open()",
                        stacklevel=2,
                    )
                self.check_open(mode=mode)

            if self._status_open:
                msg = f"Open mesh in {self!r} detected. reorient_faces() can give bad results."
                if mode == "warn":
                    warnings.warn(msg, stacklevel=2)
                elif mode == "raise":
                    raise ValueError(msg)

            self._faces = fix_trimesh_orientation(self._vertices, self._faces)
            self._status_reoriented = True

    def get_faces_subsets(self):
        """
        Obtain and return subsets of the mesh. If the mesh has n parts, returns and list of
        length n of faces (m,3) vertices indices triplets corresponding to each part.

        Returns
        -------
        status_disconnected_data : list of numpy.ndarray
            Subsets of faces data.
        """
        if self._status_disconnected_data is None:
            self._status_disconnected_data = get_disconnected_faces_subsets(self._faces)
        return self._status_disconnected_data

    def get_open_edges(self):
        """
        Obtain and return the potential open edges. If the mesh has n open edges, returns an
        corresponding (n,2) array of vertices indices doubles.

        Returns
        -------
        status_open_data : numpy.ndarray
            Open edges data.
        """
        if self._status_open_data is None:
            self._status_open_data = get_open_edges(self._faces)
        return self._status_open_data

    def get_selfintersecting_faces(self):
        """
        Obtain and return the potential self intersecting faces indices. If the mesh has n
        intersecting faces, returns a corresponding 1D array length n faces indices.

        Returns
        -------
        status_open_data : numpy.ndarray
            Open edges data.
        """
        if self._status_selfintersecting_data is None:
            self._status_selfintersecting_data = get_intersecting_triangles(
                self._vertices, self._faces
            )
        return self._status_selfintersecting_data

    @property
    def status_disconnected_data(self):
        """Status for connectedness (faces subsets)"""
        return self._status_disconnected_data

    @property
    def status_open_data(self):
        """Status for openness (open edges)"""
        return self._status_open_data

    @property
    def status_selfintersecting(self):
        """Is-selfintersecting boolean check"""
        return self._status_selfintersecting

    @property
    def status_selfintersecting_data(self):
        """return self-intersecting faces"""
        return self._status_selfintersecting_data

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
        centroid = (
            np.array([0.0, 0.0, 0.0])
            if vertices is None
            else calculate_centroid(vertices, faces)
        )
        return orientation.apply(centroid) + position

    def _input_check(self, vertices, faces):
        """input checks here ?"""
        # no. vertices must exceed largest triangle index
        # not all vertices can lie in a plane
        # unique vertices ?
        # do validation checks
        if vertices is None:
            msg = f"Parameter `vertices` of {self} must be set."
            raise MagpylibMissingInput(msg)
        if faces is None:
            msg = f"Parameter `faces` of {self} must be set."
            raise MagpylibMissingInput(msg)
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
            msg = "Some `faces` indices do not match with `vertices` array"
            raise IndexError(msg) from e
        return verts, trias

    def to_TriangleCollection(self):
        """Return a Collection of Triangle objects from the current TriangularMesh"""
        tris = [Triangle(polarization=self.polarization, vertices=v) for v in self.mesh]
        coll = Collection(tris)
        coll.position = self.position
        coll.orientation = self.orientation
        # pylint: disable=no-member
        coll.style.update(self.style.as_dict(), _match_properties=False)
        return coll

    @classmethod
    def from_ConvexHull(
        cls,
        position=(0, 0, 0),
        orientation=None,
        points=None,
        polarization=None,
        magnetization=None,
        check_open="warn",
        check_disconnected="warn",
        reorient_faces=True,
        style=None,
        **kwargs,
    ):
        """Create a TriangularMesh magnet from a point cloud via its convex hull.

        SI units are used for all inputs and outputs.

        Parameters
        ----------
        position: array_like, shape (3,) or (m,3)
            Object position(s) in the global coordinates in units of m. For m>1, the
            `position` and `orientation` attributes together represent an object path.

        orientation: scipy `Rotation` object with length 1 or m, default=`None`
            Object orientation(s) in the global coordinates. `None` corresponds to
            a unit-rotation. For m>1, the `position` and `orientation` attributes
            together represent an object path.

        points: ndarray, shape (n,3)
            Point cloud from which the convex hull is computed.

        polarization: array_like, shape (3,), default=`None`
            Magnetic polarization vector J = mu0*M in units of T,
            given in the local object coordinates (rotates with object).

        magnetization: array_like, shape (3,), default=`None`
            Magnetization vector M = J/mu0 in units of A/m,
            given in the local object coordinates (rotates with object).

        reorient_faces: bool, default=`True`
            In a properly oriented mesh, all faces must be oriented outwards.
            If `True`, check and fix the orientation of each triangle.

        check_open: {'warn', 'raise', 'ignore'}, default='warn'
            Only a closed mesh guarantees a physical magnet.
            If the mesh is open and "warn", a warning is issued.
            If the mesh is open and "raise", a ValueError is raised.
            If "ignore", no mesh check is performed.

        check_disconnected: {'warn', 'raise', 'ignore'}, default='warn'
            If the mesh is disconnected and "warn", a warning is issued.
            If the mesh is disconnected and "raise", a ValueError is raised.
            If "ignore", no mesh check is performed.

        check_selfintersecting: {'warn', 'raise', 'ignore'}, default='warn'
            If the mesh is self-intersecting and "warn", a warning is issued.
            If the mesh is self-intersecting and "raise", a ValueError is raised.
            If "ignore", no mesh check is performed.

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
        """
        return cls(
            position=position,
            orientation=orientation,
            vertices=points,
            faces=ConvexHull(points).simplices,
            polarization=polarization,
            magnetization=magnetization,
            reorient_faces=reorient_faces,
            check_open=check_open,
            check_disconnected=check_disconnected,
            style=style,
            **kwargs,
        )

    @classmethod
    def from_pyvista(
        cls,
        position=(0, 0, 0),
        orientation=None,
        polydata=None,
        polarization=None,
        magnetization=None,
        check_open="warn",
        check_disconnected="warn",
        reorient_faces=True,
        style=None,
        **kwargs,
    ):
        """Create a TriangularMesh magnet from a pyvista PolyData mesh object.

        SI units are used for all inputs and outputs.

        Parameters
        ----------
        position: array_like, shape (3,) or (m,3)
            Object position(s) in the global coordinates in units of m. For m>1, the
            `position` and `orientation` attributes together represent an object path.

        orientation: scipy `Rotation` object with length 1 or m, default=`None`
            Object orientation(s) in the global coordinates. `None` corresponds to
            a unit-rotation. For m>1, the `position` and `orientation` attributes
            together represent an object path.

        polydata: pyvista.core.pointset.PolyData object
            A valid pyvista Polydata mesh object. (e.g. `pyvista.Sphere()`)

        polarization: array_like, shape (3,), default=`None`
            Magnetic polarization vector J = mu0*M in units of T,
            given in the local object coordinates (rotates with object).

        magnetization: array_like, shape (3,), default=`None`
            Magnetization vector M = J/mu0 in units of A/m,
            given in the local object coordinates (rotates with object).

        reorient_faces: bool, default=`True`
            In a properly oriented mesh, all faces must be oriented outwards.
            If `True`, check and fix the orientation of each triangle.

        check_open: {'warn', 'raise', 'ignore'}, default='warn'
            Only a closed mesh guarantees a physical magnet.
            If the mesh is open and "warn", a warning is issued.
            If the mesh is open and "raise", a ValueError is raised.
            If "ignore", no mesh check is performed.

        check_disconnected: {'warn', 'raise', 'ignore'}, default='warn'
            If the mesh is disconnected and "warn", a warning is issued.
            If the mesh is disconnected and "raise", a ValueError is raised.
            If "ignore", no mesh check is performed.

        check_selfintersecting: {'warn', 'raise', 'ignore'}, default='warn'
            If the mesh is self-intersecting and "warn", a warning is issued.
            If the mesh is self-intersecting and "raise", a ValueError is raised.
            If "ignore", no mesh check is performed.

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
        """
        # pylint: disable=import-outside-toplevel
        try:
            import pyvista  # noqa: PLC0415
        except ImportError as missing_module:  # pragma: no cover
            msg = """In order load pyvista Polydata objects, you first need to install pyvista via pip
                or conda, see https://docs.pyvista.org/getting-started/installation.html"""
            raise ModuleNotFoundError(msg) from missing_module
        if not isinstance(polydata, pyvista.core.pointset.PolyData):
            msg = (
                "The `polydata` parameter must be an instance of `pyvista.core.pointset.PolyData`, "
                f"received {polydata!r} instead"
            )
            raise TypeError(msg)
        polydata = polydata.triangulate()
        vertices = polydata.points
        faces = polydata.faces.reshape(-1, 4)[:, 1:]

        return cls(
            position=position,
            orientation=orientation,
            vertices=vertices,
            faces=faces,
            polarization=polarization,
            magnetization=magnetization,
            reorient_faces=reorient_faces,
            check_open=check_open,
            check_disconnected=check_disconnected,
            style=style,
            **kwargs,
        )

    @classmethod
    def from_triangles(
        cls,
        position=(0, 0, 0),
        orientation=None,
        triangles=None,
        polarization=None,
        magnetization=None,
        reorient_faces=True,
        check_open="warn",
        check_disconnected="warn",
        style=None,
        **kwargs,
    ):
        """Create a TriangularMesh magnet from a list or Collection of Triangle objects.

        SI units are used for all inputs and outputs.

        Parameters
        ----------
        position: array_like, shape (3,) or (m,3)
            Object position(s) in the global coordinates in units of m. For m>1, the
            `position` and `orientation` attributes together represent an object path.

        orientation: scipy `Rotation` object with length 1 or m, default=`None`
            Object orientation(s) in the global coordinates. `None` corresponds to
            a unit-rotation. For m>1, the `position` and `orientation` attributes
            together represent an object path.

        triangles: list or Collection of Triangle objects
            Only vertices of Triangle objects are taken, magnetization is ignored.

        polarization: array_like, shape (3,), default=`None`
            Magnetic polarization vector J = mu0*M in units of T,
            given in the local object coordinates (rotates with object).

        magnetization: array_like, shape (3,), default=`None`
            Magnetization vector M = J/mu0 in units of A/m,
            given in the local object coordinates (rotates with object).

        reorient_faces: bool, default=`True`
            In a properly oriented mesh, all faces must be oriented outwards.
            If `True`, check and fix the orientation of each triangle.

        check_open: {'warn', 'raise', 'ignore'}, default='warn'
            Only a closed mesh guarantees a physical magnet.
            If the mesh is open and "warn", a warning is issued.
            If the mesh is open and "raise", a ValueError is raised.
            If "ignore", no mesh check is performed.

        check_disconnected: {'warn', 'raise', 'ignore'}, default='warn'
            If the mesh is disconnected and "warn", a warning is issued.
            If the mesh is disconnected and "raise", a ValueError is raised.
            If "ignore", no mesh check is performed.

        check_selfintersecting: {'warn', 'raise', 'ignore'}, default='warn'
            If the mesh is self-intersecting and "warn", a warning is issued.
            If the mesh is self-intersecting and "raise", a ValueError is raised.
            If "ignore", no mesh check is performed.

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
        """
        if not isinstance(triangles, list | Collection):
            msg = (
                "The `triangles` parameter must be a list or Collection of `Triangle` objects, "
                f"\nreceived type {type(triangles)} instead"
            )
            raise TypeError(msg)
        for obj in triangles:
            if not isinstance(obj, Triangle):
                msg = (
                    "All elements of `triangles` must be `Triangle` objects, "
                    f"\nreceived type {type(obj)} instead"
                )
                raise TypeError(msg)
        mesh = np.array([tria.vertices for tria in triangles])
        vertices, tr = np.unique(mesh.reshape((-1, 3)), axis=0, return_inverse=True)
        faces = tr.reshape((-1, 3))

        return cls(
            position=position,
            orientation=orientation,
            vertices=vertices,
            faces=faces,
            polarization=polarization,
            magnetization=magnetization,
            reorient_faces=reorient_faces,
            check_open=check_open,
            check_disconnected=check_disconnected,
            style=style,
            **kwargs,
        )

    @classmethod
    def from_mesh(
        cls,
        position=(0, 0, 0),
        orientation=None,
        mesh=None,
        polarization=None,
        magnetization=None,
        reorient_faces=True,
        check_open="warn",
        check_disconnected="warn",
        style=None,
        **kwargs,
    ):
        """Create a TriangularMesh magnet from a mesh input.

        SI units are used for all inputs and outputs.

        Parameters
        ----------
        position: array_like, shape (3,) or (m,3)
            Object position(s) in the global coordinates in units of m. For m>1, the
            `position` and `orientation` attributes together represent an object path.

        orientation: scipy `Rotation` object with length 1 or m, default=`None`
            Object orientation(s) in the global coordinates. `None` corresponds to
            a unit-rotation. For m>1, the `position` and `orientation` attributes
            together represent an object path.

        mesh: array_like, shape (n,3,3)
            An array_like of triangular faces that make up a triangular mesh.

        polarization: array_like, shape (3,), default=`None`
            Magnetic polarization vector J = mu0*M in units of T,
            given in the local object coordinates (rotates with object).

        magnetization: array_like, shape (3,), default=`None`
            Magnetization vector M = J/mu0 in units of A/m,
            given in the local object coordinates (rotates with object).

        reorient_faces: bool, default=`True`
            In a properly oriented mesh, all faces must be oriented outwards.
            If `True`, check and fix the orientation of each triangle.

        check_open: {'warn', 'raise', 'ignore'}, default='warn'
            Only a closed mesh guarantees a physical magnet.
            If the mesh is open and "warn", a warning is issued.
            If the mesh is open and "raise", a ValueError is raised.
            If "ignore", no mesh check is performed.

        check_disconnected: {'warn', 'raise', 'ignore'}, default='warn'
            If the mesh is disconnected and "warn", a warning is issued.
            If the mesh is disconnected and "raise", a ValueError is raised.
            If "ignore", no mesh check is performed.

        check_selfintersecting: {'warn', 'raise', 'ignore'}, default='warn'
            If the mesh is self-intersecting and "warn", a warning is issued.
            If the mesh is self-intersecting and "raise", a ValueError is raised.
            If "ignore", no mesh check is performed.

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
        """
        mesh = check_format_input_vector2(
            mesh,
            shape=[None, 3, 3],
            param_name="mesh",
        )
        vertices, tr = np.unique(mesh.reshape((-1, 3)), axis=0, return_inverse=True)
        faces = tr.reshape((-1, 3))

        return cls(
            position=position,
            orientation=orientation,
            vertices=vertices,
            faces=faces,
            polarization=polarization,
            magnetization=magnetization,
            reorient_faces=reorient_faces,
            check_open=check_open,
            check_disconnected=check_disconnected,
            style=style,
            **kwargs,
        )

    @property
    def _default_style_description(self):
        """Default style description text"""
        ntri = len(self.faces)
        return f"{ntri} face{'s'[: ntri ^ 1]}"

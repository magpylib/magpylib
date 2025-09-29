"""TriangularMesh class code"""

# pylint:disable=too-many-lines
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=too-many-positional-arguments
# pylint: disable=arguments-differ

import warnings
from typing import ClassVar

import numpy as np
from scipy.spatial import ConvexHull  # pylint: disable=no-name-in-module

from magpylib._src.display.traces_core import make_TriangularMesh
from magpylib._src.exceptions import MagpylibMissingInput
from magpylib._src.fields.field_BH_triangularmesh import (
    _BHJM_magnet_trimesh,
    _calculate_centroid,
    _fix_trimesh_orientation,
    _get_disconnected_faces_subsets,
    _get_intersecting_triangles,
    _get_open_edges,
)
from magpylib._src.input_checks import (
    check_format_input_vector,
    check_format_input_vector2,
)
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet
from magpylib._src.obj_classes.class_BaseProperties import (
    BaseDipoleMoment,
    BaseVolume,
)
from magpylib._src.obj_classes.class_BaseTarget import BaseTarget
from magpylib._src.obj_classes.class_Collection import Collection
from magpylib._src.obj_classes.class_misc_Triangle import Triangle
from magpylib._src.obj_classes.target_meshing import _target_mesh_triangularmesh
from magpylib._src.style import TriangularMeshStyle


class TriangularMesh(BaseMagnet, BaseTarget, BaseVolume, BaseDipoleMoment):
    """Magnet with homogeneous magnetization defined by a triangular surface mesh.

    Can be used as ``sources`` input for magnetic field computation and ``target``
    input for force computation.

    The vertex positions are defined in the local object coordinates (rotate with object).
    When ``position=(0, 0, 0)`` and ``orientation=None`` global and local coordinates coincide.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    position : array-like, shape (3,) or (p, 3), default (0, 0, 0)
        Object position(s) in global coordinates in units (m). ``position`` and
        ``orientation`` attributes define the object path.
    orientation : Rotation | None, default None
        Object orientation(s) in global coordinates as a scipy Rotation. Rotation can
        have length 1 or p. ``None`` generates a unit-rotation.
    vertices : array-like, shape (n, 3), default None
        Points in units (m) in the local object coordinates from which the triangular
        faces of the mesh are constructed by the additional ``faces`` input.
    faces : array-like, shape (n, 3), default None
        Triplets of vertex indices. Each triplet represents one triangle of the mesh.
    polarization : None | array-like, shape (3,), default None
        Magnetic polarization vector J = mu0*M in units (T), given in the
        local object coordinates. Sets also ``magnetization``.
    magnetization : None | array-like, shape (3,), default None
        Magnetization vector M = J/mu0 in units (A/m), given in the local
        object coordinates. Sets also ``polarization``.
    meshing : int | None, default None
        Mesh fineness for force computation. Must be a positive integer specifying
        the target mesh size.
    reorient_faces : bool | {'warn', 'raise', 'ignore', 'skip'}, default 'warn'
        In a properly oriented mesh, all faces must be oriented outwards. If ``True``,
        check and fix the orientation of each triangle. ``True`` translates to ``'warn'``
        and ``False`` to ``'skip'``.
    check_open : bool | {'warn', 'raise', 'ignore', 'skip'}, default 'warn'
        Only a closed mesh guarantees correct B-field computation. If ``True``, check if the
        mesh is open. ``True`` translates to ``'warn'`` and ``False`` to ``'skip'``.
    check_disconnected : bool | {'warn', 'raise', 'ignore', 'skip'}, default 'warn'
        Individual magnets should be connected bodies to avoid confusion. If ``True``, check if
        the mesh is disconnected. ``True`` translates to ``'warn'`` and ``False`` to ``'skip'``.
    check_selfintersecting : bool | {'warn', 'raise', 'ignore', 'skip'}, default 'warn'
        A proper body cannot have a self-intersecting mesh. If ``True``, check if the mesh is
        self-intersecting. ``True`` translates to ``'warn'`` and ``False`` to ``'skip'``.
    style : dict | None, default None
        Style dictionary. Can also be provided via style underscore magic, e.g.
        ``style_color='red'``.

    Attributes
    ----------
    position : ndarray, shape (3,) or (p, 3)
        Same as constructor parameter ``position``.
    orientation : Rotation
        Same as constructor parameter ``orientation``.
    vertices : ndarray, shape (n, 3)
        Same as constructor parameter ``vertices``.
    faces : ndarray, shape (n, 3)
        Same as constructor parameter ``faces``.
    polarization : None | ndarray, shape (3,)
        Same as constructor parameter ``polarization``.
    magnetization : None | ndarray, shape (3,)
        Same as constructor parameter ``magnetization``.
    meshing : int | None
        Same as constructor parameter ``meshing``.
    centroid : ndarray, shape (3,) or (p, 3)
        Read-only. Object centroid.
    dipole_moment : ndarray, shape (3,)
        Read-only. Object dipole moment (A·m²) in local object coordinates.
    volume : float
        Read-only. Object physical volume in units (m³).
    parent : Collection | None
        Parent collection of the object.
    style : dict
        Style dictionary defining visual properties.

    Notes
    -----
    Returns (0, 0, 0) on corners.

    Examples
    --------
    We compute the B-field in units (T) of a triangular mesh (4 vertices, 4 faces)
    with polarization (0.1, 0.2, 0.3) (T) at the observer position
    (0.01, 0.01, 0.01) (m):

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> vv = ((0, 0, 0), (0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.01))
    >>> tt = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    >>> trim = magpy.magnet.TriangularMesh(
    ...     polarization=(0.1, 0.2, 0.3), vertices=vv, faces=tt,
    ... )
    >>> with np.printoptions(precision=3):
    ...     print(trim.getB((0.01, 0.01, 0.01)) * 1000)
    [2.602 2.082 1.561]
    """

    _field_func = staticmethod(_BHJM_magnet_trimesh)
    _field_func_kwargs_ndim: ClassVar[dict[str, int]] = {"polarization": 2, "mesh": 3}
    _force_type = "magnet"
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
        meshing=None,
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
        # Initialize BaseTarget with meshing parameter
        BaseTarget.__init__(self, meshing=meshing)

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
        """Volume of object in units (m³).

        Based on algorithm from: https://n-e-r-v-o-u-s.com/blog/?p=4415
        For each triangle, compute the signed volume of tetrahedron from origin
        to triangle using: V = (1/6) * (v1 x v2) . v3
        Sum all signed volumes to get total mesh volume.

        Returns
        -------
        float
            Volume in units (m³).
        """
        if self._vertices is None or self._faces is None:
            return 0.0
        if self.status_open is None:
            self.check_open()
        if self.status_open is True:
            msg = f"Open mesh detected in {self!r}. Cannot compute volume."
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

    def _get_centroid(self, squeeze=True):
        """Centroid of object in units (m)."""
        if squeeze:
            return self.barycenter
        return self._barycenter

    def _get_dipole_moment(self):
        """Magnetic moment of object in units (A*m²)."""
        # test init
        if self.magnetization is None or self.vertices is None or self.faces is None:
            return np.array((0.0, 0.0, 0.0))
        return self.magnetization * self.volume

    def _generate_mesh(self):
        """Generate mesh for force computation."""
        # Tests in getFT ensure that meshing, dimension and excitation are set
        return _target_mesh_triangularmesh(
            self.vertices,
            self.faces,
            self.meshing,
            self.volume,
            self.magnetization,
        )

    @staticmethod
    def _validate_mode_arg(arg, arg_name="mode"):
        """Validate mode argument"""
        accepted_arg_vals = (True, False, "warn", "raise", "ignore", "skip")
        if arg not in accepted_arg_vals:
            msg = (
                f"Input {arg_name} must be one of {{'warn', 'raise', 'ignore', 'skip', True, False}}; "
                f"instead received {arg!r}. "
                "Note that True translates to 'warn' and False to 'skip'."
            )
            raise ValueError(msg)
        return "warn" if arg is True else "skip" if arg is False else arg

    def check_open(self, mode="warn"):
        """Check whether the mesh is open.

        Parameters
        ----------
        mode : bool | {'warn', 'raise', 'ignore', 'skip'}, default 'warn'
            Controls how to handle open meshes. ``True`` translates to ``'warn'`` and
            ``False`` to ``'skip'``.

        Returns
        -------
        bool
            ``True`` if the mesh is open, ``False`` otherwise.
        """
        mode = self._validate_mode_arg(mode, arg_name="check_open mode")
        if mode != "skip" and self._status_open is None:
            self._status_open = len(self._get_open_edges()) > 0
            if self._status_open:
                msg = (
                    f"Open mesh detected in {self!r}. Intrinsic inside-outside checks may "
                    "give bad results and subsequently getB() and reorient_faces() may "
                    "give bad results as well. "
                    "This check can be disabled at initialization with check_open='skip'. "
                    "Open edges can be displayed in show() with "
                    "style_mesh_open_show=True. "
                    "Open edges are stored in the status_open_data property."
                )
                if mode == "warn":
                    warnings.warn(msg, stacklevel=2)
                elif mode == "raise":
                    raise ValueError(msg)
        return self._status_open

    def check_disconnected(self, mode="warn"):
        """Check whether the mesh is disconnected.

        Parameters
        ----------
        mode : bool | {'warn', 'raise', 'ignore', 'skip'}, default 'warn'
            Controls how to handle disconnected meshes. ``True`` translates to ``'warn'`` and
            ``False`` to ``'skip'``.

        Returns
        -------
        bool
            ``True`` if the mesh is disconnected, ``False`` otherwise.
        """
        mode = self._validate_mode_arg(mode, arg_name="check_disconnected mode")
        if mode != "skip" and self._status_disconnected is None:
            self._status_disconnected = len(self.get_faces_subsets()) > 1
            if self._status_disconnected:
                msg = (
                    f"Disconnected mesh detected in {self!r}. Magnet consists of multiple "
                    "individual parts. "
                    "This check can be disabled at initialization with "
                    "check_disconnected='skip'. "
                    "Parts can be displayed in show() with "
                    "style_mesh_disconnected_show=True. "
                    "Parts are stored in the status_disconnected_data property."
                )
                if mode == "warn":
                    warnings.warn(msg, stacklevel=2)
                elif mode == "raise":
                    raise ValueError(msg)
        return self._status_disconnected

    def check_selfintersecting(self, mode="warn"):
        """Check whether the mesh is self-intersecting.

        Parameters
        ----------
        mode : bool | {'warn', 'raise', 'ignore', 'skip'}, default 'warn'
            Controls how to handle self-intersecting meshes. ``True`` translates to ``'warn'`` and
            ``False`` to ``'skip'``.

        Returns
        -------
        bool
            ``True`` if the mesh is self-intersecting, ``False`` otherwise.
        """
        mode = self._validate_mode_arg(mode, arg_name="check_selfintersecting mode")
        if mode != "skip" and self._status_selfintersecting is None:
            self._status_selfintersecting = len(self.get_selfintersecting_faces()) > 1
            if self._status_selfintersecting:
                msg = (
                    f"Self-intersecting mesh detected in {self!r}. "
                    "This check can be disabled at initialization with "
                    "check_selfintersecting='skip'. "
                    "Intersecting faces can be displayed in show() with "
                    "style_mesh_selfintersecting_show=True. "
                    "Faces are stored in the status_selfintersecting_data property."
                )
                if mode == "warn":
                    warnings.warn(msg, stacklevel=2)
                elif mode == "raise":
                    raise ValueError(msg)
        return self._status_selfintersecting

    def reorient_faces(self, mode="warn"):
        """Reorient all faces to point outwards.

        In a properly oriented mesh, all faces must be oriented outwards. This function
        fixes the orientation of each face. Note that ``mode`` is also considered in
        ``check_open()`` because the mesh is only orientable if it is closed.

        Parameters
        ----------
        mode : bool | {'warn', 'raise', 'ignore', 'skip'}, default 'warn'
            Controls how to handle open meshes during reorientation. ``True`` translates to
            ``'warn'`` and ``False`` to ``'skip'``.

        Returns
        -------
        None
        """
        mode = self._validate_mode_arg(mode, arg_name="reorient_faces mode")
        if mode != "skip":
            if self._status_open is None:
                if mode in ["warn", "raise"]:
                    warnings.warn(
                        f"Unchecked mesh status in {self!r} detected. Now applying check_open().",
                        stacklevel=2,
                    )
                self.check_open(mode=mode)

            if self._status_open:
                msg = f"Open mesh detected in {self!r}. reorient_faces() can give bad results."
                if mode == "warn":
                    warnings.warn(msg, stacklevel=2)
                elif mode == "raise":
                    raise ValueError(msg)

            self._faces = _fix_trimesh_orientation(self._vertices, self._faces)
            self._status_reoriented = True

    def get_faces_subsets(self):
        """Get subsets of faces for each disconnected part of the mesh.

        If the mesh has ``k`` disconnected parts, returns a list of length ``k`` with
        arrays of shape (m, 3) containing vertex index triplets for each part.

        Returns
        -------
        list
            List of ndarrays. Subsets of faces data.
        """
        if self._status_disconnected_data is None:
            self._status_disconnected_data = _get_disconnected_faces_subsets(
                self._faces
            )
        return self._status_disconnected_data

    def _get_open_edges(self):
        """Return open edges.

        If the mesh has n open edges, returns an array of shape (n, 2) of vertex
        indices forming those edges.

        Returns
        -------
        ndarray, shape (n, 2)
            Open edges as pairs of vertex indices.
        """
        if self._status_open_data is None:
            self._status_open_data = _get_open_edges(self._faces)
        return self._status_open_data

    def get_selfintersecting_faces(self):
        """Return indices of potentially self-intersecting faces.

        If the mesh has n intersecting faces, returns a 1D array of length n with
        their face indices.

        Returns
        -------
        ndarray, shape (n,)
            Indices of potentially self-intersecting faces.
        """
        if self._status_selfintersecting_data is None:
            self._status_selfintersecting_data = _get_intersecting_triangles(
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
            else _calculate_centroid(vertices, faces)
        )
        return orientation.apply(centroid) + position

    def _input_check(self, vertices, faces):
        """input checks here ?"""
        # no. vertices must exceed largest triangle index
        # not all vertices can lie in a plane
        # unique vertices ?
        # do validation checks
        if vertices is None:
            msg = f"Input vertices of {self!r} must be set."
            raise MagpylibMissingInput(msg)
        if faces is None:
            msg = f"Input faces of {self!r} must be set."
            raise MagpylibMissingInput(msg)
        verts = check_format_input_vector(
            vertices,
            dims=(2,),
            shape_m1=3,
            sig_name="TriangularMesh.vertices",
            sig_type="array-like (list, tuple, ndarray) of shape (n, 3)",
        )
        trias = check_format_input_vector(
            faces,
            dims=(2,),
            shape_m1=3,
            sig_name="TriangularMesh.faces",
            sig_type="array-like (list, tuple, ndarray) of shape (n, 3)",
        ).astype(int)
        try:
            verts[trias]
        except IndexError as e:
            msg = "Some faces indices do not match the vertices array."
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
        meshing=None,
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
        points : array-like, shape (n, 3)
            Point cloud from which the convex hull is computed.
        position, orientation, polarization, magnetization, meshing, reorient_faces, check_open, check_disconnected, style :
            See ``TriangularMesh`` for shared parameter semantics and defaults.

        Returns
        -------
        TriangularMesh
            New magnet instance defined by the convex hull of ``points``.
        """
        return cls(
            position=position,
            orientation=orientation,
            vertices=points,
            faces=ConvexHull(points).simplices,
            polarization=polarization,
            magnetization=magnetization,
            meshing=meshing,
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
        meshing=None,
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
        polydata : pyvista.core.pointset.PolyData
            A valid pyvista PolyData mesh object (e.g. ``pyvista.Sphere()``).
        position, orientation, polarization, magnetization, meshing, reorient_faces, check_open, check_disconnected, style :
            See ``TriangularMesh`` for shared parameter semantics and defaults.

        Returns
        -------
        TriangularMesh
            New magnet instance defined by ``polydata``.
        """
        # pylint: disable=import-outside-toplevel
        try:
            import pyvista  # noqa: PLC0415
        except ImportError as missing_module:  # pragma: no cover
            msg = (
                "For loading PolyData objects, install PyVista. "
                "See https://docs.pyvista.org/getting-started/installation.html."
            )
            raise ModuleNotFoundError(msg) from missing_module
        if not isinstance(polydata, pyvista.core.pointset.PolyData):
            msg = (
                "Input polydata must be an instance of "
                "pyvista.core.pointset.PolyData; "
                f"instead received {polydata!r}."
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
            meshing=meshing,
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
        meshing=None,
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
        triangles : list or Collection
            Only ``vertices`` of ``Triangle`` objects in list or collection are taken,
            ``magnetization`` is ignored.
        position, orientation, polarization, magnetization, meshing, reorient_faces, check_open, check_disconnected, style :
            See ``TriangularMesh`` for shared parameter semantics and defaults.

        Returns
        -------
        TriangularMesh
            New magnet instance defined by the provided ``triangles``.
        """
        if not isinstance(triangles, list | Collection):
            msg = (
                "Input triangles must be a list or Collection of Triangle objects; "
                f"instead received type {type(triangles).__name__!r}."
            )
            raise TypeError(msg)
        for obj in triangles:
            if not isinstance(obj, Triangle):
                msg = (
                    "Input triangles must be a list or Collection of Triangle objects; "
                    f"instead received type {type(obj).__name__!r}."
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
            meshing=meshing,
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
        meshing=None,
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
        mesh : array-like, shape (n, 3, 3)
            Triangular faces that make up a triangular mesh.
        position, orientation, polarization, magnetization, meshing, reorient_faces, check_open, check_disconnected, style :
            See ``TriangularMesh`` for shared parameter semantics and defaults.

        Returns
        -------
        TriangularMesh
            New magnet instance defined by the provided ``mesh``.
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
            meshing=meshing,
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

"""TriangleSheet current class code"""

# pylint: disable=too-many-positional-arguments

from __future__ import annotations

from typing import ClassVar

import numpy as np

from magpylib._src.display.traces_core import make_TriangleSheet
from magpylib._src.fields.field_BH_current_sheet import _BHJM_current_trisheet
from magpylib._src.input_checks import check_format_input_numeric
from magpylib._src.obj_classes.class_BaseExcitations import BaseSource
from magpylib._src.obj_classes.class_BaseTarget import BaseTarget
from magpylib._src.obj_classes.target_meshing import _target_mesh_triangle_current
from magpylib._src.style import CurrentSheetStyle


class TriangleSheet(BaseSource, BaseTarget):
    """Surface current density flowing along triangular faces.

    Can be used as ``sources`` input for magnetic field computation and ``target``
    input for force computation.

    The vertex positions are defined in the local object coordinates (rotate with
    object). When ``position=(0, 0, 0)`` and ``orientation=None`` global and local
    coordinates coincide.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    position : array-like, shape (3,) or (p, 3), default (0, 0, 0)
        Object position(s) in global coordinates in units (m). ``position`` and
        ``orientation`` attributes define the object path.
    orientation : Rotation | None, default None
        Object orientation(s) in global coordinates as a scipy Rotation. Rotation can
        have length 1 or p. ``None`` generates a unit-rotation.
    vertices : array-like, shape (n, 3) or (p, n, 3), default None
        Points in units (m) in the local object coordinates from which the
        triangular faces are constructed by the additional ``faces`` input.
    faces : array-like, shape (n, 3), default None
        Triplets of vertex indices. Each triplet represents one triangle of the mesh.
    current_densities : array-like, shape (n, 3) or (p, n, 3), default None
        Electrical current densities flowing on the faces in units (A/m). The
        effective current density is a projection of the given current density
        vector into the face planes. Input must have the same length as ``faces``.
    meshing : int | None, default None
        Mesh fineness for force computation. Must be an integer >= number of
        faces specifying the target mesh size. The mesh is generated via bisection
        along longest edges until target number is reached.
    style : dict | None, default None
        Style dictionary. Can also be provided via style underscore magic, e.g.
        ``style_color='red'``.

    Attributes
    ----------
    position : ndarray, shape (3,) or (p, 3)
        Same as constructor parameter ``position``.
    orientation : Rotation
        Same as constructor parameter ``orientation``.
    vertices : None | ndarray, shape (n, 3) or (p, n, 3)
        Same as constructor parameter ``vertices``.
    faces : None | ndarray, shape (n, 3)
        Same as constructor parameter ``faces``.
    current_densities : None | ndarray, shape (n, 3) or (p, n, 3)
        Same as constructor parameter ``current_densities``.
    meshing : None | int
        Same as constructor parameter ``meshing``.
    centroid : ndarray, shape (3,) or (p, 3)
        Read-only. Object centroid computed via mean of vertices in units (m)
        in global coordinates.
    parent : None | Collection
        Parent collection of the object.
    style : CurrentSheetStyle
        Object style. See CurrentSheetStyle for details.

    Notes
    -----
    Returns (0, 0, 0) on a sheet.

    Examples
    --------
    >>> import numpy as np
    >>> import magpylib as magpy
    >>> src = magpy.current.TriangleSheet(
    ...     current_densities=[(1, 0, 0), (0, 1, 0)],
    ...     vertices=((0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)),
    ...     faces=((0, 1, 2), (1, 2, 3)),
    ... )
    >>> H = src.getH((0.01, 0.01, 0.01))
    >>> with np.printoptions(precision=3):
    ...     print(H)
    [ 0.005 -0.311 -0.299]
    """

    # pylint: disable=dangerous-default-value
    _field_func = staticmethod(_BHJM_current_trisheet)
    _force_type = "current"
    _field_func_kwargs_ndim: ClassVar[dict[str, int]] = {
        "current_densities": 3,
        "vertices": 3,
        "faces": 3,
    }
    _path_properties = ("current_densities", "vertices")
    get_trace = make_TriangleSheet
    _style_class = CurrentSheetStyle

    def __init__(
        self,
        position=(0, 0, 0),
        orientation=None,
        vertices=None,
        faces=None,
        current_densities=None,
        meshing=None,
        *,
        style=None,
        **kwargs,
    ):
        # instance attributes
        self._current_densities, self._vertices, self._faces = self._input_check(
            current_densities, vertices, faces
        )

        # Inherit
        super().__init__(position, orientation, style=style, **kwargs)
        BaseTarget.__init__(self, meshing)

    # property getters and setters
    @property
    def vertices(self):
        """TriangleSheet Vertices"""
        return np.squeeze(self._vertices) if self._vertices is not None else None

    @vertices.setter
    def vertices(self, val):
        """Set vertices.

        Parameters
        ----------
        val : array-like, shape (n, 3) or (p, n, 3)
            Vertices in local object coordinates in units (m).
        """
        verts = check_format_input_numeric(
            val,
            dtype=float,
            shapes=((3,), (None, 3), (None, None, 3)),
            name="TriangleSheet.vertices",
        )
        # allow init of single vertex without extra dimension
        if verts.ndim == 1:
            verts = np.expand_dims(verts, 0)
        # reshape (n, 3) -> (1, n, 3) for path handling
        if verts.ndim == 2:
            verts = np.reshape(verts, (-1, verts.shape[0], verts.shape[1]))

        if verts.shape[-2] < 3:
            msg = (
                f"Input vertices of {self} must have at least 3 vertices; "
                f"instead received {verts.shape[-2]} vertices."
            )
            raise ValueError(msg)

        self._vertices = verts

    @property
    def faces(self):
        """TriangleSheet Faces"""
        return self._faces

    @property
    def current_densities(self):
        """TriangleSheet CurrentDensities"""
        return (
            np.squeeze(self._current_densities)
            if self._current_densities is not None
            else None
        )

    @current_densities.setter
    def current_densities(self, val):
        """Set current_densities"""
        cd = check_format_input_numeric(
            val,
            dtype=float,
            shapes=((3,), (None, 3), (None, None, 3)),
            name="TriangleSheet.current_densities",
        )
        # allow init of single faces without extra dimension
        if cd.ndim == 1:
            cd = np.expand_dims(cd, 0)
        # reshape (n, 3) -> (1, n, 3) for path handling
        if cd.ndim == 2:
            cd = np.reshape(cd, (-1, cd.shape[0], cd.shape[1]))

        if len(self._faces) != cd.shape[-2]:
            msg = f"Input current_densities and faces of {self} must have same length."
            raise ValueError(msg)

        self._current_densities = cd

    def _input_check(self, current_densities, vertices, faces):
        """check and format user inputs"""
        cd = check_format_input_numeric(
            current_densities,
            dtype=float,
            shapes=((3,), (None, 3), (None, None, 3)),
            name="TriangleSheet.current_densities",
        )
        verts = check_format_input_numeric(
            vertices,
            dtype=float,
            shapes=((3,), (None, 3), (None, None, 3)),
            name="TriangleSheet.vertices",
        )
        fac = check_format_input_numeric(
            faces,
            dtype=int,
            shapes=((3,), (None, 3)),
            name="TriangleSheet.faces",
        )

        # allow init of single faces without extra dimension
        if cd.ndim == 1:
            cd = np.expand_dims(cd, 0)
        # reshape (n, 3) -> (1, n, 3) for path handling
        if cd.ndim == 2:
            cd = np.reshape(cd, (-1, cd.shape[0], cd.shape[1]))

        if verts.ndim == 1:
            verts = np.expand_dims(verts, 0)
        # reshape (n, 3) -> (1, n, 3) for path handling
        if verts.ndim == 2:
            verts = np.reshape(verts, (-1, verts.shape[0], verts.shape[1]))

        if fac.ndim == 1:
            fac = np.expand_dims(fac, 0)

        if verts.shape[-2] < 3:
            msg = (
                f"Input vertices of {self} must have at least 3 vertices; "
                f"instead received {verts.shape[-2]} vertices."
            )
            raise ValueError(msg)

        # Check against the face count dimension (axis 1 for path arrays)
        n_faces = len(fac)
        n_current_densities = cd.shape[-2]  # Second to last dimension for (p, n, 3)
        if n_faces != n_current_densities:
            msg = f"Input current_densities and faces of {self} must have same length."
            raise ValueError(msg)

        try:
            # Check against first path position for validation
            verts[0][fac]
        except IndexError as e:
            msg = f"Some faces indices of {self} do not match with vertices array."
            raise IndexError(msg) from e

        return cd, verts, fac

    # Methods
    def _get_centroid(self, squeeze=True):
        """Centroid of object in units (m)."""
        # For path arrays (p, n, 3), mean over axis 1 to get (p, 3)
        # For non-path arrays (1, n, 3), mean over axis 1 to get (1, 3)
        centr = np.mean(self._vertices, axis=-2) + self._position
        if squeeze:
            return np.squeeze(centr)
        return centr

    def _generate_mesh(self):
        """Generate mesh for force computation."""
        verts = self._vertices
        cd = self._current_densities
        # verts has shape (p, n, 3) where p is path dimension, n is number of vertices
        # cd has shape (p, m, 3) where m is number of faces
        # faces has shape (m, 3) - indices into vertices

        # Create triangles by indexing vertices with faces
        # triangles shape: (p, m, 3, 3)
        # For each path p, for each face m, get 3 vertices of 3 coordinates
        triangles = verts[:, self.faces, :]  # shape (p, m, 3, 3)

        return _target_mesh_triangle_current(
            triangles,
            cd,
            self.meshing,
        )

    def _validate_meshing(self, value):
        """Makes only sense with at least n_mesh = n_faces."""
        if isinstance(value, int) and value >= len(self.faces):
            pass
        else:
            msg = (
                f"Input meshing of {self} must be an integer >= number of faces; "
                f"instead received {value}."
            )
            raise ValueError(msg)

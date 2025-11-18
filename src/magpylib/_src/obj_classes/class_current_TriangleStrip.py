"""TriangleStrip current class code"""

# pylint: disable=too-many-positional-arguments

from __future__ import annotations

from typing import ClassVar

import numpy as np

from magpylib._src.display.traces_core import make_TriangleStrip
from magpylib._src.fields.field_BH_current_sheet import _BHJM_current_tristrip
from magpylib._src.input_checks import check_format_input_vertices
from magpylib._src.obj_classes.class_BaseExcitations import BaseCurrent
from magpylib._src.obj_classes.class_BaseProperties import BaseDipoleMoment
from magpylib._src.obj_classes.class_BaseTarget import BaseTarget
from magpylib._src.obj_classes.target_meshing import _target_mesh_triangle_current
from magpylib._src.style import CurrentSheetStyle


class TriangleStrip(BaseCurrent, BaseTarget, BaseDipoleMoment):
    """Current flowing in straight lines along a ribbon made of adjacent triangles.

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
    vertices : None | array-like, shape (n, 3) or (p, n, 3), default None
        The current flows along a band that consists of triangles {T1, T2, ...}
        defined by the vertices {V1, V2, V3, V4, ...} as T1 = (V1, V2, V3),
        T2 = (V2, V3, V4), ... The vertices are given in units (m) in the local
        object coordinates (move/rotate with object). At least three vertices
        must be given, which define the first triangle.
    current : float | array-like, shape (p,), default None
        Total current flowing through the strip in units (A). It flows in the
        direction V1→V3 in the first triangle, V2→V4 in the second, ...
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
    current : None | float | ndarray, shape (p,)
        Same as constructor parameter ``current``.
    meshing : None | int
        Same as constructor parameter ``meshing``.
    centroid : ndarray, shape (3,) or (p, 3)
        Read-only. Object centroid computed via mean of vertices in units (m)
        in global coordinates.
    dipole_moment : ndarray, shape (3,) or (p, 3)
        Read-only. Object dipole moment (A·m²) in local object coordinates. Can
        only be computed for a closed loop.
    parent : None | Collection
        Parent collection of the object.
    style : CurrentSheetStyle
        Object style. See CurrentSheetStyle for details.

    Notes
    -----
    Returns (0, 0, 0) on a sheet.

    Examples
    --------
    ``TriangleStrip`` objects are magnetic field sources. In this example we compute
    the H-field in (A/m) of a square current sheet (two triangles) with 1 A current
    at the observer position (1, 1, 1) cm:

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> src = magpy.current.TriangleStrip(
    ...    current=1,
    ...    vertices=((0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)),
    ... )
    >>> H = src.getH((0.01, 0.01, 0.01))
    >>> with np.printoptions(precision=3):
    ...     print(H)
    [ 0.026 -0.307 -0.371]
    """

    # pylint: disable=dangerous-default-value
    _field_func = staticmethod(_BHJM_current_tristrip)
    _force_type = "current"
    _field_func_kwargs_ndim: ClassVar[dict[str, int]] = {
        "current": 1,
        "vertices": 3,
    }
    _path_properties = ("vertices",)  # also inherits from parent class
    get_trace = make_TriangleStrip
    _style_class = CurrentSheetStyle

    def __init__(
        self,
        current=None,
        vertices=None,
        position=(0, 0, 0),
        orientation=None,
        meshing=None,
        *,
        style=None,
        **kwargs,
    ):
        # init inheritance
        super().__init__(
            position,
            orientation,
            current=current,
            vertices=vertices,
            style=style,
            **kwargs,
        )
        BaseTarget.__init__(self, meshing)

    # property getters and setters
    @property
    def vertices(self):
        """Triangle strip vertices in local object coordinates."""
        return (
            None
            if self._vertices is None
            else self._vertices[0]
            if len(self._vertices) == 1
            else self._vertices
        )

    @vertices.setter
    def vertices(self, vert):
        """Set triangle strip vertices.

        Parameters
        ----------
        vert : array-like, shape (n, 3) or (p, n, 3)
            Vertices in local object coordinates in units (m). At least three
            vertices must be provided.
        """
        self._vertices = check_format_input_vertices(vert, minlength=3)

    @property
    def _default_style_description(self):
        """Default style description text"""
        if self.vertices is None:
            return "no vertices"
        return super()._default_style_description

    # Methods
    def _get_centroid(self, squeeze=True):
        """Centroid of object in units (m)."""
        if self._vertices is None:
            return self._position

        # Compute mean for each path along vertex axis: shape (p, 3)
        centroids = np.mean(self._vertices, axis=1)  # mean over vertices dimension

        # Add position: shape (p, 3) + (p, 3)
        centroids = centroids + self._position

        if squeeze and len(centroids) == 1:
            return centroids[0]
        return centroids

    def _get_dipole_moment(self, squeeze=True):
        """Magnetic moment of object in units (A*m²)."""
        # test init
        if self._vertices is None or self._current is None:
            return np.zeros_like(self._position)

        # Check if all paths are closed (first two and last two vertices identical)
        # Shape: _vertices is (p, n, 3)
        first_two = self._vertices[:, :2, :]  # shape (p, 2, 3)
        last_two = self._vertices[:, -2:, :]  # shape (p, 2, 3)
        is_closed = np.all(first_two == last_two, axis=(1, 2))  # shape (p,)

        # Check if any path is not closed
        if np.any(~is_closed):
            invalid_idx = np.where(~is_closed)[0][0]
            msg = (
                f"Cannot compute dipole moment of {self} at path index {invalid_idx}. "
                "Dipole moment is only defined for closed CurrentStrip where first two "
                "and last two vertices are identical."
            )
            raise ValueError(msg)

        # Create triangles: shape (p, n-2, 3, 3) where last two dims are (vertices, coords)
        # Each triangle has vertices [i, i+1, i+2]
        trias = np.stack(
            [
                self._vertices[:, :-2, :],  # shape (p, n-2, 3)
                self._vertices[:, 1:-1, :],  # shape (p, n-2, 3)
                self._vertices[:, 2:, :],  # shape (p, n-2, 3)
            ],
            axis=2,
        )  # shape (p, n-2, 3, 3)

        # Compute centroids: mean of three vertices, shape (p, n-2, 3)
        centroids = np.mean(trias, axis=2)

        # Compute areas: 0.5 * ||(v1-v0) x (v2-v0)||, shape (p, n-2)
        cross_products = np.cross(
            trias[:, :, 1, :] - trias[:, :, 0, :],
            trias[:, :, 2, :] - trias[:, :, 0, :],
        )
        areas = 0.5 * np.linalg.norm(cross_products, axis=2)  # shape (p, n-2)

        # Create current density input
        v1 = trias[:, :, 1, :] - trias[:, :, 0, :]  # shape (p, n-2, 3)
        v2 = trias[:, :, 2, :] - trias[:, :, 0, :]  # shape (p, n-2, 3)
        v1v1 = np.sum(v1 * v1, axis=2)  # shape (p, n-2)
        v2v2 = np.sum(v2 * v2, axis=2)  # shape (p, n-2)
        v1v2 = np.sum(v1 * v2, axis=2)  # shape (p, n-2)

        # Initialize current densities: shape (p, n-2, 3)
        curr_densities = np.zeros_like(v2)

        # Catch two times the same vertex in one triangle, and set CD to zero there
        mask = (v2v2 != 0) & (v1v1 != 0)  # shape (p, n-2)

        # Compute height for valid triangles
        h = np.zeros_like(v1v1)
        h[mask] = np.sqrt(v1v1[mask] - (v1v2[mask] ** 2 / v2v2[mask]))

        # Expand current for broadcasting: shape (p, 1)
        current_expanded = self._current[:, np.newaxis]

        # Compute current densities where mask is True
        # Shape calculations: v2[mask] has shape (k, 3) where k = number of True in mask
        # We need to properly broadcast current for each path
        denom = np.zeros_like(v2v2)
        denom[mask] = (
            np.sqrt(v2v2[mask])
            * h[mask]
            / np.broadcast_to(current_expanded, v2v2.shape)[mask]
        )

        # Avoid division by zero
        valid = mask & (denom != 0)
        curr_densities[valid] = v2[valid] / denom[valid, np.newaxis]

        # Compute dipole moment: A / 2 * sum over triangles of (cent x curr_density)
        # Shape: areas[:, :, np.newaxis] is (p, n-2, 1)
        # cross(centroids, curr_densities) is (p, n-2, 3)
        dipole_moments = np.sum(
            areas[:, :, np.newaxis] / 2 * np.cross(centroids, curr_densities), axis=1
        )  # shape (p, 3)

        if squeeze and len(dipole_moments) == 1:
            return dipole_moments[0]
        return dipole_moments

    def _generate_mesh(self):
        """Generate mesh for force computation."""
        # TODO with path: suppress all triangles in computation that are ~1e-9 times
        # smaller than the total mesh surface, not sure how to handle this since the
        # areas are changing along path
        verts, curr = self._vertices, self._current
        # verts has shape (p, n, 3) where p is path dimension, n is number of vertices
        # Create triangles for each path: shape (p, n-2, 3, 3)
        # Vectorized triangle creation using advanced indexing
        triangles = np.stack(
            [
                verts[:, :-2, :],  # First vertex of each triangle: shape (p, n-2, 3)
                verts[:, 1:-1, :],  # Second vertex of each triangle: shape (p, n-2, 3)
                verts[:, 2:, :],  # Third vertex of each triangle: shape (p, n-2, 3)
            ],
            axis=2,
        )  # Final shape: (p, n-2, 3, 3)

        # sideA and sideB shape: (p, n-2, 3)
        sideA = triangles[:, :, 1] - triangles[:, :, 0]
        sideB = triangles[:, :, 2] - triangles[:, :, 0]

        # sideAA, sideBB, sideAB shape: (p, n-2)
        sideAA = np.sum(sideA * sideA, axis=2)
        sideBB = np.sum(sideB * sideB, axis=2)
        sideAB = np.sum(sideA * sideB, axis=2)
        area2 = sideAA * sideBB - sideAB * sideAB

        # compute current densities
        base_length = np.linalg.norm(sideB, axis=2)
        height = np.sqrt(area2) / base_length

        # curr shape: (p,) needs to be broadcast to match triangles shape (p, n-2)
        curr_expanded = np.repeat(curr[:, np.newaxis], triangles.shape[1], axis=1)

        # Current densities shape: (p, n-2, 3)
        cds = (
            sideB
            / base_length[:, :, np.newaxis]
            * curr_expanded[:, :, np.newaxis]
            / height[:, :, np.newaxis]
        )

        return _target_mesh_triangle_current(
            triangles,
            cds,
            self.meshing,
        )

    def _validate_meshing(self, value):
        """Makes only sense with at least n_mesh = n_faces."""
        if isinstance(value, int) and value >= len(self.vertices) - 2:
            pass
        else:
            msg = (
                f"Input meshing of {self} must be an integer >= number of faces; "
                f"instead received {value}."
            )
            raise ValueError(msg)

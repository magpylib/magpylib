"""Polyline current class code"""

# pylint: disable=too-many-positional-arguments

import warnings
from typing import ClassVar

import numpy as np

from magpylib._src.display.traces_core import make_Polyline
from magpylib._src.fields.field_BH_polyline import _current_vertices_field
from magpylib._src.input_checks import check_format_input_vertices
from magpylib._src.obj_classes.class_BaseExcitations import BaseCurrent
from magpylib._src.obj_classes.class_BaseProperties import BaseDipoleMoment
from magpylib._src.obj_classes.class_BaseTarget import BaseTarget
from magpylib._src.obj_classes.target_meshing import _target_mesh_polyline


class Polyline(BaseCurrent, BaseTarget, BaseDipoleMoment):
    """Line current flowing in straight paths from vertex to vertex.

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
        Current flows along the vertices in units (m) in the local object coordinates. At
        least two vertices must be given.
    current : float | array-like, shape (p,), default None
        Electrical current (A).
    meshing : int | None, default None
        Mesh fineness for force computation. Must be a positive integer at least the
        number of segments. Each segment gets one mesh point at its center. All
        remaining mesh points are distributed evenly along the polyline.
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
    style : CurrentStyle
        Object style. See CurrentStyle for details.

    Notes
    -----
    Returns (0, 0, 0) on the line segments.

    Examples
    --------
    ``Polyline`` objects are magnetic field sources. In this example we compute the
    H-field (A/m) of a square-shaped line current with 1 A at the observer position
    (1, 1, 1) (cm):

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> src = magpy.current.Polyline(
    ...     current=1,
    ...     vertices=((0.01, 0, 0), (0, 0.01, 0), (-0.01, 0, 0), (0, -0.01, 0), (0.01, 0, 0)),
    ... )
    >>> H = src.getH((0.01, 0.01, 0.01))
    >>> with np.printoptions(precision=3):
    ...     print(H)
    [3.161 3.161 0.767]
    """

    # pylint: disable=dangerous-default-value
    _field_func = staticmethod(_current_vertices_field)
    _force_type = "current"
    _field_func_kwargs_ndim: ClassVar[dict[str, int]] = {
        "current": 1,
        "vertices": 3,
        "segment_start": 2,
        "segment_end": 2,
    }
    _path_properties = ("vertices",)  # also inherits from parent class
    get_trace = make_Polyline

    def __init__(
        self,
        current=None,
        vertices=None,
        position=(0, 0, 0),
        orientation=None,
        meshing=None,
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

        # Initialize BaseTarget
        BaseTarget.__init__(self, meshing)

    # Properties
    @property
    def vertices(self):
        """Polyline vertices.

        The current flows along the vertices which are given in units (m) in the
        local object coordinates (move/rotate with object). At least two vertices
        must be given.
        """
        return (
            None
            if self._vertices is None
            else self._vertices[0]
            if len(self._vertices) == 1
            else self._vertices
        )

    @vertices.setter
    def vertices(self, vert):
        """Set polyline vertices.

        Parameters
        ----------
        vert : None | array-like, shape (n, 3) or (p, n, 3)
            Vertex list (m) in local object coordinates. At least two vertices
            must be given.
        """
        self._vertices = check_format_input_vertices(vert)

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

        # Check if all paths are closed polylines
        first_vertices = self._vertices[:, 0, :]  # shape (p, 3)
        last_vertices = self._vertices[:, -1, :]  # shape (p, 3)
        is_closed = np.all(first_vertices == last_vertices, axis=1)  # shape (p,)

        # Check if any path is not closed and has more than 1 vertex
        has_multiple_vertices = self._vertices.shape[1] > 1
        invalid_paths = has_multiple_vertices & ~is_closed

        if np.any(invalid_paths):
            invalid_idx = np.where(invalid_paths)[0][0]
            msg = (
                f"Cannot compute dipole moment of {self} at path index {invalid_idx}. "
                "Dipole moment is only defined for closed Polylines "
                "(first and last vertex must be identical)."
            )
            raise ValueError(msg)

        # Compute dipole moments for all paths (vectorized)
        # Prepare vertices pairs: v[:-1] and v[1:] for cross product
        # Shape: (p, n-1, 3) where p is path length, n is number of vertices
        v_start = self._vertices[:, :-1, :]  # shape (p, n-1, 3)
        v_end = self._vertices[:, 1:, :]  # shape (p, n-1, 3)

        # Cross product along segments: shape (p, n-1, 3)
        crosses = np.cross(v_start, v_end)

        # Sum along segments axis: shape (p, 3)
        cross_sums = np.sum(crosses, axis=1)

        # Multiply by current/2: shape (p, 3)
        dipole_moments = self._current[:, np.newaxis] / 2 * cross_sums
        if squeeze and len(dipole_moments) == 1:
            return dipole_moments[0]
        return dipole_moments

    def _generate_mesh(self):
        """Generate mesh for force computation."""
        # Tests in getFT ensure that meshing, dimension and excitation are set

        # Special special case: fewer points than segments, cannot be caught in
        #    meshing setter because vertices might not have been set yet
        n_points = self.meshing
        n_segments = len(self._vertices[0]) - 1
        if self.meshing < n_segments:
            msg = (
                f"Input meshing of {self} must be an integer > number of Polyline "
                f"segments ({n_segments}); instead received {self.meshing}. "
                "Setting one point per segment in computation."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            n_points = n_segments

        return _target_mesh_polyline(self._vertices, self._current, n_points)

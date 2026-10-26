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
from magpylib._src.obj_classes.target_meshing import generate_mesh_polyline


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
        style=None,
        **kwargs,
    ):
        super().__init__(
            position,
            orientation,
            current=current,
            vertices=vertices,
            style=style,
            **kwargs,
        )

    # Properties
    @property
    def vertices(self):
        """Polyline vertices.

        The current flows along the vertices which are given in units (m) in the
        local object coordinates (move/rotate with object). At least two vertices
        must be given.
        """
        return self._squeeze_path_property(self._vertices)

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
            centroid = np.array([0.0, 0.0, 0.0])
            orientation, position = self._orientation, self._position
        else:
            # vertices are local coordinates: pair each step's vertices with its
            # pose so the centroid turns with the object
            synced = self._sync_path_lengths(("vertices", "position", "orientation"))
            centroid = np.mean(synced["vertices"], axis=1)  # (p, 3)
            orientation, position = synced["orientation"], synced["position"]
        result = orientation.apply(centroid) + position
        if squeeze and len(result) == 1:
            return result[0]
        return result

    def _get_dipole_moment(self, squeeze=True):
        """Magnetic moment of object in units (A*m²)."""
        # test init
        if self._vertices is None or self._current is None:
            dip = np.zeros_like(self._position)
            if squeeze and len(dip) == 1:
                return dip[0]
            return dip

        synced = self._sync_path_lengths(("vertices", "current"))
        vertices, current = synced["vertices"], synced["current"]

        # Check if all paths are closed polylines
        first_vertices = vertices[:, 0, :]  # shape (p, 3)
        last_vertices = vertices[:, -1, :]  # shape (p, 3)
        is_closed = np.all(first_vertices == last_vertices, axis=1)  # shape (p,)

        # Check if any path is not closed and has more than 1 vertex
        has_multiple_vertices = vertices.shape[1] > 1
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
        v_start = vertices[:, :-1, :]  # shape (p, n-1, 3)
        v_end = vertices[:, 1:, :]  # shape (p, n-1, 3)

        # Cross product along segments: shape (p, n-1, 3)
        crosses = np.cross(v_start, v_end)

        # Sum along segments axis: shape (p, 3)
        cross_sums = np.sum(crosses, axis=1)

        # Multiply by current/2: shape (p, 3)
        dipole_moments = current[:, np.newaxis] / 2 * cross_sums
        if squeeze and len(dipole_moments) == 1:
            return dipole_moments[0]
        return dipole_moments

    def _generate_mesh(self, meshing):
        """Generate mesh for force computation."""
        synced = self._sync_path_lengths(("vertices", "current"))
        verts, curr = synced["vertices"], synced["current"]

        n_points = meshing
        n_segments = len(verts[0]) - 1
        if meshing < n_segments:
            msg = (
                f"Input meshing of {self} must be an integer > number of Polyline "
                f"segments ({n_segments}); instead received {meshing}. "
                "Setting one point per segment in computation."
            )
            warnings.warn(msg, UserWarning, stacklevel=4)
            n_points = n_segments

        return generate_mesh_polyline(verts, curr, n_points)

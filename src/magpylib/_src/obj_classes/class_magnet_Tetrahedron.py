"""Magnet Tetrahedron class code"""

# pylint: disable=too-many-positional-arguments

from typing import ClassVar

import numpy as np

from magpylib._src.display.traces_core import make_Tetrahedron
from magpylib._src.fields.field_BH_tetrahedron import _BHJM_magnet_tetrahedron
from magpylib._src.input_checks import check_format_input_numeric
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet
from magpylib._src.obj_classes.class_BaseProperties import (
    BaseDipoleMoment,
    BaseVolume,
)
from magpylib._src.obj_classes.class_BaseTarget import BaseTarget
from magpylib._src.obj_classes.target_meshing import generate_mesh_tetrahedron


class Tetrahedron(BaseMagnet, BaseTarget, BaseVolume, BaseDipoleMoment):
    """Tetrahedron magnet with homogeneous magnetization.

    Can be used as ``sources`` input for magnetic field computation and ``target``
    input for force computation.

    When ``position=(0, 0, 0)`` and ``orientation=None`` the Tetrahedron vertex coordinates
    are the same as in the global coordinate system. The geometric center of the Tetrahedron
    is determined by its vertices and is not necessarily located in the origin. It can be
    computed with the ``centroid`` property.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    position : array-like, shape (3,) or (p, 3), default (0, 0, 0)
        Object position(s) in global coordinates in units (m). ``position`` and
        ``orientation`` attributes define the object path. When setting ``vertices``,
        the initial position is set to the centroid.
    orientation : Rotation | None, default None
        Object orientation(s) in global coordinates as a scipy Rotation. Rotation can
        have length 1 or p. ``None`` generates a unit-rotation.
    vertices : None | array-like, shape (4, 3) or (p, 4, 3), default None
        Vertices ``[(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]`` in the
        local object coordinates.
    polarization : None | array-like, shape (3,) or (p, 3), default None
        Magnetic polarization vector J = mu0*M in units (T), given in the
        local object coordinates. Sets also ``magnetization``.
    magnetization : None | array-like, shape (3,) or (p, 3), default None
        Magnetization vector M = J/mu0 in units (A/m), given in the local
        object coordinates. Sets also ``polarization``.
    meshing : int | None, default None
        Mesh fineness for force computation. Must be a positive integer specifying
        the target mesh size.
    style : dict | None, default None
        Style dictionary. Can also be provided via style underscore magic, e.g.
        ``style_color='red'``.

    Attributes
    ----------
    position : ndarray, shape (3,) or (p, 3)
        Same as constructor parameter ``position``.
    orientation : Rotation
        Same as constructor parameter ``orientation``.
    vertices : ndarray, shape (4, 3) or (p, 4, 3)
        Same as constructor parameter ``vertices``.
    polarization : None | ndarray, shape (3,) or (p, 3)
        Same as constructor parameter ``polarization``.
    magnetization : None | ndarray, shape (3,) or (p, 3)
        Same as constructor parameter ``magnetization``.
    meshing : int | None
        Same as constructor parameter ``meshing``.
    centroid : ndarray, shape (3,) or (p, 3)
        Read-only. Object centroid in units (m) in global coordinates.
    dipole_moment : ndarray, shape (3,) or (p, 3)
        Read-only. Object dipole moment (A·m²) in local object coordinates.
    volume : float | ndarray, shape (p,)
        Read-only. Object physical volume in units (m³).
    parent : None | Collection
        Parent collection of the object.
    style : MagnetStyle
        Object style. See MagnetStyle for details.

    Notes
    -----
    Returns (0, 0, 0) on corners.

    Examples
    --------
    ``Tetrahedron`` magnets are magnetic field sources. Below we compute the H-field in (A/m) of a
    tetrahedron magnet with polarization (0.1, 0.2, 0.3) in units (T) and dimensions defined
    through the vertices (0, 0, 0), (0.01, 0, 0), (0, 0.01, 0) and (0, 0, 0.01) (m)
    at the observer position (0.01, 0.01, 0.01) (m):

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> verts = [(0, 0, 0), (0.01, 0, 0), (0, 0.01, 0), (0, 0, 0.01)]
    >>> src = magpy.magnet.Tetrahedron(polarization=(0.1, 0.2, 0.3), vertices=verts)
    >>> H = src.getH((0.01, 0.01, 0.01))
    >>> with np.printoptions(precision=3):
    ...     print(H)
    [2070.898 1656.718 1242.539]
    """

    _field_func = staticmethod(_BHJM_magnet_tetrahedron)
    _force_type = "magnet"
    _field_func_kwargs_ndim: ClassVar[dict[str, int]] = {
        "polarization": 1,
        "vertices": 3,
    }
    _path_properties = ("vertices",)  # also inherits from parent class
    get_trace = make_Tetrahedron

    def __init__(
        self,
        position=(0, 0, 0),
        orientation=None,
        vertices=None,
        polarization=None,
        magnetization=None,
        meshing=None,
        style=None,
        **kwargs,
    ):
        super().__init__(
            position,
            orientation,
            magnetization=magnetization,
            polarization=polarization,
            vertices=vertices,
            style=style,
            **kwargs,
        )

        BaseTarget.__init__(self, meshing)

    # Properties
    @property
    def vertices(self):
        """Tetrahedron vertices in local object coordinates."""
        return self._squeeze_path_property(self._vertices)

    @vertices.setter
    def vertices(self, dim):
        """Set tetrahedron vertices.

        Parameters
        ----------
        dim : None or array-like, shape (4, 3) or (p, 4, 3)
            Vertices in local object coordinates in units (m).
        """
        self._vertices = check_format_input_numeric(
            dim,
            dtype=float,
            shapes=((4, 3), (None, 4, 3)),
            name="Tetrahedron.vertices",
            allow_None=True,
            reshape=(-1, 4, 3),
        )

    @property
    def _default_style_description(self):
        """Default style description text"""
        if self.vertices is None:
            return "no vertices"
        return ""

    # Methods
    def _get_volume(self, squeeze=True):
        """Volume of object in units (m³)."""
        if self._vertices is None:
            return 0.0 if squeeze else np.array([0.0])

        verts = self._vertices  # shape (p, 4, 3)
        # v1, v2, v3 shapes: (p, 3)
        v1 = verts[:, 1] - verts[:, 0]
        v2 = verts[:, 2] - verts[:, 0]
        v3 = verts[:, 3] - verts[:, 0]

        # Build per-path 3x3 matrices: shape (p, 3, 3)
        matrices = np.stack([v1, v2, v3], axis=1)
        dets = np.linalg.det(matrices)
        vols = np.abs(dets) / 6.0
        if squeeze and len(vols) == 1:
            return float(vols[0])
        return vols

    def _get_centroid(self, squeeze=True):
        """Centroid of object in units (m)."""
        if self._vertices is None:
            centroid = np.array([0.0, 0.0, 0.0])
            orientation, position = self._orientation, self._position
        else:
            # sync vertices with position/orientation so the per-step centroid
            # stays paired with its pose under the lazy-storage path model
            synced = self._sync_path_lengths(("vertices", "position", "orientation"))
            centroid = np.mean(synced["vertices"], axis=1)  # (p,3)
            orientation, position = synced["orientation"], synced["position"]
        result = orientation.apply(centroid) + position
        if squeeze and len(result) == 1:
            return result[0]
        return result

    def _get_dipole_moment(self, squeeze=True):
        """Magnetic moment of object in units (A*m²)."""
        if self._magnetization is None or self._vertices is None:
            dip = np.zeros_like(self._position)
            if squeeze and len(dip) == 1:
                return dip[0]
            return dip

        synced = self._sync_path_lengths(("vertices", "magnetization"))
        verts = synced["vertices"]  # shape (p, 4, 3)
        matrices = np.stack(
            [
                verts[:, 1] - verts[:, 0],
                verts[:, 2] - verts[:, 0],
                verts[:, 3] - verts[:, 0],
            ],
            axis=1,
        )
        vols = np.abs(np.linalg.det(matrices)) / 6.0
        dipoles = synced["magnetization"] * vols[:, np.newaxis]
        if squeeze and len(dipoles) == 1:
            return dipoles[0]
        return dipoles

    def _generate_mesh(self):
        """Generate mesh for force computation by delegating to target mesher."""
        synced = self._sync_path_lengths(("vertices", "magnetization"))
        return generate_mesh_tetrahedron(
            synced["vertices"], synced["magnetization"], self.meshing
        )

"""Magnet Tetrahedron class code"""

# pylint: disable=too-many-positional-arguments

from typing import ClassVar

import numpy as np

from magpylib._src.display.traces_core import make_Tetrahedron
from magpylib._src.fields.field_BH_tetrahedron import _BHJM_magnet_tetrahedron
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet
from magpylib._src.obj_classes.class_BaseProperties import (
    BaseDipoleMoment,
    BaseVolume,
)
from magpylib._src.obj_classes.class_BaseTarget import BaseTarget
from magpylib._src.obj_classes.target_meshing import _target_mesh_tetrahedron


class Tetrahedron(BaseMagnet, BaseTarget, BaseVolume, BaseDipoleMoment):
    """Tetrahedron magnet with homogeneous magnetization.

    Can be used as ``sources`` input for magnetic field computation and ``target``
    input for force computation.

    When ``position=(0, 0, 0)`` and ``orientation=None`` the Tetrahedron vertex coordinates
    are the same as in the global coordinate system. The geometric center of the Tetrahedron
    is determined by its vertices and is not necessarily located in the origin. It can be
    computed with the ``barycenter`` property.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    position : array-like, shape (3,) or (p, 3), default (0, 0, 0)
        Object position(s) in global coordinates in units (m). ``position`` and
        ``orientation`` attributes define the object path. When setting ``vertices``,
        the initial position is set to the barycenter.
    orientation : Rotation | None, default None
        Object orientation(s) in global coordinates as a scipy Rotation. Rotation can
        have length 1 or p. ``None`` generates a unit-rotation.
    vertices : None | array-like, shape (4, 3), default None
        Vertices ``[(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]`` in the
        local object coordinates.
    polarization : None | array-like, shape (3,), default None
        Magnetic polarization vector J = mu0*M in units (T), given in the
        local object coordinates. Sets also ``magnetization``.
    magnetization : None | array-like, shape (3,), default None
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
    vertices : ndarray, shape (4, 3)
        Same as constructor parameter ``vertices``.
    polarization : None | ndarray, shape (3,)
        Same as constructor parameter ``polarization``.
    magnetization : None | ndarray, shape (3,)
        Same as constructor parameter ``magnetization``.
    meshing : int | None
        Same as constructor parameter ``meshing``.
    centroid : ndarray, shape (3,) or (p, 3)
        Read-only. Object centroid in units (m) in global coordinates.
        Can be a path.
    dipole_moment : ndarray, shape (3,)
        Read-only. Object dipole moment (A·m²) in local object coordinates.
    volume : float
        Read-only. Object physical volume in units (m³).
    parent : Collection or None
        Parent collection of the object.
    style : dict
        Style dictionary defining visual properties.
    barycenter : ndarray, shape (3,)
        Read-only. Geometric barycenter (= center of mass) of the object.

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
        # instance attributes
        self.vertices = vertices

        # init inheritance
        super().__init__(
            position, orientation, magnetization, polarization, style, **kwargs
        )

        # Initialize BaseTarget
        BaseTarget.__init__(self, meshing)

    # Properties
    @property
    def vertices(self):
        """Tetrahedron vertices in local object coordinates."""
        return self._vertices

    @vertices.setter
    def vertices(self, dim):
        """Set tetrahedron vertices.

        Parameters
        ----------
        dim : None or array-like, shape (4, 3)
            Vertices in local object coordinates in units (m).
        """
        self._vertices = check_format_input_vector(
            dim,
            dims=(2,),
            shape_m1=3,
            length=4,
            sig_name="Tetrahedron.vertices",
            sig_type="array-like (list, tuple, ndarray) of shape (4, 3)",
            allow_None=True,
        )

    @property
    def _barycenter(self):
        """Object barycenter."""
        return self._get_barycenter(self._position, self._orientation, self.vertices)

    @property
    def barycenter(self):
        """Object barycenter."""
        return np.squeeze(self._barycenter)

    @property
    def _default_style_description(self):
        """Default style description text"""
        if self.vertices is None:
            return "no vertices"
        return ""

    # Methods
    def _get_volume(self):
        """Volume of object in units (m³)."""
        if self.vertices is None:
            return 0.0

        # Tetrahedron volume formula: |det(B-A, C-A, D-A)| / 6
        vertices = self.vertices
        v1 = vertices[1] - vertices[0]  # B - A
        v2 = vertices[2] - vertices[0]  # C - A
        v3 = vertices[3] - vertices[0]  # D - A

        # Create 3x3 matrix and compute determinant
        matrix = np.column_stack([v1, v2, v3])
        return abs(np.linalg.det(matrix)) / 6.0

    def _get_centroid(self, squeeze=True):
        """Centroid of object in units (m)."""
        if squeeze:
            return self.barycenter
        return self._barycenter

    def _get_dipole_moment(self):
        """Magnetic moment of object in units (A*m²)."""
        # test init
        if self.magnetization is None or self.vertices is None:
            return np.array((0.0, 0.0, 0.0))
        return self.magnetization * self.volume

    def _generate_mesh(self):
        """Generate mesh for force computation."""
        return _target_mesh_tetrahedron(self.meshing, self.vertices, self.magnetization)

    # Static methods
    @staticmethod
    def _get_barycenter(position, orientation, vertices):
        """Returns the barycenter of a tetrahedron."""
        centroid = (
            np.array([0.0, 0.0, 0.0]) if vertices is None else np.mean(vertices, axis=0)
        )
        return orientation.apply(centroid) + position

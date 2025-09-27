"""Magnet Cylinder class code"""

# pylint: disable=too-many-positional-arguments

from typing import ClassVar

import numpy as np

from magpylib._src.display.traces_core import make_CylinderSegment
from magpylib._src.fields.field_BH_cylinder_segment import (
    _BHJM_cylinder_segment_internal,
)
from magpylib._src.input_checks import check_format_input_cylinder_segment
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet
from magpylib._src.obj_classes.class_BaseProperties import (
    BaseDipoleMoment,
    BaseVolume,
)
from magpylib._src.obj_classes.class_BaseTarget import BaseTarget
from magpylib._src.obj_classes.target_meshing import _target_mesh_cylinder
from magpylib._src.utility import unit_prefix


class CylinderSegment(BaseMagnet, BaseTarget, BaseVolume, BaseDipoleMoment):
    """Cylinder segment (ring-section) magnet with homogeneous magnetization.

    Can be used as ``sources`` input for magnetic field computation and ``target``
    input for force computation.

    When ``position=(0, 0, 0)`` and ``orientation=None`` the geometric center of the
    cylinder lies in the origin of the global coordinate system and the cylinder axis
    coincides with the global z-axis. Section angle 0 corresponds to an x-z plane
    section of the cylinder.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    position : array-like, shape (3,) or (p, 3), default (0, 0, 0)
        Object position(s) in global coordinates in units (m). ``position`` and
        ``orientation`` attributes define the object path.
    orientation : Rotation | None, default None
        Object orientation(s) in global coordinates as a scipy Rotation. Rotation can
        have length 1 or p. ``None`` generates a unit-rotation.
    dimension : None | array-like, shape (5,), default None
        Cylinder segment size (r1, r2, h, phi1, phi2) where r1 < r2 are inner
        and outer radii in units (m), phi1 < phi2 are section angles in units (deg),
        and h is the height in units (m).
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
    polarization : None | ndarray, shape (3,)
        Same as constructor parameter ``polarization``.
    magnetization : None | ndarray, shape (3,)
        Same as constructor parameter ``magnetization``.
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
    Returns (0, 0, 0) on surface, edges, and corners.

    Examples
    --------
    ``CylinderSegment`` magnets are magnetic field sources. In this example we compute the
    H-field in (A/m) of such a cylinder segment magnet with polarization
    (0.1, 0.2, 0.3) (T), inner radius 1 (cm), outer radius 2 (cm),
    height 1 (cm), and section angles 0 and 45 (deg) at the observer
    position (2, 2, 2) (cm):

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> src = magpy.magnet.CylinderSegment(
    ...     polarization=(0.1, 0.2, 0.3),
    ...     dimension=(0.01, 0.02, 0.01, 0.0, 45.0),
    ... )
    >>> H = src.getH((0.02, 0.02, 0.02))
    >>> with np.printoptions(precision=3):
    ...     print(H)
    [ 807.847 1934.228 2741.168]
    """

    _field_func = staticmethod(_BHJM_cylinder_segment_internal)
    _force_type = "magnet"
    _field_func_kwargs_ndim: ClassVar[dict[str, int]] = {
        "polarization": 2,
        "dimension": 2,
    }
    get_trace = make_CylinderSegment

    def __init__(
        self,
        position=(0, 0, 0),
        orientation=None,
        dimension=None,
        polarization=None,
        magnetization=None,
        meshing=None,
        style=None,
        **kwargs,
    ):
        # instance attributes
        self.dimension = dimension

        # init inheritance
        super().__init__(
            position, orientation, magnetization, polarization, style, **kwargs
        )

        # Initialize BaseTarget
        BaseTarget.__init__(self, meshing)

    # property getters and setters
    @property
    def dimension(self):
        """Cylinder segment size (r1, r2, h, phi1, phi2).

        r1 < r2 denote inner and outer radii in units (m), phi1 < phi2 the
        section angles in units (deg), and h the height in units (m).
        """
        return self._dimension

    @dimension.setter
    def dimension(self, dim):
        """Set cylinder segment size.

        Parameters
        ----------
        dim : None or array-like, shape (5,)
            Size (r1, r2, h, phi1, phi2) where r1 < r2 are radii in (m),
            phi1 < phi2 are section angles in (deg), and h is the height (m).
        """
        self._dimension = check_format_input_cylinder_segment(dim)

    @property
    def _barycenter(self):
        """Object barycenter."""
        return self._get_barycenter(self._position, self._orientation, self.dimension)

    @property
    def barycenter(self):
        """Object barycenter."""
        return np.squeeze(self._barycenter)

    @property
    def _default_style_description(self):
        """Default style description text"""
        if self.dimension is None:
            return "no dimension"
        d = [unit_prefix(d) for d in self.dimension]
        return f"r={d[0]}m|{d[1]}m, h={d[2]}m, φ={d[3]}°|{d[4]}°"

    # Methods
    def _get_volume(self):
        """Volume of object in units (m³)."""
        if self.dimension is None:
            return 0.0

        r1, r2, h, phi1, phi2 = self.dimension
        return (r2**2 - r1**2) * np.pi * h * (phi2 - phi1) / 360

    def _get_centroid(self, squeeze=True):
        """Centroid of object in units (m)."""
        if squeeze:
            return self.barycenter
        return self._barycenter

    def _get_dipole_moment(self):
        """Magnetic moment of object in units (A*m²)."""
        # test init
        if self.magnetization is None or self.dimension is None:
            return np.array((0.0, 0.0, 0.0))
        return self.magnetization * self.volume

    def _generate_mesh(self):
        """Generate mesh for force computation."""
        # Tests in getFT ensure that meshing, dimension and excitation are set
        r1, r2, h, phi1, phi2 = self.dimension
        return _target_mesh_cylinder(
            r1, r2, h, phi1, phi2, self.meshing, self.magnetization
        )

    # Static methods
    @staticmethod
    def _get_barycenter(position, orientation, dimension):
        """Returns the barycenter of a cylinder segment.
        Input checks should make sure:
            -360 < phi1 < phi2 < 360 and 0 < r1 < r2
        """
        if dimension is None:
            centroid = np.array([0.0, 0.0, 0.0])
        else:
            r1, r2, _, phi1, phi2 = dimension
            alpha = np.deg2rad((phi2 - phi1) / 2)
            phi = np.deg2rad((phi1 + phi2) / 2)
            # get centroid x for unrotated annular sector
            centroid_x = (
                2 / 3 * np.sin(alpha) / alpha * (r2**3 - r1**3) / (r2**2 - r1**2)
            )
            # get centroid for rotated annular sector
            x, y, z = centroid_x * np.cos(phi), centroid_x * np.sin(phi), 0
            centroid = np.array([x, y, z])
        return orientation.apply(centroid) + position

"""Magnet Cylinder class code"""

# pylint: disable=too-many-positional-arguments

from typing import ClassVar

import numpy as np

from magpylib._src.display.traces_core import make_Cylinder
from magpylib._src.fields.field_BH_cylinder import _BHJM_magnet_cylinder
from magpylib._src.input_checks import check_format_input_numeric
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet
from magpylib._src.obj_classes.class_BaseProperties import (
    BaseDipoleMoment,
    BaseVolume,
)
from magpylib._src.obj_classes.class_BaseTarget import BaseTarget
from magpylib._src.obj_classes.target_meshing import generate_mesh_cylindersegment
from magpylib._src.utility import unit_prefix


class Cylinder(BaseMagnet, BaseTarget, BaseVolume, BaseDipoleMoment):
    """Cylinder magnet with homogeneous magnetization.

    Can be used as ``sources`` input for magnetic field computation and ``target``
    input for force computation.

    When ``position=(0, 0, 0)`` and ``orientation=None`` the geometric center of the
    cylinder lies in the origin of the global coordinate system and the cylinder
    axis coincides with the global z-axis.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    position : array-like, shape (3,) or (p, 3), default (0, 0, 0)
        Object position(s) in global coordinates in units (m). ``position`` and
        ``orientation`` attributes define the object path.
    orientation : Rotation | None, default None
        Object orientation(s) in global coordinates as a scipy Rotation. Rotation can
        have length 1 or p. ``None`` generates a unit-rotation.
    dimension : None | array-like, shape (2,) or (p, 2), default None
        Cylinder diameter and height ``(d, h)`` in units (m).
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
    dimension : None | ndarray, shape (2,) or (p, 2)
        Same as constructor parameter ``dimension``.
    polarization : None | ndarray, shape (3,) or (p, 3)
        Same as constructor parameter ``polarization``.
    magnetization : None | ndarray, shape (3,) or (p, 3)
        Same as constructor parameter ``magnetization``.
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
    Returns (0, 0, 0) on edges.

    Examples
    --------
    ``Cylinder`` magnets are magnetic field sources. Below we compute the H-field in (A/m) of a
    cylinder magnet with polarization ``(0.1, 0.2, 0.3)`` in units (T) and diameter and
    height ``0.01 m`` at the observer position ``(0.01, 0.01, 0.01)`` (m):

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> src = magpy.magnet.Cylinder(polarization=(0.1, 0.2, 0.3), dimension=(0.01, 0.01))
    >>> H = src.getH((0.01, 0.01, 0.01))
    >>> with np.printoptions(precision=3):
    ...     print(H)
    [4849.913 3883.178 2739.732]
    """

    _field_func = staticmethod(_BHJM_magnet_cylinder)
    _force_type = "magnet"
    _field_func_kwargs_ndim: ClassVar[dict[str, int]] = {
        "polarization": 2,
        "dimension": 2,
    }
    _path_properties = ("dimension",)  # also inherits from parent class
    get_trace = make_Cylinder

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
        super().__init__(
            position,
            orientation,
            magnetization=magnetization,
            polarization=polarization,
            dimension=dimension,
            style=style,
            **kwargs,
        )

        BaseTarget.__init__(self, meshing)

    # Properties
    @property
    def dimension(self):
        """Cylinder diameter and height ``(d, h)`` in units (m)."""
        return np.squeeze(self._dimension) if self._dimension is not None else None

    @dimension.setter
    def dimension(self, dim):
        """Set cylinder diameter and height.

        Parameters
        ----------
        dim : None or array-like, shape (2,) or (p, 2)
            Diameter and height ``(d, h)`` in units (m).
        """
        self._dimension = check_format_input_numeric(
            dim,
            dtype=float,
            shapes=((2,), (None, 2)),
            name="Cylinder.dimension",
            allow_None=True,
            reshape=(-1, 2),
            value_conditions=[("gt", 0, "all")],
        )

    @property
    def _default_style_description(self):
        """Default style description text"""
        if self.dimension is None:
            return "no dimension"
        # Handle path dimensions
        dims = self._dimension
        if len(dims) == 1 or np.all(dims == dims[0], axis=0).all():
            # Single dimension or all dimensions are the same
            d = [unit_prefix(v) for v in dims[0]]
            return f"D={d[0]}m, H={d[1]}m"
        # Multiple different dimensions - show range
        dmin, dmax = np.nanmin(dims, axis=0), np.nanmax(dims, axis=0)
        parts = []
        for i, label in enumerate(["D", "H"]):
            if np.allclose(dmin[i], dmax[i]):
                # No variation in this dimension
                val = unit_prefix(dmin[i])
                parts.append(f"{label}={val}m")
            else:
                # Show range
                vmin, vmax = unit_prefix(dmin[i]), unit_prefix(dmax[i])
                parts.append(f"{label}={vmin}m↔{vmax}m")
        return ", ".join(parts)

    # Methods
    def _get_volume(self, squeeze=True):
        """Volume of object in units of m³."""
        if self._dimension is None:
            return 0.0 if squeeze else np.array([0.0])

        dims = self._dimension  # Use internal (p, 2) shape
        d, h = dims[..., 0], dims[..., 1]
        vols = d**2 * np.pi * h / 4
        if squeeze and len(vols) == 1:
            return float(vols[0])
        return vols

    def _get_centroid(self, squeeze=True):
        """Centroid of object in units of m."""
        if squeeze:
            return self.position
        return self._position

    def _get_dipole_moment(self, squeeze=True):
        """Magnetic moment of object in units Am²."""
        if self._magnetization is None or self._dimension is None:
            dip = np.zeros_like(self._position)
            if squeeze and len(dip) == 1:
                return dip[0]
            return dip

        vols = self._get_volume(squeeze=False)
        dipoles = self._magnetization * vols[:, np.newaxis]

        if squeeze and len(dipoles) == 1:
            return dipoles[0]
        return dipoles

    def _generate_mesh(self):
        """Generate mesh for force computation."""
        # Tests in getFT ensure that meshing, dimension and excitation are set
        # Pass full path-enabled arrays (p, 2) and (p, 3)
        return generate_mesh_cylindersegment(
            np.zeros(self._dimension.shape[0]),  # r1 = 0 (solid cylinder)
            self._dimension[:, 0] / 2,  # r2 = d/2
            self._dimension[:, 1],  # h
            np.zeros(self._dimension.shape[0]),  # phi1 = 0
            np.full(self._dimension.shape[0], 360),  # phi2 = 360 (full cylinder)
            self._magnetization,
            self.meshing,
        )

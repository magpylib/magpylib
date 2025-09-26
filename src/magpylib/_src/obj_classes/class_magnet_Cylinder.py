"""Magnet Cylinder class code"""

# pylint: disable=too-many-positional-arguments

from typing import ClassVar

import numpy as np

from magpylib._src.display.traces_core import make_Cylinder
from magpylib._src.fields.field_BH_cylinder import _BHJM_magnet_cylinder
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet
from magpylib._src.obj_classes.class_BaseProperties import (
    BaseDipoleMoment,
    BaseVolume,
)
from magpylib._src.obj_classes.class_BaseTarget import BaseTarget
from magpylib._src.obj_classes.target_meshing import _target_mesh_cylinder
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
    dimension : None | array-like, shape (2,), default None
        Cylinder diameter and height ``(d, h)`` in units (m).
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
        # instance attributes
        self.dimension = dimension

        # init inheritance
        super().__init__(
            position, orientation, magnetization, polarization, style, **kwargs
        )

        # Initialize BaseTarget
        BaseTarget.__init__(self, meshing)

    # Properties
    @property
    def dimension(self):
        """Cylinder diameter and height ``(d, h)`` in units (m)."""
        return self._dimension

    @dimension.setter
    def dimension(self, dim):
        """Set cylinder diameter and height.

        Parameters
        ----------
        dim : None or array-like, shape (2,)
            Diameter and height ``(d, h)`` in units (m).
        """
        self._dimension = check_format_input_vector(
            dim,
            dims=(1,),
            shape_m1=2,
            sig_name="Cylinder.dimension",
            sig_type="array-like (list, tuple, ndarray) with shape (2,) with positive values",
            allow_None=True,
            forbid_negative0=True,
        )

    @property
    def _default_style_description(self):
        """Default style description text"""
        if self.dimension is None:
            return "no dimension"
        d = [unit_prefix(d) for d in self.dimension]
        return f"D={d[0]}m, H={d[1]}m"

    # Methods
    def _get_volume(self):
        """Volume of object in units of m³."""
        if self.dimension is None:
            return 0.0

        d, h = self.dimension
        return d**2 * np.pi * h / 4

    def _get_centroid(self, squeeze=True):
        """Centroid of object in units of m."""
        if squeeze:
            return self.position
        return self._position

    def _get_dipole_moment(self):
        """Magnetic moment of object in units Am²."""
        # test init
        if self.magnetization is None or self.dimension is None:
            return np.array((0.0, 0.0, 0.0))
        return self.magnetization * self.volume

    def _generate_mesh(self):
        """Generate mesh for force computation."""
        # Tests in getFT ensure that meshing, dimension and excitation are set
        d, h = self.dimension
        return _target_mesh_cylinder(
            0, d / 2, h, 0, 360, self.meshing, self.magnetization
        )

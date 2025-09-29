"""Magnet Sphere class code"""

# pylint: disable=too-many-positional-arguments

from typing import ClassVar

import numpy as np

from magpylib._src.display.traces_core import make_Sphere
from magpylib._src.fields.field_BH_sphere import _BHJM_magnet_sphere
from magpylib._src.input_checks import check_format_input_scalar
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet
from magpylib._src.obj_classes.class_BaseProperties import (
    BaseDipoleMoment,
    BaseVolume,
)
from magpylib._src.utility import unit_prefix


class Sphere(BaseMagnet, BaseVolume, BaseDipoleMoment):
    """Spherical magnet with homogeneous magnetization.

    Can be used as ``sources`` input for magnetic field computation and ``target``
    input for force computation. No ``meshing`` parameter is required.

    When ``position=(0, 0, 0)`` and ``orientation=None`` the sphere center is located
    in the origin of the global coordinate system.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    position : array-like, shape (3,) or (p, 3), default (0, 0, 0)
        Object position(s) in global coordinates in units (m). ``position`` and
        ``orientation`` attributes define the object path.
    orientation : Rotation | None, default None
        Object orientation(s) in global coordinates as a scipy Rotation. Rotation can
        have length 1 or p. ``None`` generates a unit-rotation.
    diameter : float | None, default None
        Diameter of the sphere in units (m).
    polarization : None | array-like, shape (3,), default None
        Magnetic polarization vector J = mu0*M in units (T), given in the
        local object coordinates. Sets also ``magnetization``.
    magnetization : None | array-like, shape (3,), default None
        Magnetization vector M = J/mu0 in units (A/m), given in the local
        object coordinates. Sets also ``polarization``.
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

    Examples
    --------
    ``Sphere`` objects are magnetic field sources. In this example we compute the H-field in (A/m)
    of a spherical magnet with polarization (0.1, 0.2, 0.3) in units (T) and diameter
    0.01 m at the observer position (0.01, 0.01, 0.01) (m):

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> src = magpy.magnet.Sphere(polarization=(0.1, 0.2, 0.3), diameter=0.01)
    >>> H = src.getH((0.01, 0.01, 0.01))
    >>> with np.printoptions(precision=3):
    ...     print(H)
    [3190.561 2552.449 1914.336]
    """

    _field_func = staticmethod(_BHJM_magnet_sphere)
    _force_type = "magnet"
    _field_func_kwargs_ndim: ClassVar[dict[str, int]] = {
        "polarization": 2,
        "diameter": 1,
    }
    get_trace = make_Sphere

    def __init__(
        self,
        position=(0, 0, 0),
        orientation=None,
        diameter=None,
        polarization=None,
        magnetization=None,
        style=None,
        **kwargs,
    ):
        # instance attributes
        self.diameter = diameter

        # init inheritance
        super().__init__(
            position, orientation, magnetization, polarization, style, **kwargs
        )

    # Properties
    @property
    def diameter(self):
        """Diameter of the sphere in units (m)."""
        return self._diameter

    @diameter.setter
    def diameter(self, dia):
        """Set sphere diameter.

        Parameters
        ----------
        dia : None or float
            Diameter in units (m).
        """
        self._diameter = check_format_input_scalar(
            dia,
            sig_name="diameter",
            sig_type="None or a positive number (int, float)",
            allow_None=True,
            forbid_negative=True,
        )

    @property
    def _default_style_description(self):
        """Default style description text"""
        if self.diameter is None:
            return "no dimension"
        return f"D={unit_prefix(self.diameter)}m"

    # Methods
    def _get_volume(self):
        """Volume of object in units (m³)."""
        if self.diameter is None:
            return 0.0

        return self.diameter**3 * np.pi / 6

    def _get_centroid(self, squeeze=True):
        """Centroid of object in units (m)."""
        if squeeze:
            return self.position
        return self._position

    def _get_dipole_moment(self):
        """Magnetic moment of object in units (A*m²)."""
        # test init
        if self.magnetization is None or self.diameter is None:
            return np.array((0.0, 0.0, 0.0))
        return self.magnetization * self.volume

    def _generate_mesh(self):
        """Generate mesh for force computation."""
        points = np.array([(0, 0, 0)])
        moments = np.array([self.volume * self.magnetization])
        return {"pts": points, "moments": moments}

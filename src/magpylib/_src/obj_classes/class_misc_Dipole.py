"""Dipole class code"""

from typing import ClassVar

import numpy as np

from magpylib._src.display.traces_core import make_Dipole
from magpylib._src.fields.field_BH_dipole import _BHJM_dipole
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.obj_classes.class_BaseExcitations import BaseSource
from magpylib._src.obj_classes.class_BaseProperties import BaseDipoleMoment
from magpylib._src.style import DipoleStyle
from magpylib._src.utility import unit_prefix


class Dipole(BaseSource, BaseDipoleMoment):
    """Magnetic dipole moment.

    Can be used as ``sources`` input for magnetic field computation and ``target``
    input for force computation.

    When ``position=(0, 0, 0)`` and ``orientation=None`` global and local coordinates
    coincide.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    position : array-like, shape (3,) or (p, 3), default (0, 0, 0)
        Object position(s) in global coordinates in units (m). ``position`` and
        ``orientation`` attributes define the object path.
    orientation : None | Rotation, default None
        Object orientation(s) in global coordinates as a scipy Rotation. Rotation can
        have length 1 or p. ``None`` generates a unit-rotation.
    moment : None | array-like, shape (3,), default None
        Magnetic dipole moment (A·m²) in local object coordinates. For homogeneous
        magnets the relation ``moment = magnetization * volume`` holds. For current
        loops the relation ``moment = current * loop_surface`` holds.
    style : None | dict, default None
        Style dictionary. Can also be provided via style underscore magic, e.g.
        ``style_color='red'``.

    Attributes
    ----------
    position : ndarray, shape (3,) or (p, 3)
        Same as constructor parameter ``position``.
    orientation : Rotation
        Same as constructor parameter ``orientation``.
    moment : None | ndarray, shape (3,)
        Same as constructor parameter ``moment``.
    centroid : ndarray, shape (3,) or (p, 3)
        Read-only. Object centroid in units (m) in global coordinates.
        Can be a path.
    dipole_moment : ndarray, shape (3,)
        Read-only. Object dipole moment (A·m²) in local object coordinates.
    parent : Collection | None
        Parent collection of the object.
    style : dict
        Style dictionary defining visual properties.

    Notes
    -----
    Returns inf or nan at the dipole position.

    Examples
    --------
    ``Dipole`` objects are magnetic field sources. In this example we compute the H-field
    (A/m) of a magnetic dipole with moment ``(10, 10, 10)`` (A·m²) at the observer
    position ``(10, 10, 10)`` centimeters:

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> src = magpy.misc.Dipole(moment=(10, 10, 10))
    >>> H = src.getH((0.01, 0.01, 0.01))
    >>> with np.printoptions(precision=0):
    ...     print(H)
    [306294. 306294. 306294.]
    """

    _field_func = staticmethod(_BHJM_dipole)
    _force_type = "magnet"
    _field_func_kwargs_ndim: ClassVar[dict[str, int]] = {"moment": 2}
    _style_class = DipoleStyle
    get_trace = make_Dipole
    _autosize = True

    def __init__(
        self,
        position=(0, 0, 0),
        orientation=None,
        moment=None,
        style=None,
        **kwargs,
    ):
        # instance attributes
        self.moment = moment

        # init inheritance
        super().__init__(position, orientation, style, **kwargs)

    # Properties
    @property
    def moment(self):
        """Magnetic dipole moment (A·m²) in local object coordinates."""
        return self._moment

    @moment.setter
    def moment(self, mom):
        """Set dipole moment.

        Parameters
        ----------
        mom : None | array-like, shape (3,)
            Dipole moment vector (A·m²) in local object coordinates.
        """
        self._moment = check_format_input_vector(
            mom,
            dims=(1,),
            shape_m1=3,
            sig_name="moment",
            sig_type="array-like (list, tuple, ndarray) with shape (3,)",
            allow_None=True,
        )

    @property
    def _default_style_description(self):
        """Default style description text"""
        moment = np.array([0.0, 0.0, 0.0]) if self.moment is None else self.moment
        moment_mag = np.linalg.norm(moment)
        if moment_mag == 0:
            return "no moment"
        return f"moment={unit_prefix(moment_mag)}A·m²"

    # Methods
    def _get_centroid(self, squeeze=True):
        """Centroid of object in units (m)."""
        if squeeze:
            return self.position
        return self._position

    def _generate_mesh(self):
        """Generate mesh for force computation."""
        points = np.array([(0, 0, 0)])
        moments = np.array([self.moment])
        return {"pts": points, "moments": moments}

    def _get_dipole_moment(self):
        """Magnetic moment of object (A·m²)."""
        # test init
        if self.moment is None:
            return np.array((0.0, 0.0, 0.0))
        return self.moment

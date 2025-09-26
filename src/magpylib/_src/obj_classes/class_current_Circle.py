"""CircularCircle current class code"""

from typing import ClassVar

import numpy as np

from magpylib._src.display.traces_core import make_Circle
from magpylib._src.fields.field_BH_circle import _BHJM_circle
from magpylib._src.input_checks import check_format_input_scalar
from magpylib._src.obj_classes.class_BaseExcitations import BaseCurrent
from magpylib._src.obj_classes.class_BaseProperties import BaseDipoleMoment
from magpylib._src.obj_classes.class_BaseTarget import BaseTarget
from magpylib._src.obj_classes.target_meshing import _target_mesh_circle
from magpylib._src.utility import unit_prefix


class Circle(BaseCurrent, BaseTarget, BaseDipoleMoment):
    """Circular current loop.

    Can be used as ``sources`` input for magnetic field computation and ``target``
    input for force computation.

    When ``position=(0, 0, 0)`` and ``orientation=None`` the current loop lies
    in the x-y plane of the global coordinate system, with its center in the
    origin.

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
        Loop diameter (m).
    current : float | None, default None
        Electrical current (A).
    meshing : int | None, default None
        Mesh fineness for force computation. Must be an integer ``>= 4``. Points
        are equally distributed on the circle.
    style : dict | None, default None
        Style dictionary. Can also be provided via style underscore magic, e.g.
        ``style_color='red'``.

    Attributes
    ----------
    position : ndarray, shape (3,) or (p, 3)
        Same as constructor parameter ``position``.
    orientation : Rotation
        Same as constructor parameter ``orientation``.
    diameter : None or float
        Same as constructor parameter ``diameter``.
    current : None or float
        Same as constructor parameter ``current``.
    meshing : None or int
        Same as constructor parameter ``meshing``.
    centroid : ndarray, shape (3,) or (p, 3)
        Read-only. Object centroid in units (m) in global coordinates.
        Can be a path.
    dipole_moment : ndarray, shape (3,)
        Read-only. Object dipole moment (A·m²) in local object coordinates.
    parent : Collection or None
        Parent collection of the object.
    style : dict
        Style dictionary defining visual properties.

    Notes
    -----
    Returns (0, 0, 0) on the loop.

    Examples
    --------
    ``Circle`` objects are magnetic field sources. In this example we compute the
    H-field (A/m) of such a current loop with 100 A current and a diameter of
    2 meters at the observer position (1, 1, 1) (cm):

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> src = magpy.current.Circle(current=100, diameter=2)
    >>> H = src.getH((0.01, 0.01, 0.01))
    >>> with np.printoptions(precision=3):
    ...     print(H)
    [7.501e-03 7.501e-03 5.000e+01]
    """

    _field_func = staticmethod(_BHJM_circle)
    _force_type = "current"
    _field_func_kwargs_ndim: ClassVar[dict[str, int]] = {"current": 1, "diameter": 1}
    get_trace = make_Circle

    def __init__(
        self,
        position=(0, 0, 0),
        orientation=None,
        diameter=None,
        current=None,
        meshing=None,
        *,
        style=None,
        **kwargs,
    ):
        # instance attributes
        self.diameter = diameter

        # init inheritance
        super().__init__(position, orientation, current, style, **kwargs)

        # Initialize BaseTarget
        BaseTarget.__init__(self, meshing)

    # Properties
    @property
    def diameter(self):
        """Diameter of the loop in units of m."""
        return self._diameter

    @diameter.setter
    def diameter(self, dia):
        """Set loop diameter.

        Parameters
        ----------
        dia : float | None
            Loop diameter in units (m).
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
        return f"{unit_prefix(self.current)}A" if self.current else "no current"

    # Methods
    def _get_centroid(self, squeeze=True):
        """Centroid of object in units of (m)."""
        if squeeze:
            return self.position
        return self._position

    def _get_dipole_moment(self):
        """Magnetic moment of object in units (A*m²)."""
        # test init
        if self.diameter is None or self.current is None:
            return np.array((0.0, 0.0, 0.0))

        return self.diameter**2 / 4 * np.pi * self.current * np.array((0, 0, 1))

    def _generate_mesh(self):
        """Generate mesh for force computation."""
        return _target_mesh_circle(self.diameter / 2, self.meshing, self.current)

    def _validate_meshing(self, value):
        """Circle makes only sense with at least 4 mesh points."""
        if isinstance(value, int) and value > 3:
            pass
        else:
            msg = (
                f"Input meshing must be an integer > 3 for {self!r}; "
                f"instead received {value!r}."
            )
            raise ValueError(msg)

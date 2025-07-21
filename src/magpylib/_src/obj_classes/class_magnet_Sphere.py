# pylint: disable=too-many-positional-arguments

"""Magnet Sphere class code"""

from typing import ClassVar

import numpy as np

from magpylib._src.display.traces_core import make_Sphere
from magpylib._src.fields.field_BH_sphere import BHJM_magnet_sphere
from magpylib._src.input_checks import check_format_input_scalar
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet
from magpylib._src.utility import unit_prefix


class Sphere(BaseMagnet):
    """Spherical magnet with homogeneous magnetization.

    Can be used as `sources` input for magnetic field computation.

    When `position=(0,0,0)` and `orientation=None` the sphere center is located
    in the origin of the global coordinate system.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    position: array_like, shape (3,) or (m,3), default=`(0,0,0)`
        Object position(s) in the global coordinates in units of m. For m>1, the
        `position` and `orientation` attributes together represent an object path.

    orientation: scipy `Rotation` object with length 1 or m, default=`None`
        Object orientation(s) in the global coordinates. `None` corresponds to
        a unit-rotation. For m>1, the `position` and `orientation` attributes
        together represent an object path.

    diameter: float, default=`None`
        Diameter of the sphere in units of m.

    polarization: array_like, shape (3,), default=`None`
        Magnetic polarization vector J = mu0*M in units of T,
        given in the local object coordinates (rotates with object).

    magnetization: array_like, shape (3,), default=`None`
        Magnetization vector M = J/mu0 in units of A/m,
        given in the local object coordinates (rotates with object).

    volume: float
        Object physical volume in units of m^3.

    parent: `Collection` object or `None`
        The object is a child of it's parent collection.

    style: dict
        Object style inputs must be in dictionary form, e.g. `{'color':'red'}` or
        using style underscore magic, e.g. `style_color='red'`.


    Returns
    -------
    magnet source: `Sphere` object

    Examples
    --------
    `Sphere` objects are magnetic field sources. In this example we compute the H-field in A/m
    of a spherical magnet with polarization (0.1,0.2,0.3) in units of T and diameter
    of 0.01 meter at the observer position (0.01,0.01,0.01) given in units of m:

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> src = magpy.magnet.Sphere(polarization=(.1,.2,.3), diameter=.01)
    >>> H = src.getH((.01,.01,.01))
    >>> with np.printoptions(precision=3):
    ...     print(H)
    [3190.561 2552.449 1914.336]
    """

    _field_func = staticmethod(BHJM_magnet_sphere)
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

    # property getters and setters
    @property
    def diameter(self):
        """Diameter of the sphere in units of m."""
        return self._diameter

    @diameter.setter
    def diameter(self, dia):
        """Set Sphere diameter, float, meter."""
        self._diameter = check_format_input_scalar(
            dia,
            sig_name="diameter",
            sig_type="`None` or a positive number (int, float)",
            allow_None=True,
            forbid_negative=True,
        )

    @property
    def volume(self):
        """Volume of object in units of m³."""
        if self.diameter is None:
            return 0.0

        return self.diameter**3 * np.pi / 6

    @property
    def _default_style_description(self):
        """Default style description text"""
        if self.diameter is None:
            return "no dimension"
        return f"D={unit_prefix(self.diameter)}m"

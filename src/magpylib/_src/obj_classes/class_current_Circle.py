"""CircularCircle current class code"""

import warnings
from typing import ClassVar

from magpylib._src.display.traces_core import make_Circle
from magpylib._src.exceptions import MagpylibDeprecationWarning
from magpylib._src.fields.field_BH_circle import BHJM_circle
from magpylib._src.input_checks import check_format_scalar_or_vector
from magpylib._src.obj_classes.class_BaseExcitations import BaseCurrent
from magpylib._src.utility import unit_prefix


class Circle(BaseCurrent):
    """Circular current loop.

    Can be used as `sources` input for magnetic field computation.

    When `position=(0,0,0)` and `orientation=None` the current loop lies
    in the x-y plane of the global coordinate system, with its center in
    the origin.

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

    diameter: array_like, shape (2,), or float, default=`None`
        Diameter of the loop in units of m.

    current: float, default=`None`
        Electrical current in units of A.

    volume: float
        Read-only. Object physical volume in units of m^3.

    centroid: np.ndarray, shape (3,) or (m,3)
        Read-only. Object centroid in units of m.

    parent: `Collection` object or `None`
        The object is a child of it's parent collection.

    style: dict
        Object style inputs must be in dictionary form, e.g. `{'color':'red'}` or
        using style underscore magic, e.g. `style_color='red'`.

    Returns
    -------
    current source: `Circle` object

    Examples
    --------
    `Circle` objects are magnetic field sources. In this example we compute the H-field in A/m
    of such a current loop with 100 A current and a diameter of 2 meters at the observer position
    (0.01,0.01,0.01) given in units of m:

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> src = magpy.current.Circle(current=100, diameter=2)
    >>> H = src.getH((.01,.01,.01))
    >>> with np.printoptions(precision=3):
    ...     print(H)
    [7.501e-03 7.501e-03 5.000e+01]
    """

    _field_func = staticmethod(BHJM_circle)
    _field_func_kwargs_ndim: ClassVar[dict[str, int]] = {"current": 1, "diameter": 1}
    get_trace = make_Circle

    def __init__(
        self,
        position=(0, 0, 0),
        orientation=None,
        diameter=None,
        current=None,
        *,
        style=None,
        **kwargs,
    ):
        # instance attributes
        self.diameter = check_format_scalar_or_vector(
            diameter, allow_None=True, length=2
        )
        if str(self.diameter) != "None":
            self.diameter = abs(self.diameter)

        # init inheritance
        super().__init__(position, orientation, current, style, **kwargs)

    # Properties
    @property
    def diameter(self):
        """Diameter of the loop in units of m."""
        return self._diameter

    @diameter.setter
    def diameter(self, dia):
        """Set Circle loop diameter, float, meter."""
        self._diameter = check_format_scalar_or_vector(
            dia,
            sig_name="diameter",
            sig_type="`None`, a scalar (int, float), or a array_like (list, tuple, ndarray) of shape (1,) and length 2",
            length=2,
            allow_None=True,
        )
        if str(self._diameter) != "None":
            self._diameter = abs(self._diameter)

    @property
    def _default_style_description(self):
        """Default style description text"""
        if self.diameter is None:
            return "no dimension"
        return f"{unit_prefix(self.current)}A" if self.current else "no current"

    # Methods
    def _get_volume(self):
        """Volume of object in units of m³."""
        return 0.0

    def _get_centroid(self):
        """Centroid of object in units of m."""
        return self.position


class Loop(Circle):
    """Loop is deprecated, see Circle"""

    # pylint: disable=method-hidden
    @staticmethod
    def _field_func(*args, **kwargs):
        """Catch Deprecation warning in getBH_dict"""
        _deprecation_warn()
        return BHJM_circle(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        _deprecation_warn()
        super().__init__(*args, **kwargs)


def _deprecation_warn():
    warnings.warn(
        (
            "Loop is deprecated  and will be removed in a future version, "
            "use Circle instead."
        ),
        MagpylibDeprecationWarning,
        stacklevel=2,
    )

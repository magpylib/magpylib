"""CircularCircle current class code"""
import warnings

from magpylib._src.display.traces_core import make_Circle
from magpylib._src.exceptions import MagpylibDeprecationWarning
from magpylib._src.fields.field_BH_circle import current_circle_field
from magpylib._src.input_checks import check_format_input_scalar
from magpylib._src.obj_classes.class_BaseExcitations import BaseCurrent
from magpylib._src.utility import unit_prefix


class Circle(BaseCurrent):
    """Circular current loop.

    Can be used as `sources` input for magnetic field computation.

    When `position=(0,0,0)` and `orientation=None` the current loop lies
    in the x-y plane of the global coordinate system, with its center in
    the origin. The Circle class has a dipole moment of pi**2/10*diameter**2*current.

    Parameters
    ----------
    current: float, default=`None`
        Electrical current in units of A.

    diameter: float, default=`None`
        Diameter of the loop in units of mm.

    position: array_like, shape (3,) or (m,3), default=`(0,0,0)`
        Object position(s) in the global coordinates in units of mm. For m>1, the
        `position` and `orientation` attributes together represent an object path.

    orientation: scipy `Rotation` object with length 1 or m, default=`None`
        Object orientation(s) in the global coordinates. `None` corresponds to
        a unit-rotation. For m>1, the `position` and `orientation` attributes
        together represent an object path.

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
    `Circle` objects are magnetic field sources. In this example we compute the H-field kA/m
    of such a current loop with 100 A current and a diameter of 2 mm at the observer position
    (1,1,1) given in units of mm:

    >>> import magpylib as magpy
    >>> src = magpy.current.Circle(current=100, diameter=2)
    >>> H = src.getH((1,1,1))
    >>> print(H)
    [4.96243034 4.96243034 2.12454191]

    We rotate the source object, and compute the B-field, this time at a set of observer positions:

    >>> src.rotate_from_angax(45, 'x')
    Circle(id=...)
    >>> B = src.getB([(1,1,1), (2,2,2), (3,3,3)])
    >>> print(B)
    [[-1.44441884e-15  6.72068135e+00 -6.72068135e+00]
     [-9.88027010e-17  5.89248328e-01 -5.89248328e-01]
     [-3.55802727e-17  1.65201495e-01 -1.65201495e-01]]

    The same result is obtained when the rotated source moves along a path away from an
    observer at position (1,1,1). This time we use a `Sensor` object as observer.

    >>> src.move([(-1,-1,-1), (-2,-2,-2)])
    Circle(id=...)
    >>> sens = magpy.Sensor(position=(1,1,1))
    >>> B = src.getB(sens)
    >>> print(B)
    [[-1.44441884e-15  6.72068135e+00 -6.72068135e+00]
     [-9.88027010e-17  5.89248328e-01 -5.89248328e-01]
     [-3.55802727e-17  1.65201495e-01 -1.65201495e-01]]
    """

    _field_func = staticmethod(current_circle_field)
    _field_func_kwargs_ndim = {"current": 1, "diameter": 1}
    get_trace = make_Circle

    def __init__(
        self,
        current=None,
        diameter=None,
        position=(0, 0, 0),
        orientation=None,
        style=None,
        **kwargs,
    ):
        # instance attributes
        self.diameter = diameter

        # init inheritance
        super().__init__(position, orientation, current, style, **kwargs)

    # property getters and setters
    @property
    def diameter(self):
        """Diameter of the loop in units of mm."""
        return self._diameter

    @diameter.setter
    def diameter(self, dia):
        """Set Circle loop diameter, float, mm."""
        self._diameter = check_format_input_scalar(
            dia,
            sig_name="diameter",
            sig_type="`None` or a positive number (int, float)",
            allow_None=True,
            forbid_negative=True,
        )

    @property
    def _default_style_description(self):
        """Default style description text"""
        if self.diameter is None:
            return "no dimension"
        return f"{unit_prefix(self.current)}A" if self.current else "no current"


class Loop(Circle):
    """Loop is deprecated, see Circle"""

    # pylint: disable=method-hidden
    @staticmethod
    def _field_func(*args, **kwargs):
        """Catch Deprecation warning in getBH_dict"""
        _deprecation_warn()
        return current_circle_field(*args, **kwargs)

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

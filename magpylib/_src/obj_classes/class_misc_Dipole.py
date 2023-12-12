"""Dipole class code"""
import numpy as np

from magpylib._src.display.traces_core import make_Dipole
from magpylib._src.fields.field_BH_dipole import dipole_field
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.obj_classes.class_BaseExcitations import BaseSource
from magpylib._src.style import DipoleStyle
from magpylib._src.utility import unit_prefix


class Dipole(BaseSource):
    """Magnetic dipole moment.

    Can be used as `sources` input for magnetic field computation.

    When `position=(0,0,0)` and `orientation=None` the dipole is located in the origin of
    global coordinate system.

    Parameters
    ----------
    moment: array_like, shape (3,), unit mT*mm^3, default=`None`
        Magnetic dipole moment in units of mT*mm^3 given in the local object coordinates.
        For homogeneous magnets the relation moment=magnetization*volume holds. The dipole
        moment of a Circle object is pi**2/10*diameter**2*current.

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
    source: `Dipole` object

    Examples
    --------
    `Dipole` objects are magnetic field sources. In this example we compute the H-field kA/m
    of such a magnetic dipole with a moment of (100,100,100) in units of mT*mm^2 at an
    observer position (1,1,1) given in units of mm:

    >>> import magpylib as magpy
    >>> src = magpy.misc.Dipole(moment=(100,100,100))
    >>> H = src.getH((1,1,1))
    >>> print(H)
    [2.43740886 2.43740886 2.43740886]

    We rotate the source object, and compute the B-field, this time at a set of observer positions:

    >>> src.rotate_from_angax(45, 'x')
    Dipole(id=...)
    >>> B = src.getB([(1,1,1), (2,2,2), (3,3,3)])
    >>> print(B)
    [[2.16582445 3.6972936  1.53146915]
     [0.27072806 0.4621617  0.19143364]
     [0.08021572 0.1369368  0.05672108]]

    The same result is obtained when the rotated source moves along a path away from an
    observer at position (1,1,1). This time we use a `Sensor` object as observer.

    >>> src.move([(-1,-1,-1), (-2,-2,-2)])
    Dipole(id=...)
    >>> sens = magpy.Sensor(position=(1,1,1))
    >>> B = src.getB(sens)
    >>> print(B)
    [[2.16582445 3.6972936  1.53146915]
     [0.27072806 0.4621617  0.19143364]
     [0.08021572 0.1369368  0.05672108]]
    """

    _field_func = staticmethod(dipole_field)
    _field_func_kwargs_ndim = {"moment": 2}
    _style_class = DipoleStyle
    get_trace = make_Dipole
    _autosize = True

    def __init__(
        self,
        moment=None,
        position=(0, 0, 0),
        orientation=None,
        style=None,
        **kwargs,
    ):
        # instance attributes
        self.moment = moment

        # init inheritance
        super().__init__(position, orientation, style, **kwargs)

    # property getters and setters
    @property
    def moment(self):
        """Magnetic dipole moment in units of mT*mm^3 given in the local object coordinates."""
        return self._moment

    @moment.setter
    def moment(self, mom):
        """Set dipole moment vector, shape (3,), unit mT*mm^3."""
        self._moment = check_format_input_vector(
            mom,
            dims=(1,),
            shape_m1=3,
            sig_name="moment",
            sig_type="array_like (list, tuple, ndarray) with shape (3,)",
            allow_None=True,
        )

    @property
    def _default_style_description(self):
        """Default style description text"""
        moment = np.array([0.0, 0.0, 0.0]) if self.moment is None else self.moment
        moment_mag = np.linalg.norm(moment)
        if moment_mag == 0:
            return "no moment"
        return f"moment={unit_prefix(moment_mag)}mT mm³"

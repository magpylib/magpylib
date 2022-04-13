"""Loop current class code
DOCSTRINGS V4 READY
"""
from magpylib._src.input_checks import check_format_input_scalar
from magpylib._src.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._src.obj_classes.class_BaseExcitations import BaseCurrent
from magpylib._src.obj_classes.class_BaseGeo import BaseGeo
from magpylib._src.obj_classes.class_BaseGetBH import BaseGetBH


class Loop(BaseGeo, BaseDisplayRepr, BaseGetBH, BaseCurrent):
    """Circular current loop.

    Can be used as `sources` input for magnetic field computation.

    When `position=(0,0,0)` and `orientation=None` the current loop lies
    in the x-y plane of the global coordinate system, with its center in
    the origin.

    Parameters
    ----------
    current: float, default=`None`
        Electrical current in units of [A].

    diameter: float, default=`None`
        Diameter of the loop in units of [mm].

    position: array_like, shape (3,) or (m,3), default=`(0,0,0)`
        Object position(s) in the global coordinates in units of [mm]. For m>1, the
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
    current source: `Loop` object

    Examples
    --------
    `Loop` objects are magnetic field sources. In this example we compute the H-field [kA/m]
    of such a current loop with 100 [A] current and a diameter of 2 [mm] at the observer position
    (1,1,1) given in units of [mm]:

    >>> import magpylib as magpy
    >>> src = magpy.current.Loop(current=100, diameter=2)
    >>> H = src.getH((1,1,1))
    >>> print(H)
    [4.96243034 4.96243034 2.12454191]

    We rotate the source object, and compute the B-field, this time at a set of observer positions:

    >>> src.rotate_from_angax(45, 'x')
    Loop(id=...)
    >>> B = src.getB([(1,1,1), (2,2,2), (3,3,3)])
    >>> print(B)
    [[-1.44441884e-15  6.72068135e+00 -6.72068135e+00]
     [-9.88027010e-17  5.89248328e-01 -5.89248328e-01]
     [-3.55802727e-17  1.65201495e-01 -1.65201495e-01]]

    The same result is obtained when the rotated source moves along a path away from an
    observer at position (1,1,1). This time we use a `Sensor` object as observer.

    >>> src.move([(-1,-1,-1), (-2,-2,-2)])
    Loop(id=...)
    >>> sens = magpy.Sensor(position=(1,1,1))
    >>> B = src.getB(sens)
    >>> print(B)
    [[-1.44441884e-15  6.72068135e+00 -6.72068135e+00]
     [-9.88027010e-17  5.89248328e-01 -5.89248328e-01]
     [-3.55802727e-17  1.65201495e-01 -1.65201495e-01]]
    """

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
        self._object_type = "Loop"

        # init inheritance
        BaseGeo.__init__(self, position, orientation, style=style, **kwargs)
        BaseDisplayRepr.__init__(self)
        BaseCurrent.__init__(self, current)

    # property getters and setters
    @property
    def diameter(self):
        """Diameter of the loop in units of [mm]."""
        return self._diameter

    @diameter.setter
    def diameter(self, dia):
        """Set Loop loop diameter, float, [mm]."""
        self._diameter = check_format_input_scalar(
            dia,
            sig_name="diameter",
            sig_type="`None` or a positive number (int, float)",
            allow_None=True,
            forbid_negative=True,
        )

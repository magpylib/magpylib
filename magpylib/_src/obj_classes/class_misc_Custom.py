"""Custom class code
DOCSTRINGS V4 READY
"""

from magpylib._src.obj_classes.class_BaseGeo import BaseGeo
from magpylib._src.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._src.obj_classes.class_BaseGetBH import BaseGetBH
from magpylib._src.input_checks import validate_field_lambda


# ON INTERFACE
class CustomSource(BaseGeo, BaseDisplayRepr, BaseGetBH):
    """User-defined custom source.

    Can be used as `sources` input for magnetic field computation.

    When `position=(0,0,0)` and `orientation=None` local object coordinates
    coincide with the gobal coordinate system.

    Parameters
    ----------
    field_B_lambda: callable, default=`None`
        Field function for the B-field. must accept position input with format (n,3) and
        return the B-field with similar shape.

    field_H_lambda: callable, default=`None`
        Field function for the H-field. must accept position input with format (n,3) and
        return the H-field with similar shape.

    position: array_like, shape (3,) or (m,3), default=`(0,0,0)`
        Object position(s) in the global coordinates in units of [mm]. For m>1, the
        `position` and `orientation` attributes together represent an object path.

    orientation: scipy `Rotation` object with length 1 or m, default=`None`
        Object orientation(s) in the global coordinates. `None` corresponds to
        a unit-rotation. For m>1, the `position` and `orientation` attributes
        together represent an object path.

    style: dict
        Object style inputs must be in dictionary form, e.g. `{'color':'red'}` or
        using style_underscore_magic, e.g. `style_color='red'`.

    Returns
    -------
    source: `CustomSource` object

    Examples
    --------
    With version 4.0.0 `CustomSource` objects enable users to define their own source
    objects, and to embedd them in the Magpylib object oriented interface. In this example
    we create a source that generates a constant field and evaluate the field at observer
    position (1,1,1) given in [mm]:

    >>> import numpy as np
    >>> import magpylib as magpy
    >>>
    >>> bfield = lambda observer: np.array([(100,0,0)]*len(observer))
    >>> hfield = lambda observer: np.array([(80,0,0)]*len(observer))
    >>> src = magpy.misc.CustomSource(field_B_lambda=bfield, field_H_lambda=hfield)
    >>> H = src.getH((1,1,1))
    >>> print(H)
    [80.  0.  0.]

    We rotate the source object, and compute the B-field, this time at a set of observer positions:

    >>> src.rotate_from_angax(45, 'z')
    >>> B = src.getB([(1,1,1), (2,2,2), (3,3,3)])
    >>> print(B)
    [[70.71067812 70.71067812  0.        ]
     [70.71067812 70.71067812  0.        ]
     [70.71067812 70.71067812  0.        ]]

    The same result is obtained when the rotated source moves along a path away from an
    observer at position (1,1,1). This time we use a `Sensor` object as observer.

    >>> src.move([(-1,-1,-1), (-2,-2,-2)])
    >>> sens = magpy.Sensor(position=(1,1,1))
    >>> B = src.getB(sens)
    >>> print(B)
    [[70.71067812 70.71067812  0.        ]
     [70.71067812 70.71067812  0.        ]
     [70.71067812 70.71067812  0.        ]]
    """

    def __init__(
        self,
        field_B_lambda=None,
        field_H_lambda=None,
        position=(0, 0, 0),
        orientation=None,
        style=None,
        **kwargs,
    ):
        # instance attributes
        self.field_B_lambda = field_B_lambda
        self.field_H_lambda = field_H_lambda
        self._object_type = "CustomSource"

        # init inheritance
        BaseGeo.__init__(self, position, orientation, style=style, **kwargs)
        BaseDisplayRepr.__init__(self)

    @property
    def field_B_lambda(self):
        """Field function for B-field, should accept array_like positions of shape (n,3)
        in units of [mm] and return a B-field array of same shape in the global
        coordinate system in units of [mT].
        """
        return self._field_B_lambda

    @field_B_lambda.setter
    def field_B_lambda(self, val):
        self._field_B_lambda = validate_field_lambda(val, "B")

    @property
    def field_H_lambda(self):
        """Field function for H-field, should accept array_like positions of shape (n,3)
        in units of [mm] and return a H-field array of same shape in the global
        coordinate system in units of [kA/m].
        """
        return self._field_H_lambda

    @field_H_lambda.setter
    def field_H_lambda(self, val):
        self._field_H_lambda = validate_field_lambda(val, "H")

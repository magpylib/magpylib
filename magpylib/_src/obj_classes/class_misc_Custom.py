"""Custom class code"""

from magpylib._src.obj_classes.class_BaseGeo import BaseGeo
from magpylib._src.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._src.obj_classes.class_BaseGetBH import BaseGetBH
from magpylib._src.input_checks import validate_field_lambda


# ON INTERFACE
class CustomSource(BaseGeo, BaseDisplayRepr, BaseGetBH):
    """
    Custom Source class

    Parameters
    ----------
    field_B_lambda: function
        field function for B-field, should accept (n,3) position array and return
        B-field array of same shape in the global coordinate system.

    field_H_lambda: function
        field function for H-field, should accept (n,3) position array and return
        H-field array of same shape in the global coordinate system.

    position: array_like, shape (3,) or (M,3), default=(0,0,0)
        Object position (local CS origin) in the global CS in units of [mm].
        For M>1, the position represents a path. The position and orientation
        parameters must always be of the same length.

    orientation: scipy Rotation object with length 1 or M, default=unit rotation
        Object orientation (local CS orientation) in the global CS. For M>1
        orientation represents different values along a path. The position and
        orientation parameters must always be of the same length.

    Returns
    -------
    CustomSource object: CustomSource

    Examples
    --------
    By default the CustomSource is initialized at position (0,0,0), with unit rotation:

    >>> import magpylib as magpy
    >>> import numpy as np

    Define a external B-field function which returns constant vector in x-direction

    >>> def constant_Bfield(position=((0,0,0))):
    ...    return np.array([[1,0,0]]*len(position))

    Construct a ``CustomSource`` from the field function

    >>> external_field = magpy.misc.CustomSource(field_B_lambda=constant_Bfield)
    >>> B = external_field.getB([[1,2,3],[4,5,6]])
    >>> print(B)
    [[1. 0. 0.]
     [1. 0. 0.]]

    The custom source can be rotated as any other source object in the library.

    >>> external_field.rotate_from_angax(90, 'z')
    >>> B = external_field.getB([[1,2,3],[4,5,6]])
    >>> print(B) # Notice the output field is now pointing in y-direction
    [[0. 1. 0.]
     [0. 1. 0.]]
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
        """field function for B-field, should accept (n,3) position array and return
        B-field array of same shape in the global coordinate system."""
        return self._field_B_lambda

    @field_B_lambda.setter
    def field_B_lambda(self, val):
        self._field_B_lambda = validate_field_lambda(val, "B")

    @property
    def field_H_lambda(self):
        """field function for H-field, should accept (n,3) position array and return
        H-field array of same shape in the global coordinate system."""
        return self._field_H_lambda

    @field_H_lambda.setter
    def field_H_lambda(self, val):
        self._field_H_lambda = validate_field_lambda(val, "H")

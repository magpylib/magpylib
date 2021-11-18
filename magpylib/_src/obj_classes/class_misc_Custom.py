"""Custom class code"""

import numpy as np
from magpylib._src.obj_classes.class_BaseGeo import BaseGeo
from magpylib._src.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._src.obj_classes.class_BaseGetBH import BaseGetBH


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
    By default a Dipole is initialized at position (0,0,0), with unit rotation:

    """

    def __init__(
        self,
        field_B_lambda=None,
        field_H_lambda=None,
        position=(0, 0, 0),
        orientation=None,
        style=None,
    ):
        # instance attributes
        self.field_B_lambda = field_B_lambda
        self.field_H_lambda = field_H_lambda
        self._object_type = "CustomSource"

        # init inheritance
        BaseGeo.__init__(self, position, orientation, style=style)
        BaseDisplayRepr.__init__(self)

    @property
    def field_B_lambda(self):
        return self._field_B_lambda

    @field_B_lambda.setter
    def field_B_lambda(self, val):
        self._field_B_lambda = self._validate_field_lambda(val, "B")

    @property
    def field_H_lambda(self):
        return self._field_H_lambda

    @field_H_lambda.setter
    def field_H_lambda(self, val):
        self._field_H_lambda = self._validate_field_lambda(val, "H")

    def _validate_field_lambda(self, val, bh):
        if val is not None:
            assert callable(val), f"field_{bh}_lambda must be a callable"
            out = val(np.array([[1, 2, 3], [4, 5, 6]]))
            out_shape = np.array(out).shape
            assert out_shape == (2, 3), (
                f"field_{bh}_lambda input shape and output "
                "shape must match and be of dimension (n,3)\n"
                f"received shape={out_shape} instead"
            )
        return val


"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import magpylib as magpy
# create virtual measured data

dim = [1,1,1]
Nelem = [2,2,2]
slices = [slice(-d/2,d/2,N*1j) for d,N in zip(dim,Nelem)]
positions = np.mgrid[slices].reshape(len(slices),-1).T

cube  = magpy.magnet.Cuboid(magnetization=(0,0,1000), position=(0,0,-20), dimension=(10,10,10))
sens = magpy.Sensor(pixel=positions)

magpy.display(cube, sens, backend='plotly')

def interpolate_field(data, field_type='B', bounds_error=False, fill_value=np.nan):
    '''data: x,y,z,Bx,By,Bz numpy array sorted by (x,y,z)
    returns: 
        interpolating function for B field values'''
    x,y,z,Bx,By,Bz = data.T
    nx,ny,nz = len(np.unique(x)), len(np.unique(y)), len(np.unique(z))
    X = np.linspace(np.min(x), np.max(x), nx)
    Y = np.linspace(np.min(y), np.max(y), ny)
    Z = np.linspace(np.min(z), np.max(z), nz)
    BX_interp = RegularGridInterpolator((X,Y,Z), Bx.reshape(nx,ny,nz), bounds_error=bounds_error, fill_value=fill_value)
    BY_interp = RegularGridInterpolator((X,Y,Z), By.reshape(nx,ny,nz), bounds_error=bounds_error, fill_value=fill_value)
    BZ_interp = RegularGridInterpolator((X,Y,Z), Bz.reshape(nx,ny,nz), bounds_error=bounds_error, fill_value=fill_value)
    return lambda x: np.array([BX_interp(x),BY_interp(x),BZ_interp(x)]).T

B = sens.getB(cube)
#B *= 1 + np.random.random_sample(B.shape)*0.01 # add 1% white noise
data = np.hstack([positions,B])
field_B_lambda = interpolate_field(data, field_type='B')
display(B, field_B_lambda(positions))

exp_data = magpy.misc.CustomSource(field_B_lambda=field_B_lambda)
cube  = magpy.magnet.Cuboid(magnetization=(0,0,1000), position=(0,0,-20), dimension=(10,10,10))
sens = magpy.Sensor(pixel=positions)

display(sens.getB(exp_data), sens.getB(cube))

rotation = dict(angle=35, axis=(2,5,0.3), anchor=(-4, 2.5, 0)) # random rotation parameters
exp_data.rotate_from_angax(**rotation)
sens.rotate_from_angax(**rotation)
cube.rotate_from_angax(**rotation)

display(sens.getB(exp_data), sens.getB(cube))

exp_data = magpy.misc.CustomSource(field_B_lambda=field_B_lambda)
cube  = magpy.magnet.Cuboid(magnetization=(0,0,1000), position=(0,0,-20), dimension=(10,10,10))
sens = magpy.Sensor(pixel=positions)

display(sens.getB(exp_data), sens.getB(cube))

displ = (4,6,5)
exp_data.move(displ)
sens.move(displ)
cube.move(displ)

display(sens.getB(exp_data), sens.getB(cube))
"""
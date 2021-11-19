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
    By default the CustomSource is initialized at position (0,0,0), with unit rotation:
    >>> import magpylib as magpy
    >>> import numpy as np

    Define a external B-field function which returns constant vector in x-direction
    >>> def constant_Bfield(position=((0,0,0))):
    ...    return np.tile([1,0,0], (len(position),1))

    Construct a ``CustomSource`` from the field function
    >>> external_field = magpy.misc.CustomSource(field_B_lambda=constant_Bfield)
    >>> B = external_field.getB([[1,2,3],[4,5,6]])
    >>> print(B)
    [[1. 0. 0.]
     [1. 0. 0.]]
    The custom source can be rotated as any other source object in the library.
    >>> external_field.rotate_from_angax(90, 'z')
    >>> B = external_field.getB([[1,2,3],[4,5,6]])
    >>> print(B) # Notice the outut field is now pointing in y-direction
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
        """field function for B-field, should accept (n,3) position array and return
        B-field array of same shape in the global coordinate system."""
        return self._field_B_lambda

    @field_B_lambda.setter
    def field_B_lambda(self, val):
        self._field_B_lambda = self._validate_field_lambda(val, "B")

    @property
    def field_H_lambda(self):
        """field function for H-field, should accept (n,3) position array and return
        H-field array of same shape in the global coordinate system."""
        return self._field_H_lambda

    @field_H_lambda.setter
    def field_H_lambda(self, val):
        self._field_H_lambda = self._validate_field_lambda(val, "H")

    @staticmethod
    def _validate_field_lambda(val, bh):
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


example = """
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import magpylib as magpy

magpy.defaults.display.backend='plotly'

def interpolate_field(data, method='linear', bounds_error=False, fill_value=np.nan):
    ''' Creates a 3d-vector field interpolation of a rasterized data from a regular grid

    Parameters
    ----------
    data: numpy.ndarray or array-like
        `numpy.ndarray` of shape (n,6). In order to be a regular grid, the first dimension n
        corresponds to the product of the unique values in x,y,z-directions. The second dimension
        should have following ordering:
         x, y, z, field_x, field_y, field_z

    see `scipy.interpolate.RegularGridInterpolator` for the other parameters

    Returns
    -------
        callable: interpolating function for field values
    '''
    data = np.array(data)
    idx = np.lexsort((data[:,2], data[:,1], data[:,0])) # sort data by x,y,z
    x,y,z, *field_vec = data[idx].T
    X,Y,Z = np.unique(x), np.unique(y), np.unique(z)
    nx,ny,nz = len(X), len(Y), len(Z)
    kwargs = dict(bounds_error=bounds_error, fill_value=fill_value, method=method)
    field_interp = []
    for k, kn in zip((X,Y,Z), 'xyz'):
        assert np.unique(np.diff(k)).shape[0]==1, f"not a regular grid in {kn}-direction"
    for field in field_vec:
        rgi = RegularGridInterpolator((X,Y,Z), field.reshape(nx,ny,nz), **kwargs)
        field_interp.append(rgi)
    return lambda x: np.array([field(x) for field in field_interp]).T

# create virtual measured data

dim = [2,2,2]
Nelem = [2,2,2]
slices = [slice(-d/2,d/2,N*1j) for d,N in zip(dim,Nelem)]
positions = np.mgrid[slices].reshape(len(slices),-1).T

cube  = magpy.magnet.Cuboid(magnetization=(0,0,1000), position=(0,0,-20), dimension=(10,10,10))
B = cube.getB(positions)
#B *= 1 + np.random.random_sample(B.shape)*0.01 # add 1% white noise
Bdata = np.hstack([positions,B])
#Bdata[-1,2] = 0.6
field_B_lambda = interpolate_field(Bdata)
display('Check field function vs magpylib Cuboid', field_B_lambda(positions))

xp = magpy.misc.CustomSource(field_B_lambda=field_B_lambda)
cube  = magpy.magnet.Cuboid(magnetization=(0,0,1000), position=(0,0,-20), dimension=(10,10,10))
# avoid edge of domain which may yield `np.nan` because of floating pt err
sens = magpy.Sensor(pixel=positions*0.9)

magpy.display(cube, sens)
display('Before rotation %err vs magpylib Cuboid',
        ((sens.getB(xp) -sens.getB(cube))/sens.getB(xp)*100).round(3)
)

rotation = dict(angle=35, axis=(1,5,.4), anchor=(1, 80, -4)) # random rotation parameters
xp.rotate_from_angax(**rotation)
sens.rotate_from_angax(**rotation)
cube.rotate_from_angax(**rotation)

magpy.display(cube, sens)
display('After rotation %err vs magpylib Cuboid',
        ((sens.getB(xp) -sens.getB(cube))/sens.getB(xp)*100).round(3)
)
"""

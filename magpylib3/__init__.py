"""
Magpylib provides 3D magnetic field computation based on analytical formulas.

#### Sources:
Create source objects that represent physical magnetic frield sources. Classes can be found in top-level sub-packages. 
    
- .magnet
    - .Box()
    - .Cylinder()
    - .Sphere()

- .current
    - .Line()
    - .Circular()

- .moment
    - .Dipole()

Manipulate sources through provided methods and parameters

- src.mag = new_magnetization
- src.dim = new_dimension
- src.pos = new_position
- src.rot = new_orientation
- src.move(displacement)
- src.rotate(rotation input)

#### Collections:
Group sources for common manipulation.
All methods that work for sources also work for collections.

- col = src1 + src2 + src3 ...
- .Collection(src1, src2, ...)

#### Field computation 
There are three ways to compute the field of sources. In addition to getB there is getH.

1. src.getB(positions)
2. .getB(*sources, pos_obs = positions)
3. .getBv(**kwargs)

#### Graphic output
Display sources using Matplotlib through

- .display(sources, collections, lists, ...)
- src.display()

"""

# module level dunders
__version__ = '3.0.0'
__author__ =  'Michael Ortner & friends'
__credits__ = 'Silicon Austria Labs - Sensor Systems'

# interface
__all__ = ['magnet', 'current', 'moment', 'getB', 'getH', 'getBv', 'getHv', 'Collection', 'display', 'config']

# create interface
from . import magnet
from . import current
from . import moment
from ._lib.config import config
from ._lib.fields.field_BHwrapper import getB, getH, getBv, getHv
from ._lib.obj_classes import Collection
from ._lib.graphics import display

"""
Magpylib provides 3D magnetic field computation based on analytical
formulas.

Ressources
----------
www.https://magpylib.readthedocs.io/en/latest/
https://github.com/magpylib/magpylib
https://www.sciencedirect.com/science/article/pii/S2352711020300170

Sources
-------
Create source objects that represent physical magnetic field sources.
Classes can be found in top-level sub-packages.

magpylib.magnet
- .Box()
- .Cylinder()
- .Sphere()

magpylib.current
- .Line()
- .Circular()

magpylib.moment
- .Dipole()

Manipulate sources through provided methods and parameters

- src.pos = new_position
- src.rot = new_orientation
- src.move_by(displacement)
- src.move_to(target_position)
- src.rotate(rotation input)
- src.rotate_from_angax(rotation input)

pos and rot can also represent complete source paths. Use "steps"
variable with move and rotate methods to conveniently generate such
paths.

Collections
-----------
Group sources for common manipulation.

- col = src1 + src2 + src3 ...
- magpylib.Collection(src1, src2, ...)

All methods that work for sources also work for collections.

Field computation
-----------------
There are three ways to compute the field.

1. src.getB(positions) ----------------> field of one source
2. magpylib.getB(sources, positions) --> fields of many sources
3. magpylib.getBv(**kwargs) -----------> direct access to core formulas

In addition to getB there is getH.

Graphic output
--------------
Display sources, collections, paths and sensors using Matplotlib

- magpylib.display(sources)
- src.display()

"""

# module level dunders
__version__ = '3.0.0'
__author__ =  'Michael Ortner & friends'
__credits__ = 'Silicon Austria Labs - Sensor Systems'
__all__ = ['magnet', 'current', 'moment',
           'getB', 'getH', 'getBv', 'getHv','Sensor',
           'Collection', 'display', 'Config','multi_motion']

# create interface
from . import magnet
from . import current
from . import moment
from ._lib.config import Config
from ._lib.fields.field_BH_wrapper import getB, getH, getBv, getHv
from ._lib.obj_classes import Collection, Sensor
from ._lib.display import display
from ._lib.obj_classes.class_BaseGeo import multi_motion

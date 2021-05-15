"""
Magpylib provides static 3D magnetic field computation based on analytical
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

magpylib.misc
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

sources are e.g. magnet.Box objects, current.Line objects
observers are positions (array_like) or Sensor objects

1. src.getB(observers) ----------------> field of one source at all observers
2. sens.getB(sources) -----------------> field of all sources at one sensor
2. magpylib.getB(sources, observers) --> fields of all sources at all observers
3. magpylib.getBv(**kwargs) -----------> direct access to core formulas (fastest)

In addition to getB there is getH.

Graphic output
--------------
Display sources, collections, paths and sensors using Matplotlib

- magpylib.display(src1, src2, sens, col, ...)
- src.display()
- sens.display()
- col.display()
"""

# module level dunders
__version__ = '3.0.0'
__author__ =  'Michael Ortner & friends'
__credits__ = 'Silicon Austria Labs - Sensor Systems'
__all__ = ['magnet', 'current', 'misc',
           'getB', 'getH', 'getBv', 'getHv','Sensor',
           'Collection', 'display', 'Config','path_merge']

# create interface to outside of package
from magpylib import magnet
from magpylib import current
from magpylib import misc
from magpylib._lib.config import Config
from magpylib._lib.fields import getB, getH, getBv, getHv
from magpylib._lib.obj_classes import Collection, Sensor, path_merge
from magpylib._lib.display import display

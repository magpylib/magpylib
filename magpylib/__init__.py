"""
Welcome to Magpylib !
---------------------

Magpylib provides static 3D magnetic field computation for permanent magnets,
currents and other sources using (semi-) analytical formulas from the literature.

Resources
---------

Documentation on Read-the-docs:

https://magpylib.readthedocs.io/en/latest/

Github repository:

https://github.com/magpylib/magpylib

Original software publication (version 2):

https://www.sciencedirect.com/science/article/pii/S2352711020300170

Introduction
------------
Magpylib uses units of

    - [mT]: for the B-field and the magnetization (mu0*M).
    - [kA/m]: for the H-field.
    - [mm]: for all position inputs.
    - [deg]: for angle inputs by default.
    - [A]: for current inputs.

Magpylib objects represent magnetic field sources and sensors:

>>> import magpylib as mag3
>>> # magnets
>>> src1 = mag3.magnet.Box()
>>> src2 = mag3.magnet.Cylinder()
>>> src3 = mag3.magnet.Sphere()
>>> # currents
>>> src4 = mag3.current.Line()
>>> src5 = mag3.current.Circular()
>>> # dipoles
>>> src6 = mag3.misc.Dipole()
>>> # sensors
>>> sens = mag3.Sensor()

These objects are endowed with position and orientation attributes in a global
coordinate system. Manipulate position and orientation directly through source
attributes,

>>> src.position = new_position
>>> src.orientation = new_orientation

or through provided methods,

>>> src.move()
>>> src.rotate()
>>> src.rotate_from_angax()

Source position and rotation attributes can also represent complete source paths in the
global coordinate system. Such paths can be generated conveniently using the `.move` and
`.rotate` methods.

Grouping objects
----------------

Use the Collection class to group objects for common manipulation.

>>> col = src1 + src2 + src3 ...
>>> col = mag3.Collection(src1, src2, ...)

All methods that work for objects also work for Collections.

>>> col.move()    # moves all objects in the Collection
>>> col.rotate()  # rotates all objects in the Collection

Field computation
-----------------

The magnetic field can be computed through the top level functions `getB` and `getH`,

>>> import magpylib as mag3
>>> mag3.getB(sources, observers)

or throught object methods

>>> src.getB(observers)
>>> sens.getB(sources)
>>> col.getB(observers)

Sources are magpylib source objects like Box or Line. Observers are magpylib Sensor
objects or simply sets (list, tuple, ndarray) of positions.

Finally there is a direct (very fast) interface to the field computation formulas

>>> mag3.getBv()

Graphic output
--------------
Display sources, collections, paths and sensors using Matplotlib from top level
functions,

>>> import magpylib as mag3
>>> mag3.display(src1, src2, sens, col, ...)

or through object methods

>>> src.display()
>>> sens.display()
>>> col.display()

"""

# module level dunders
__version__ = '3.0.0'
__author__ =  'Michael Ortner & friends'
__credits__ = 'Silicon Austria Labs - Sensor Systems'
__all__ = ['magnet', 'current', 'misc',
           'getB', 'getH', 'getBv', 'getHv','Sensor',
           'Collection', 'display', 'Config']

# create interface to outside of package
from magpylib import magnet
from magpylib import current
from magpylib import misc
from magpylib._lib.config import Config
from magpylib._lib.fields import getB, getH, getBv, getHv
from magpylib._lib.obj_classes import Collection, Sensor
from magpylib._lib.display import display

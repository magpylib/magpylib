# pylint: disable=line-too-long

"""
Welcome to Magpylib !
---------------------

Magpylib combines static 3D magnetic field computation for permanent magnets,
currents and other sources. The computations are based on analytical expressions
and are combined with an object oriented (magnet, current, sensor) position/orientation
interface and graphical output to display the system geometry.

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

Define magnets, currents and sensors as python objects, set their position and orientation in a global coordinate system and compute the magnetic field.

>>> import magpylib as magpy

Define a ``Cuboid`` magnet source object.

>>> src1 = magpy.magnet.Cuboid(magnetization=(0,0,1000), dimension=(1,2,3))
>>> print(src1.position)
>>> print(src1.orientation.as_euler('xyz', degrees=True))
[0. 0. 0.]
[0. 0. 0.]

Define a sensor object at a specific position.

>>> sens1 = magpy.Sensor(position=(1,2,3))
>>> print(sens1.position)
[1. 2. 3.]

Use the built-in move method to move a second source around.

>>> src2 = magpy.current.Loop(current=15, diameter=3)
>>> src2.move((1, 1, 1))
>>> print(src2.position)
[1. 1. 1.]

Use the built-in rotate methods to move a second sensor around.

>>> sens2 = magpy.Sensor(position=(1,0,0))
>>> sens2.rotate_from_angax(angle=45, axis=(0,0,1), anchor=(0,0,0))
>>> print(sens2.position)
>>> print(sens2.orientation.as_euler('xyz', degrees=True))
[0.70710678 0.70710678  0.        ]
[0.         0.         45.       ]

Compute the B-field generated by the source `src1` at the sensor `sens1`.

>>> B = magpy.getB(src1, sens1)
>>> print(B)
[ 7.48940807 13.41208607  8.02900384]

Compute the H-field of two sources at two sensors with one line of code.

>>> H = magpy.getH([src1, src2], [sens1, sens2])
>>> print(H)
[[[ 5.95988158e+00  1.06729990e+01  6.38927824e+00]
  [ 1.98854533e-14  1.98854533e-14 -6.10055863e+01]]

 [[ 2.68813151e-17  4.39005190e-01  8.11887842e-01]
  [ 5.64983190e-01 -2.77555756e-16  2.81121230e+00]]]

Position and orientation attributes can be paths.

>>> src1.move([(1,1,1), (2,2,2), (3,3,3)])
>>> print(src1.position)
[[0. 0. 0.]
 [1. 1. 1.]
 [2. 2. 2.]
 [3. 3. 3.]]

Field computation is automatically performed on the whole path in a vectorized form.

>>> B = src1.getB(sens1)
>>> print(B)
[[  7.48940807  13.41208607   8.02900384]
 [  0.          99.07366165 109.1400359 ]
 [-80.14272938   0.         -71.27583002]
 [  0.           0.         -24.62209631]]

Group sources and sensors for common manipulation using the `Collection` class.

>>> col = magpy.Collection(sens1, src2)
>>> print(sens1.position)
>>> print(src2.position)
>>> print(col.position)
[1. 2. 3.]
[1. 1. 1.]
[0. 0. 0.]

>>> col.move((1,1,1))
>>> print(sens1.position)
>>> print(src2.position)
>>> print(col.position)
[2. 3. 4.]
[2. 2. 2.]
[1. 1. 1.]

When all source and sensor objects are created and all paths are defined the `show()` (top level function and method of all Magpylib objects) provides a convenient way to graphically display the geometric arrangement through the Matplotlib,

>>> magpy.show(col)
---> graphic output from matplotlib

and and the Plotly graphic backend.

>>> magpy.show(col, backend='plotly')
---> graphic output from plotly
"""

# module level dunders
__version__ = '4.0.0-beta2'
__author__ =  'Michael Ortner & Alexandre Boissolet'
__credits__ = 'The Magpylib community'
__all__ = ['magnet', 'current', 'misc', 'getB', 'getH',
    'Sensor', 'Collection', 'show', 'defaults', '__version__',
    '__author__', '__credits__', 'core', 'display']

# create interface to outside of package
from magpylib import magnet, current, misc, core, display
from magpylib._src.defaults.defaults_classes import default_settings as defaults
from magpylib._src.fields import getB, getH
from magpylib._src.obj_classes import Sensor
from magpylib._src.obj_classes.class_Collection import Collection
from magpylib._src.display.display import show

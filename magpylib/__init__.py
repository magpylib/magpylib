# pylint: disable=line-too-long

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

Introduction (version 4.0.0)
----------------------------
Magpylib uses units of

    - [mT]: for the B-field and the magnetization (mu0*M).
    - [kA/m]: for the H-field.
    - [mm]: for all position inputs.
    - [deg]: for angle inputs by default.
    - [A]: for current inputs.

API: Magpylib objects
---------------------

The most convenient way to compute magnetic fields is through the object oriented interface. Magpylib objects represent magnetic field sources and sensors with various defining attributes.

>>> import magpylib as magpy
>>>
>>> # magnets
>>> src1 = magpy.magnet.Cuboid(magnetization=(0,0,1000), dimension=(1,2,3))
>>> src2 = magpy.magnet.Cylinder(magnetization=(0,1000,0), dimension=(1,2))
>>> src3 = magpy.magnet.CylinderSegment(magnetization=(0,1000,0), dimension=(1,2,2,45,90))
>>> src4 = magpy.magnet.Sphere(magnetization=(1000,0,0), diameter=1)
>>>
>>> # currents
>>> src5 = magpy.current.Circular(current=15, diameter=3)
>>> src6 = magpy.current.Line(current=15, vertices=[(0,0,0), (1,2,3)])
>>>
>>> # misc
>>> src7 = magpy.misc.Dipole(moment=(100,200,300))
>>>
>>> # sensor
>>> sens = magpy.Sensor()
>>>
>>> # print object representation
>>> for obj in [src1, src2, src3, src4, src5, src6, src7, sens]:
>>>     print(obj)
Cuboid(id=1331541150016)
Cylinder(id=1331541148672)
CylinderSegment(id=1331541762784)
Sphere(id=1331541762448)
Circular(id=1331543166304)
Line(id=1331543188720)
Dipole(id=1331543189632)
Sensor(id=1331642701760)

API: Position and orientation
-----------------------------

All Magpylib objects are endowed with ``position`` `(ndarray, shape (m,3))` and ``orientation`` `(scipy Rotation object, shape (m,3))` attributes that describe their state in a global coordinate system. Details on default object position (0-position) and alignment (unit-rotation) are found in the respective docstrings.

>>> import magpylib as magpy
>>> sens = magpy.Sensor()
>>> print(sens.position)
>>> print(sens.orientation.as_euler('xyz', degrees=True))
[0. 0. 0.]
[0. 0. 0.]

Manipulate position and orientation attributes directly through source attributes, or by using built-in ``move``, ``rotate`` or ``rotate_from_angax`` methods.

>>> import magpylib as magpy
>>> from scipy.spatial.transform import Rotation as R
>>>
>>> sens = magpy.Sensor(position=(1,1,1))
>>> print(sens.position)
>>>
>>> sens.move((1,1,1))
>>> print(sens.position)
>>>
[1. 1. 1.]
[2. 2. 2.]

>>> sens = magpy.Sensor(orientation=R.from_euler('x', 10, degrees=True))
>>> print(sens.orientation.as_euler('xyz'))
>>>
>>> sens.rotate(R.from_euler('x', 10, degrees=True)))
>>> print(sens.orientation.as_euler('xyz'))
>>>
>>> sens.rotate_from_angax(angle=10, axis=(1,0,0))
>>> print(sens.orientation.as_euler('xyz'))
[10 0. 0.]
[20 0. 0.]
[30 0. 0.]

Source position and orientation attributes can also represent complete source paths in the global coordinate system. Such paths can be generated conveniently using the ``move`` and ``rotate`` methods.

>>> import magpylib as magpy
>>>
>>> src = magpy.magnet.Cuboid(magnetization=(1,2,3), dimension=(1,2,3))
>>> src.move([(1,1,1),(2,2,2),(3,3,3),(4,4,4)], start='append')
>>> print(src.position)
[[0. 0. 0.]  [1. 1. 1.]  [2. 2. 2.]  [3. 3. 3.]  [4. 4. 4.]]

Details on rotation arguments, and how to conveniently generate complex paths are found in the docstings and some examples below.

API: Grouping objects with `Collection`
---------------------------------------

The top level class ``magpylib.Collection`` allows a user to group sources for common manipulation. A Collection functions like a list of source objects extended by Magpylib source methods: all operations applied to a Collection are applied to each source individually. Specific sources in the Collection can still be accessed and manipulated individually.

>>> import magpylib as magpy
>>>
>>> src1 = magpy.magnet.Cuboid(magnetization=(0,0,11), dimension=(1,2,3))
>>> src2 = magpy.magnet.Cylinder(magnetization=(0,22,0), dimension=(1,2))
>>> src3 = magpy.magnet.Sphere(magnetization=(33,0,0), diameter=2)
>>>
>>> col = magpy.Collection(src1, src2, src3)
>>> col.move((1,2,3))
>>> src1.move((1,2,3))
>>>
>>> for src in col:
>>>     print(src.position)
[2. 4. 6.]
[1. 2. 3.]
[1. 2. 3.]

Magpylib sources have addition and subtraction methods defined, adding up to a Collection, or removing a specific source from a Collection.

>>> import magpylib as magpy
>>>
>>> src1 = magpy.misc.Dipole(moment=(1,2,3))
>>> src2 = magpy.current.Circular(current=1, diameter=2)
>>> src3 = magpy.magnet.Sphere(magnetization=(1,2,3), diameter=1)
>>>
>>> col = src1 + src2 + src3
>>>
>>> for src in col:
>>>     print(src)
Dipole(id=2158565624128)
Circular(id=2158565622784)
Sphere(id=2158566236896)

>>> col - src1
>>>
>>> for src in col:
>>>     print(src)
Circular(id=2158565622784)
Sphere(id=2158566236896)

API: Graphic output with `display`
----------------------------------

When all source and sensor objects are created and all paths are defined ``display`` (top level function and method of all Magpylib objects) provides a convenient way to graphically view the geometric arrangement through Matplotlib.

>>> import magpylib as magpy
>>>
>>> # create a Collection of three sources
>>> s1 = magpy.magnet.Sphere(magnetization=(0,0,100), diameter=3, position=(3,0,0))
>>> s2 = magpy.magnet.Cuboid(magnetization=(0,0,100), dimension=(2,2,2), position=(-3,0,0))
>>> col = s1 + s2
>>>
>>> # generate a spiral path
>>> s1.move([(.2,0,0)]*100, increment=True)
>>> s2.move([(-.2,0,0)]*100, increment=True)
>>> col.rotate_from_angax([5]*100, 'z', anchor=0, increment=True, start=0)
>>>
>>> # display
>>> col.display(zoom=-.3, show_path=10)
---> graphic output

Various arguments like `axis, show_direction, show_path, size_sensors, size_direction, size_dipoles` and `zoom` can be used to customize the output and are described in the docstring in detail.

API: Field computation
----------------------

Field computation is done through the ``getB`` and ``getH`` function/methods. They always require `sources` and `observers` inputs. Sources are single Magpylib source objects, Collections or lists thereof.  Observers are arbitrary tensors of position vectors `(shape (n1,n2,n3,...,3))`, sensors or lists thereof. A most fundamental field computation example is

>>> from magpylib.magnet import Cylinder
>>>
>>> src = Cylinder(magnetization=(222,333,444), dimension=(2,2))
>>> B = src.getB((1,2,3))
>>> print(B)
[-2.74825633  9.77282601 21.43280135]

The magnetization input is in units of [mT], the B-field is returned in [mT], the H-field in [kA/m]. Field computation is also valid inside of the magnets.

>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> import magpylib as magpy
>>>
>>> # define Pyplot figure
>>> fig, [ax1,ax2] = plt.subplots(1, 2, figsize=(10,5))
>>>
>>> # define Magpylib source
>>> src = magpy.magnet.Cuboid(magnetization=(500,0,500), dimension=(2,2,2))
>>>
>>> # create a grid in the xz-symmetry plane
>>> ts = np.linspace(-3, 3, 30)
>>> grid = np.array([[(x,0,z) for x in ts] for z in ts])
>>>
>>> # compute B field on grid using a source method
>>> B = src.getB(grid)
>>> ampB = np.linalg.norm(B, axis=2)
>>>
>>> # compute H-field on grid using the top-level function
>>> H = magpy.getH(src, grid)
>>> ampH = np.linalg.norm(H, axis=2)
>>>
>>> # display field with Pyplot
>>> ax1.streamplot(grid[:,:,0], grid[:,:,2], B[:,:,0], B[:,:,2],
>>>     density=2, color=np.log(ampB), linewidth=1, cmap='autumn')
>>>
>>> ax2.streamplot(grid[:,:,0], grid[:,:,2], H[:,:,0], H[:,:,2],
>>>     density=2, color=np.log(ampH), linewidth=1, cmap='winter')
>>>
>>> # outline magnet boundary
>>> for ax in [ax1,ax2]:
>>>     ax.plot([1,1,-1,-1,1], [1,-1,-1,1,1], 'k--')
>>>
>>> plt.tight_layout()
>>> plt.show()
---> graphic output

The output of the most general field computation through the top level function ``magpylib.getB(sources, observers)`` is an ndarray of shape `(l,m,k,n1,n2,n3,...,3)` where `l` is the number of input sources, `m` the pathlength, `k` the number of sensors, `n1,n2,n3,...` the sensor pixel shape or shape of position vector and `3` the three magnetic field components `(Bx,By,Bz)`.

>>> import magpylib as magpy
>>>
>>> # three sources
>>> s1 = magpy.misc.Dipole(moment=(0,0,100))
>>> s2 = magpy.current.Circular(current=1, diameter=3)
>>> col = s1 + s2
>>>
>>> # two observers with 4x5 pixel
>>> pix = [[(1,2,3)]*4]*5
>>> sens1 = magpy.Sensor(pixel=pix)
>>> sens2 = magpy.Sensor(pixel=pix)
>>>
>>> # path of length 11
>>> s1.move([(1,1,1)]*11)
>>>
>>> B = magpy.getB([s1,s2,col], [sens1, sens2])
>>> print(B.shape)
(3, 11, 2, 5, 4, 3)

The object-oriented interface automatically vectorizes the computation for the user. Similar source types of multiple input-objects are automatically tiled up.

API: getB_dict and getH_dict
----------------------------

The ``magpylib.getB_dict`` and ``magpylib.getH_dict`` top-level functions avoid the object oriented interface, yet enable usage of the position/orientation implementations. The input arguments must be shape `(n,x)` vectors/lists/tuple. Static inputs e.g. of shape `(x,)` are automatically tiled up to shape `(n,x)`. Depending on the `source_type`, different input arguments are expected (see docstring for details).

>>> import magpylib as magpy
>>>
>>> # observer positions
>>> poso = [(0,0,x) for x in range(5)]
>>>
>>> # magnet dimensions
>>> dim = [(d,d,d) for d in range(1,6)]
>>>
>>> # getB_dict computation - magnetization is automatically tiled
>>> B = magpy.getB_dict(
>>>     source_type='Cuboid',
>>>     magnetization=(0,0,1000),
>>>     dimension=dim,
>>>     observer=poso)
>>> print(B)
[[  0.           0.         666.66666667]
 [  0.           0.         435.90578315]
 [  0.           0.         306.84039675]
 [  0.           0.         251.12200327]
 [  0.           0.         221.82226656]]

The ``getBH_dict`` functions can be up to 2 times faster than the object oriented interface. However, this requires that the user knows how to properly generate the vectorized input.

API: Direct access to field implementations
-------------------------------------------

For users who do not want to use the position/orientation interface, Magpylib offers direct access to the vectorized analytical implementations that lie at the bottom of the library through the ``magpylib.lib`` subpackage. Details on the implementations can be found in the respective function docstrings.

>>> import numpy as np
>>> import magpylib as magpy
>>>
>>> mag = np.array([(100,0,0)]*5)
>>> dim = np.array([(1,2,45,90,-1,1)]*5)
>>> poso = np.array([(0,0,0)]*5)
>>>
>>> B = magpy.lib.magnet_cyl_tile_H_Slanovc2021(mag, dim, poso)
>>> print(B)
[[   0.           0.        -186.1347833]
 [   0.           0.        -186.1347833]
 [   0.           0.        -186.1347833]
 [   0.           0.        -186.1347833]
 [   0.           0.        -186.1347833]]

As all input checks, coordinate transformations and position/orientation implementation are avoided, this is the fastest way to compute fields in Magpylib.

"""

# module level dunders
__version__ = '4.0.0'
__author__ =  'Michael Ortner & friends'
__credits__ = 'Silicon Austria Labs - Sensor Systems'
__all__ = ['magnet', 'current', 'misc', 'lib', 'getB', 'getH', 'getB_dict', 'getH_dict',
    'Sensor', 'Collection', 'display', 'Config', '__version__',
    '__author__', '__credits__']

# create interface to outside of package
from magpylib import magnet
from magpylib import current
from magpylib import misc
from magpylib import lib
from magpylib._lib.config import Config
from magpylib._lib.fields import getB, getH, getB_dict, getH_dict
from magpylib._lib.obj_classes import Collection, Sensor
from magpylib._lib.display import display

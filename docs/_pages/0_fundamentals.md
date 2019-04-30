# Fundamentals of magpylib

```eval_rst
The idea behind magpylib is to provide simple and easy to use classes for calculating magnetic fields. The core of the library is the :mod:`~magpylib.source` class which can be a permanent magnet, a current distributions or a magnetic moment. The library provides simple ways to generate such sources, to manipulate them geometrically, to group several sources into a :class:`~magpylib.Collection` and calculate the fields of such systems.
```

In this part of the documentation the fundamental structure of the magpylib library is detailed.

  - [Package Structure](#package-structure)
  - [Units and IO Types](#units-and-IO-types)
  - [The Source Class](#the-source-class)
    - [Variables and Initialization](#variables-and-initialization)
    - [Methods for Geometric Manipulation](#methods-for-geometric-manipulation)
  - [The Collection Class](#the-collection-class)
    - [Advanced Shapes with Collections](#advanced-shapes-with-collections)
  - [Math Package](#math-package)


## Package Structure

```eval_rst
The top level of magpylib contains the sub-packages :mod:`~magpylib.math` and :mod:`~magpylib.source` as well as the class :class:`magpylib.Collection`.

Within the :mod:`~magpylib.math` module several practical functions are provided. They include e.g. elementary geometric operations like rotations and their transformations between Euler-Angles and Quaternion representation.

The :mod:`~magpylib.source` module includes the core classes of the library, i.e. the magnetic sources. They are grouped in sub-packages :mod:`~magpylib.source.magnet`, :mod:`~magpylib.source.current` and :mod:`~magpylib.source.moment` which contain the respective source classes.

The :class:`magpylib.Collection` class offers an easy way of grouping multiple source objects for common manipulation.

.. currentmodule:: magpylib

.. image:: ../_static/images/summary/lib.png
   :align: center
   :scale: 75 %
```


## Units and IO Types

In magpylib all inputs and outputs are made in the physical units of **Millimeter** for lengths, **Degree** for angles and **Millitesla** for magnetization, magnetic moment and magnetic field and **Ampere** for currents.

The library is constructed so that any scalar input can be `int`, `float` or `numpy.float` type and any vector/matrix input can be given either in the form of a `list`, as a `tuple` or as a `numpy.array`.

The library output and all object variables are either of `np.float64` or `numpy.array64` type.


## The Source Class

This is the core class of the library. The idea is that source objects represent physical magnetic sources in cartesian three-dimensional space. They are characterized by the source type and the respective variables and can be manipulated by convenient methods as described below. The following source types are currently implemented.

```eval_rst
.. image:: ../_static/images/SourceTypes.JPG
   :align: center
   :scale: 75 %
```


### Variables and Initialization:

Different source types are characterized by different variables given through their mathematical representation.
```eval_rst
.. note::
  Detailed information about the variables of each specific source type and how to initialize them can be found in the docstrings.
```

The most fundamental properties of every source object `s` are position and orientation which are represented through the variables `s.position` (3D-array), `s.angle` (float) and `s.axis`(3D-array). If no values are specified, a source object is initialized by default with `position=(0,0,0)`, and **init orientation** defined to be `angle=0` and `axis=(0,0,1)`.

The `position` generally refers to the geometric center of the source while the orientation (`angle`,`axis`) refers to a rotation of the source by `angle` about `axis` anchored at `position` RELATIVE TO the **init orientation**. The **init orientation** generally refers to sources standing upright (see previous image), oriented along the cartesian coordinates axes.

The source geometry is generally described by the `dimension` variable. However, as each source requires different input parameters the format is always different.

Magnet sources represent homogeneously magnetized permanent magnets. The magnetization vector is described by the `magnetization` variable which is always a 3D-array indicating direction and magnitude. The current sources represent line currents. They require a scalar `current` input. The moment class represents a magnetic dipole moment which requires a `moment` (3D-array) input.

```eval_rst
.. note::
  For convenience `magnetization`, `current`, `dimension`, `position` are initialized through the keywords *mag*, *curr*, *dim* and *pos*.
```

The following code shows how to initialize a source object, a permanent magnet D4H5 cylinder with diagonal magnetization, positioned with the center in the origin, standing upright with axis in z-direction.

```python
import magpylib as magpy

pm = magpy.source.magnet.Cylinder( mag = [500,0,500], # The magnetization vector in mT.
                                   dim = [4,5])       # dimension (diameter,height) in mm.
                                                      # no pos, angle, axis specified so default values are used

print(pm.magnetization)  # Output: [500. 0. 500.]
print(pm.dimension)      # Output: [4. 5.]
print(pm.position)       # Output: [0. 0. 0.]
print(pm.angle)          # Output: 0.0
print(pm.axis)           # Output: [0. 0. 1.]
```


### Methods for Geometric Manipulation

In most cases we want to move the magnet to a designated position, orient it in a desired way or change its dimension dynamically. There are several ways to achieve this, each with advantages and disadvantages:

```eval_rst
At initialization:
  When initializing the source we can set all variables as desired.

Manipulation after initialization: 
  We initialize the source and manipulate it afterwards,
  1. By directly setting the source variables to desired values
  2. By using provided methods of manipulation
```

The source class provides a set of methods for convenient geometric manipulation. The methods include `setPosition`and `move` for translation of the objects as well as `setOrientation` and `rotate` for rotation operations. These methods are implementations of the respective geometric operations. Upon application to source objects they will simply modify the object variables accordingly.


+------------------+-------------------------------------+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
|  Method name     | Argument Type                       | Argument Designation | Description of the method                                                                                                                     |
+==================+=====================================+======================+===============================================================================================================================================+
| `setPosition`    | 3D-vector                           | position Vector      | Moves the object to a desiganted position given by the inpVector. **s.position -> inpVector**                                                 |
+------------------+-------------------------------------+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| `move`           | 3D-vector                           | position Vector      | Moves the object BY the inpVector. **s.position -> s.position + inpVector**                                                                   |
+------------------+-------------------------------------+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| `setOrientation` | scalar, 3D-vector                   | angle, axis          | Changes object orientation to given input values (inpAngle,inpAxis). **s.angle -> inpAngle, s.axis -> inpAxis**                               |
+------------------+-------------------------------------+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------+
| `rotate`         | scalar, 3D-vector, anchor=3D-vector | angle, axis, anchor  | This method rotates the object by angle about axis anchored at anchor. As a result position and orientation variables                         |
|                  |                                     |                      | are changed. If no value for anchor is specified, the anchor is set to object position, which means that the object rotates about itself.     |
+------------------+-------------------------------------+----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------+

The following videos graphically show the application of the four methods for geometric manipulation.

<i><p align="center" style="font-weight: 600;"> Rotating and Orienting </p></i>

```eval_rst

|rotate| |setOrientation|

.. |setOrientation| image:: ../_static/images/fundamentals/setOrientation.gif
   :width: 45%

.. |rotate| image:: ../_static/images/fundamentals/rotate.gif
   :width: 45%

```

<i><p align="center" style="font-weight: 600;"> Moving and Positioning </p></i>

```eval_rst

|move| |setPosition|

.. |setPosition| image:: ../_static/images/fundamentals/setPosition.gif
   :width: 45%

.. |move| image:: ../_static/images/fundamentals/move.gif
   :width: 45%
```


### Calculating the Magnetic Field

field only from the source addressed

superposition principle

getBsweep


## The Collection Class

The linear nature of the field equations utilized **provides a superposition principle**. This means that **arbitrary magnet compounds can be generated** by “Union” and “Difference” operations.

To **group and display** Source Objects or to **perform group rotations and compound analysis**, the Collection Class is utilized. Otherwise, source objects will not interact.

Collections can be utilized in many ways, and may include other Collections inside of themselves. 

The following animation shows the creation of a 5mm long coil with 0.1mm spacing between each turn, constituting 50 turns.
The Coil **Collection is then moved and rotated** in two axes (Y and Z Tilt). 

```eval_rst
.. image:: ../_static/images/fundamentals/collectionExample.gif
   :align: center
```

The coils are defined as having 10 Amps running through each. 
The electromagnetic field **analysis of the compounded objects** looks like the following:

```eval_rst
.. image:: ../_static/images/fundamentals/collectionAnalysis.png
   :align: center
```

Movement may also be realized with the use of an **anchored pivot point**.

```eval_rst
.. image:: ../_static/images/fundamentals/pivot.gif
   :align: center
```

```eval_rst
.. image:: ../_static/images/fundamentals/pivotAnalysis.gif
   :align: center
```

### Advanced Shapes with Collections

Complex magnet formations may be created due to the superposition principle, where magnets of complex shapes are defined by Collections of basic ones.

Magnets with holes may be described by adding sources of conflicting magnetism inside of other sources.


## Math Package

The Math Package is utilized to assist users in performing Angle/Axis conversions and rotations utilizing [the quaternion notation](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation).

[paper]: http://mystery-404.herokuapp.com
# Library Documentation

```eval_rst
The idea behind magpylib is to provide simple and easy to use classes for calculating magnetic fields. The core of the library is the :mod:`~magpylib.source` class which can represent permanent magnets, current distributions or magnetic moments. The library provides simple ways to generate such source objects, to manipulate them geometrically, to group several sources into a :class:`~magpylib.Collection` and to calculate the fields of such systems.
```

In this part of the documentation the fundamental structure of the magpylib library is detailed.

  - [Package Structure](#package-structure)
  - [Units and IO Types](#units-and-io-types)
  - [The Source Class](#the-source-class)
    - [Variables and Initialization](#variables-and-initialization)
    - [Methods for Geometric Manipulation](#methods-for-geometric-manipulation)
    - [Calculating the Magnetic Field](#calculating-the-magnetic-field)
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
   :scale: 70 %
```


## Units and IO Types

In magpylib all inputs and outputs are made in the physical units of **Millimeter** for lengths, **Degree** for angles and **Millitesla** for magnetization, magnetic moment and magnetic field and **Ampere** for currents. Details about how the solutions are set up can be found in the [Physics section](9_physics.md).

The library is constructed so that any scalar input can be `int`, `float` or `numpy.float` type and any vector/matrix input can be given either in the form of a `list`, as a `tuple` or as a `numpy.array`.

The library output and all object variables are either of `np.float64` or `numpy.array64` type.


## The Source Class

This is the core class of the library. The idea is that source objects represent physical magnetic sources in cartesian three-dimensional space. They are characterized by the source type and the respective variables and can be manipulated by convenient methods as described below. The following source types are currently implemented.

```eval_rst
.. image:: ../_static/images/SourceTypes.JPG
   :align: center
   :scale: 60 %
```

The source class provides a rich collection of variables and methods that describe the sources, can be used for geometric manipulation and calculating the magnetic fields. They are described in detail in the following sections. The following figure gives a graphical overview.

```eval_rst
.. image:: ../_static/images/sourceVars_Methods.JPG
   :align: center
   :scale: 60 %
```

### Variables and Initialization:

Different source types are characterized by different variables given by their mathematical representation. 



The most fundamental properties of every source object `s` are position and orientation which are represented through the variables `s.position` (3D-array), `s.angle` (float) and `s.axis`(3D-array). If no values are specified, a source object is initialized by default with `position=(0,0,0)`, and **init orientation** defined to be `angle=0` and `axis=(0,0,1)`. The **init orientation** generally refers to sources standing upright (see previous image), oriented along the cartesian coordinates axes.

The `position` generally refers to the geometric center of the source. The orientation (`angle`,`axis`) refers to a rotation of the source RELATIVE TO the **init orientation** about an axis specified by the `axis` vector anchored at the source `position`. The angle of this rotation is given by the `angle` variable.

```eval_rst
.. image:: ../_static/images/source_Orientation.JPG
   :align: center
   :scale: 60 %
```

The source geometry is generally described by the `dimension` variable. However, as each source requires different input parameters the format is always different.

Magnet sources represent homogeneously magnetized permanent magnets. The magnetization vector is described by the `magnetization` variable which is always a 3D-array indicating direction and magnitude. The current sources represent line currents. They require a scalar `current` input. The moment class represents a magnetic dipole moment which requires a `moment` (3D-array) input.


```eval_rst
.. note::
Detailed information about the source parameters of each specific source type and how to initialize them can be found in the respecive class docstrings accessible through your IDE.
```

```eval_rst
.. note::
  For convenience `magnetization`, `current`, `dimension`, `position` are initialized through the keywords *mag*, *curr*, *dim* and *pos*.
```

The following code shows how to initialize a source object, a permanent magnet D4H5 cylinder with diagonal magnetization, positioned with the center in the origin, standing upright with axis in z-direction.

```python
import magpylib as magpy

s = magpy.source.magnet.Cylinder( mag = [500,0,500], # The magnetization vector in mT.
                                   dim = [4,5])       # dimension (diameter,height) in mm.
                                                      # no pos, angle, axis specified so default values are used

print(s.magnetization)  # Output: [500. 0. 500.]
print(s.dimension)      # Output: [4. 5.]
print(s.position)       # Output: [0. 0. 0.]
print(s.angle)          # Output: 0.0
print(s.axis)           # Output: [0. 0. 1.]
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

- `s.setPosition(newPos)`: Moves the source to the position given by the argument vector (*newPos*. *s.position -> newPos*)
- `s.move(displacement)`: Moves the source by the argument vector *displacement*. (*s.position -> s.position + displacement*) 
- `s.setOrientation(angle,axis)`: This method sets a new source orientation given by *angle* and *axis*. (*s.angle -> angle, s.axis -> axis*)
- `s.rotate(angle,axis,anchor=self.position)`: Rotates the source object by *angle* about the axis *axis* which passes through a position given by *anchor*. As a result position and orientation variables are modified. If no value for anchor is specified, the anchor is set to the object position, which means that the object rotates about itself.

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

Once a source object `s` is defined one can calculate the magnetic field generated by it using the two methods `getB` and `getBsweep`. Here the call `s.getB(pos)` simply returns the value of the field which is generated by the source *s* at the sensor position *pos*.

In most cases, however, one will be interested to determine the field for a set of sensor positions, or for different magnet positions and orientations. While this can manually be achieved by looping `getB`, magpylib also provides the advanced method `s.getBsweep(input)` for ease of use and for the possibility of parallelization (to be discussed below). Here *input* can have two possible formats:
1. *input* is a list of *N* sensor positions. In this case the magnetic field of the source is determined for all *N* sensor positions and returned in an *Nx3* matrix.
2. *input* is a list of the following format [(sensorPos1, sourcePos1, sourceOrient1),...]. Here for each case of sensor position and source state the field is evaluated and returned in an *Nx3* matrix. This corresponds to a system where sensor and magnet move simultaneously.

To calculate the fields, magpylib uses mostly analytical expressions that can be found in the literature. A detailed analysis of the precision and applicability of this solution can be found in the [Physics section](9_physics.md).

```eval_rst
.. note::
  It is critical to note that the analytical solution does not treat interaction between the sources. This means that the total magnetic field is simply given by the superposition of the field of each source, and each source can be evaluate individually.
```


## The Collection Class


#### Common Manipulation

The idea behind the collection class is to group multiple source objects for common evaluation and manipulation. In principle a collection is simply a list of source objects. An operation that is applied to such a collection is applied to each object within the collection. This includes geometric manipulation through `setPosition`, `move`, `setOrientation` and `rotate`, but also evaluation of the total magnetic field  using `getB` and `getBsweep`. All of these methods are also methods of the collection class.

<i><p align="center" style="font-weight: 600;"> Grouping Sources in Collections </p></i>
```eval_rst

|Collection| |total Field|

.. |Collection| image:: ../_static/images/fundamentals/collectionExample.gif
   :width: 45%

.. |total Field| image:: ../_static/images/fundamentals/collectionAnalysis.png
   :width: 45%

```

#### Constructing Collections and Display

Collections can be constructed at initialization by simply giving the sources as arguments. It is possible not only to add sources, but also to add lists od sources as well as other collections. With the default kwarg *dupWarning=True*, a warning will be displayed if one source object has been added multiple times to the collection. In this case an operation applied to the collection will be applied multiple times to that source.

In addition, the collection class features methods to add and remove sources for command line like manipulation. The method `coll.addSources(*sources)` will add all sources given to it to the collection `coll`. The method `coll.removeSource(ref)` will remove the referenced source from the collection. Here *ref* can be either a source or an integer indicating the reference position in the collection.

Finally, collections provide a method for graphical display of the system, termed `coll.displaySystem()`. The idea is to quickly check the geometry of the source assembly. For convenient display the *displaySystem* method has three * *kwargs*. The *marker* arg to provide markers for specific reference positions of interest, the *supress* arg which suppresses the figure display so that only the figure object is returned and the *direct* arg which additionally displays current and magnetization directions in the figure. The following example code shows how a collection is initialized and displayed.

```python
import magpylib as magpy

s1 = magpy.source.magnet.Cylinder( mag = [0,0,1],dim = [4,5])
s2 = magpy.source.magnet.Cylinder( mag = [0,1,1],dim = [3,4])
s3 = magpy.source.magnet.Cylinder( mag = [1,0,1],dim = [2,3])

coll = magpy.Collection(s1,s2)
coll.addSources(s3,s1)
coll.removeSource(3)

WORK IN PROGRESSS

```



## Math Package

The Math Package is utilized to assist users in performing Angle/Axis conversions and rotations utilizing [the quaternion notation](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation).

[paper]: http://mystery-404.herokuapp.com
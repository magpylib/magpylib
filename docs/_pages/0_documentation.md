# Library Documentation
```eval_rst
The idea behind magpylib is to provide simple and easy to use classes for calculating magnetic fields. The core of the library is the :mod:`~magpylib.source` class which can represent permanent magnets, current distributions or magnetic moments. The library provides simple ways to generate such source objects, to manipulate them geometrically, to group several sources into a :class:`~magpylib.Collection` and to calculate the fields of such systems.
```

In this part of the documentation the fundamental structure of the magpylib library is detailed.

  - [Package Structure](#package-structure)
  - [Units and IO Types](#units-and-io-types)
  - [The Source Class](#the-source-class)
    - [Attributes and Keyword Initialization](#attributes-and-keyword-initialization)
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

.. image:: ../_static/images/documentation/lib_structure.JPG
   :align: center
   :scale: 50 %
```
<i><p align="center" style="font-weight: 100; font-size: 10pt"> <b>Figure:</b> Outline of the magpylib package structure. </p></i>

## Units and IO Types

In magpylib all inputs and outputs are made in the physical units of **Millimeter** for lengths, **Degree** for angles and **Millitesla** for magnetization, magnetic moment and magnetic field and **Ampere** for currents. Details about how the solutions are set up can be found in the [Physics section](9_physics.md).

The library is constructed so that any scalar input can be `int`, `float` or `numpy.float` type and any vector/matrix input can be given either in the form of a `list`, as a `tuple` or as a `numpy.array`.

The library output and all object attributes are either of `np.float64` or `numpy.array64` type.


## The Source Class

This is the core class of the library. The idea is that source objects represent physical magnetic sources in cartesian three-dimensional space. The following source types are currently implemented in magpylib.

```eval_rst
.. image:: ../_static/images/documentation/SourceTypes.JPG
   :align: center
   :scale: 65 %
```
<i><p align="center" style="font-weight: 100; font-size: 10pt"> <b>Figure:</b> Source types currently available in magpylib. </p></i>

The different source types contain various attributes and methods. The attributes characterize the source (e.g. position) while the methods can be used for geometric manipulation and calculating the magnetic fields. They are described in detail in the following sections. The figure below gives a graphical overview.

```eval_rst
.. image:: ../_static/images/documentation/sourceVars_Methods.JPG
   :align: center
   :scale: 60 %
```
<i><p align="center" style="font-weight: 100; font-size: 10pt"> <b>Figure:</b> Illustration of attributes and methods of the source class. </p></i>

### Attributes and Keyword Initialization:

The most fundamental properties of every source object `s` are position and orientation which are represented through the attributes `s.position` (3D-array), `s.angle` (float) and `s.axis`(3D-array). If no values are specified, a source object is initialized by default with `position=(0,0,0)`, and **init orientation** defined to be `angle=0` and `axis=(0,0,1)`. 

Due to their different nature each source type is characterized by different attributes. However, in general the `position` attribute refers to the position of the geometric center of the source. The **init orientation** generally defines sources standing upright oriented along the cartesian coordinates axes, see e.g. the following image. 

An orientation given by (`angle`,`axis`) refers to a rotation of the source RELATIVE TO the **init orientation** about an axis specified by the `axis` vector anchored at the source `position`. The angle of this rotation is given by the `angle` attribute. Mathematically, every possible orientation can be expressed by such a single angle-axis rotation. For easier use of the angle-axis rotation and transformation to Euler angles the [math package](#math-package) provides some useful methods. 

```eval_rst
.. image:: ../_static/images/documentation/source_Orientation.JPG
   :align: center
   :scale: 50 %
```
<i><p align="center" style="font-weight: 100; font-size: 10pt"> <b>Figure:</b> Illustration of the angle-axis system for source orientations. </p></i>

The source geometry is generally described by the `dimension` attribute. However, as each source requires different input parameters the format is always different.

Magnet sources represent homogeneously magnetized permanent magnets. The magnetization vector is described by the `magnetization` attribute which is always a 3D-array indicating direction and magnitude. The current sources represent line currents. They require a scalar `current` input. The moment class represents a magnetic dipole moment which requires a `moment` (3D-array) input.


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
```eval_rst
.. image:: ../_static/images/documentation/Source_Display.JPG
   :align: center
   :scale: 50 %
```
<i><p align="center" style="font-weight: 100; font-size: 10pt"> <b>Figure:</b> Magnet geometry created by above code. </p></i>

### Methods for Geometric Manipulation

In most cases we want to move the magnet to a designated position, orient it in a desired way or change its dimension dynamically. There are several ways to achieve this, each with advantages and disadvantages:

```eval_rst
At initialization:
  When initializing the source we can set all attributes as desired.

Manipulation after initialization: 
  We initialize the source and manipulate it afterwards,
  
  1. By directly setting the source attributes to desired values.
  
  2. By using provided methods of manipulation.
```

The source class provides a set of methods for convenient geometric manipulation. The methods include `setPosition`and `move` for translation of the objects as well as `setOrientation` and `rotate` for rotation operations. These methods are implementations of the respective geometric operations. Upon application to source objects they will simply modify the object attributes accordingly.

- `s.setPosition(newPos)`: Moves the source to the position given by the argument vector (*newPos*. *s.position -> newPos*)
- `s.move(displacement)`: Moves the source by the argument vector *displacement*. (*s.position -> s.position + displacement*) 
- `s.setOrientation(angle,axis)`: This method sets a new source orientation given by *angle* and *axis*. (*s.angle -> angle, s.axis -> axis*)
- `s.rotate(angle,axis,anchor=self.position)`: Rotates the source object by *angle* about the axis *axis* which passes through a position given by *anchor*. As a result position and orientation attributes are modified. If no value for anchor is specified, the anchor is set to the object position, which means that the object rotates about itself.

The following videos graphically show the application of the four methods for geometric manipulation.

<i><p align="center" style="font-weight: 600;"> move and setPosition </p></i>

```eval_rst

|move| |setPosition|

.. |setPosition| image:: ../_static/images/documentation/setPosition.gif
   :width: 45%

.. |move| image:: ../_static/images/documentation/move.gif
   :width: 45%
```

<i><p align="center" style="font-weight: 600;"> rotate and setOrientation </p></i>

```eval_rst

|rotate| |setOrientation|

.. |setOrientation| image:: ../_static/images/documentation/setOrientation.gif
   :width: 45%

.. |rotate| image:: ../_static/images/documentation/rotate.gif
   :width: 45%

```

### Calculating the Magnetic Field

Once a source object `s` is defined one can calculate the magnetic field generated by it using the two methods `getB` and `getBsweep`. Here the call `s.getB(pos)` simply returns the value of the field which is generated by the source *s* at the sensor position *pos*.

In most cases, however, one will be interested to determine the field for a set of sensor positions, or for different magnet positions and orientations. While this can manually be achieved by looping `getB`, magpylib also provides the advanced method `s.getBsweep(INPUT)` for ease of use and for the possibility of parallelization (to be discussed below). Here *INPUT* can have two possible formats:
1. *INPUT TYPE 1* is a list of *N* sensor positions. In this case the magnetic field of the source is determined for all *N* sensor positions and returned in an *Nx3* matrix.
2. *INPUT TYPE 2* is a list of the following format [(sensorPos1, sourcePos1, sourceOrient1),...]. Here for each case of sensor position and source state the field is evaluated and returned in an *Nx3* matrix. This corresponds to a system where sensor and magnet move simultaneously.

```eval_rst

|sweep1| |sweep2|

.. |sweep1| image:: ../_static/images/documentation/sweep1.gif
   :width: 45%

.. |sweep2| image:: ../_static/images/documentation/sweep2.gif
   :width: 45%
```
<i><p align="center" style="font-weight: 100; font-size: 10pt"> <b>Figure:</b> Illustrations of the two getBsweep input types. </p></i>

To calculate the fields, magpylib uses mostly analytical expressions that can be found in the literature. A detailed analysis of the precision and applicability of this solution can be found in the [Physics section](9_physics.md).

```eval_rst
.. note::
  It is critical to note that the analytical solution does not treat interaction between the sources. This means that even if multiple sources are defined, `s.getB` will return only the unperturbed field from the source `s`. The total magnetic field is simply given by the superposition of the fields of all sources.
```


## The Collection Class

The idea behind the collection class is to group multiple source objects for common evaluation and manipulation.

#### Constructing Collections

In principle a collection `c` is simply a list of source objects that are collected in the attribute `c.sources`.

Collections can be constructed at initialization by simply giving the sources as arguments. It is possible to add single sources, lists of multiple sources and even other collection objects. All sources are simply added to the `sources` attribute of the target collection.

With the collection kwarg `dupWarning=True`, adding multiples of the same source will be blocked, and a warning will be displayed informing the user that a source object is already in the collection's `source` attribute. This can be unblocked by providing the `dupWarning=False` kwarg.

In addition, the collection class features methods to add and remove sources for command line like manipulation. The method `c.addSources(*sources)` will add all sources given to it to the collection `c`. The method `c.removeSource(ref)` will remove the referenced source from the collection. Here the `ref` argument can be either a source or an integer indicating the reference position in the collection, and it defaults to the latest added source in the Collection.

#### Common Manipulation and total Magnetic Field

```eval_rst
All methods for geometric operations (`setPosition`, `move`, `setOrientation` and `rotate`) are also methods of the collection class. A geometric operation applied to a collection is applied to each object within that collection individually. 

The :meth:`~magpylib.Collection.getB` and :meth:`~magpylib.Collection.getBsweep` methods for calculating magnetic field are also present, calculating the total magnetic field generated by all sources in the collection. 

``` 

<!--
<i><p align="center" style="font-weight: 600;"> Grouping Sources in Collections, Common Manipulation and Total Field </p></i> 
-->
```eval_rst

|Collection| |total Field|

.. |Collection| image:: ../_static/images/documentation/collectionExample.gif
   :width: 45%

.. |total Field| image:: ../_static/images/documentation/collectionAnalysis.png
   :width: 50%
```
<i><p align="center" style="font-weight: 100; font-size: 10pt"> <b>Figure:</b>
Circular current sources are grouped into a collectin to form a coil. The whole
coil is then geometrically manipulated and the total magnetic field is shown in
the xz-plane </p></i>

```eval_rst

.. note::
   Due to abstraction restraints, Collections' :meth:`~magpylib.Collection.getBsweep` can only be called using `INPUT Type 1 <#calculating-the-magnetic-field>`_. 
  
```

#### Display Collection Graphically

Finally, the collection class provides the method `displaySystem` to quickly check the geometry of the source assembly. 
It comes with three keyword arguments (kwargs): 

- `markers=listOfPos` - For displaying reference positions defined in a list.
- `suppress=True` - For suppressing the figure output once `matplotlib.pyplot.ioff()` is set.
- `direc=True` - For displaying current and magnetization directions in the figure. 

The following example code shows how a collection is initialized and displayed.

```python
import magpylib as magpy

s1 = magpy.source.magnet.Cylinder( mag = [0,0,1],dim = [4,5], pos = [-10,0,0])
s2 = magpy.source.magnet.Box( mag = [0,1,1],dim = [3,4,5], pos = [0,0,0])
s3 = magpy.source.magnet.Sphere( mag = [1,0,1],dim = 4, pos = [10,0,0])

c = magpy.Collection(s1,s2,s3)

c.rotate(45,[0,0,1],anchor=[0,0,0])
listOfPos = [(0,0,6),(10,10,10),(-10,-10,-10)]
c.displaySystem(markers=listOfPos)
```
```eval_rst
.. image:: ../_static/images/documentation/Collection_Display.JPG
   :align: center
   :scale: 60 %
```
<i><p align="center" style="font-weight: 100; font-size: 10pt"> <b>Figure:</b> Code output. </p></i>

## Math Package

The math package provides some functions for easier use of the angle-axis (Quaternion) rotation used in magpylib. 

- `anglesFromAxis(axis)`: This function takes an arbitrary `axis` argument (3-vector) and returns its orientation given by the angles (PHI,THETA)) that are defined as in spherical coordinates. PHI is the azimuth angle and THETA is the polar angle.
  ```python
  import magpylib as magpy
  angles = magpy.math.anglesFromAxis([1,1,0])
  print(angles)                             #Output = [45. 90.]
  ```

- `axisFromAngles(angles)`: This function generates an axis (3-vector) from the `angles` input (PHI,THETA) where PHI is the azimuth angle and THETA is the polar angle of a spherical coordinate system.
  ```python
  import magpylib as magpy
  ax = magpy.math.axisFromAngles([90,90])
  print(ax)                                 #Output = [0.0 1.0 0.0]
  ```

- `randomAxis()`: Designed for Monte Carlo simulations, this function returns a random axis of length 1 with equal angular distribution.
  ```python
  import magpylib as magpy
  ax = magpy.math.randomAxis()
  print(ax)                                 #Output = [-0.24834468  0.96858637  0.01285925]
  ```

- `rotatePosition(pos, angle, axis, anchor=[0,0,0])`: This function uses angle-axis rotation to rotate the position vector `pos` by the `angle` argument about an axis defined by the `axis` vector which passes through the `anchor` position.
  ```python
  import magpylib as magpy
  pos0 = [1,1,0]
  angle = -90
  axis = [0,0,1]
  anchor = [1,0,0]
  positionNew = magpy.math.rotatePosition(pos0,angle,axis,anchor)
  print(positionNew)                  #Output = [2. 0. 0.]
  ```

# Library Documentation
```eval_rst
The idea behind magpylib is to provide simple and easy to use classes for calculating magnetic fields. The core of the library is the :mod:`~magpylib.source` class which can represent permanent magnets, current distributions or magnetic moments. The library provides simple ways to generate such source objects, to manipulate them geometrically, to group several sources into a :class:`~magpylib.Collection` and to calculate the fields of such systems.
```

In this part of the documentation the fundamental structure of the magpylib library is detailed.

- [Library Documentation](#Library-Documentation)
  - [Package Structure](#Package-Structure)
  - [Units and IO Types](#Units-and-IO-Types)
  - [The Source Class](#The-Source-Class)
    - [Source Attributes and Initialization](#Source-Attributes-and-Initialization)
      - [Position and Orientation](#Position-and-Orientation)
      - [Geometry and Excitation](#Geometry-and-Excitation)
    - [Methods for Geometric Manipulation](#Methods-for-Geometric-Manipulation)
    - [Calculating the Magnetic Field](#Calculating-the-Magnetic-Field)
  - [The Collection Class](#The-Collection-Class)
      - [Constructing Collections](#Constructing-Collections)
      - [Common Manipulation and total Magnetic Field](#Common-Manipulation-and-total-Magnetic-Field)
      - [Complex Magnet Geometries](#Complex-Magnet-Geometries)
      - [Display Collection Graphically](#Display-Collection-Graphically)
  - [Math Package](#Math-Package)

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

In magpylib all inputs and outputs are made in the physical units of

- **Millimeter** for lengths
- **Degree** for angles
- **Millitesla** for magnetization, magnetic moment and magnetic field,
- **Ampere** for currents.

Details about how the solutions are set up can be found in the [Physics section](9_physics.md).

The library is constructed so that any

- **scalar input** can be `int`, `float` or of `numpy.float` type
- **vector/matrix input** can be given either in the form of a `list`, as a `tuple` or as a `numpy.array`.

The library output and all object attributes are either of `numpy.float64` or `numpy.array64` type.

## The Source Class

This is the core class of the library. The idea is that source objects represent physical magnetic sources in Cartesian three-dimensional space and their respective fields can be calculated. The following source types are currently implemented in magpylib.

```eval_rst
.. image:: ../_static/images/documentation/SourceTypes.JPG
   :align: center
   :scale: 65 %
```
<i><p align="center" style="font-weight: 100; font-size: 10pt"> <b>Figure:</b> Source types currently available in magpylib. </p></i>

The different source types contain various attributes and methods. The attributes characterize the source (e.g. position) while the methods can be used for geometric manipulation and calculating the magnetic fields. They are described in detail in the docstrings

```eval_rst
 - :mod:`~magpylib.source.magnet.Box`
 - :mod:`~magpylib.source.magnet.Cylinder`
 - :mod:`~magpylib.source.magnet.Box`
 - :mod:`~magpylib.source.current.Line`
 - :mod:`~magpylib.source.current.Circular`
 - :mod:`~magpylib.source.moment.Dipole`
```

and in the following sections. The figure below gives a graphical overview.

```eval_rst
.. image:: ../_static/images/documentation/sourceVars_Methods.JPG
   :align: center
   :scale: 60 %
```
<i><p align="center" style="font-weight: 100; font-size: 10pt"> <b>Figure:</b> Illustration of attributes and methods of the source class. </p></i>

### Source Attributes and Initialization

#### Position and Orientation

The most fundamental properties of every source object `s` are position and orientation which are represented through the attributes `s.position` (3D-array), `s.angle` (float) and `s.axis`(3D-array). If no values are specified, a source object is initialized by default with `position=(0,0,0)`, and **init orientation** defined to be `angle=0` and `axis=(0,0,1)`.

Due to their different nature each source type is characterized by different attributes. However, in general the `position` attribute refers to the position of the geometric center of the source. The **init orientation** generally defines sources standing upright oriented along the Cartesian coordinates axes, see e.g. the following image.

An orientation given by (`angle`,`axis`) refers to a rotation of the source RELATIVE TO the **init orientation** about an axis specified by the `axis` vector anchored at the source `position`. The angle of this rotation is given by the `angle` attribute. Mathematically, every possible orientation can be expressed by such a single angle-axis rotation. For easier use of the angle-axis rotation and transformation to Euler angles the [math package](#math-package) provides some useful methods. 

```eval_rst
.. image:: ../_static/images/documentation/source_Orientation.JPG
   :align: center
   :scale: 50 %
```
<i><p align="center" style="font-weight: 100; font-size: 10pt"> <b>Figure:</b> Illustration of the angle-axis system for source orientations. </p></i>

#### Geometry and Excitation

While position and orientation have default values, a source is defined through its geometry (e.g. Cylinder) and excitation (e.g. Magnetization Vector) which must be initialized by hand.

The source geometry is generally described by the `dimension` attribute. However, as each source requires different input parameters, the format is always different. Detailed information about the attributes of each specific source type and how to initialize them can be found in the respective class docstrings:

```eval_rst
:mod:`~magpylib.source.magnet.Box`, :mod:`~magpylib.source.magnet.Cylinder`, :mod:`~magpylib.source.magnet.Box`, :mod:`~magpylib.source.current.Line`, :mod:`~magpylib.source.current.Circular`, :mod:`~magpylib.source.moment.Dipole` 
```

The excitation is either the magnet magnetization, the current or the magnetic moment. Magnet sources represent homogeneously magnetized permanent magnets (other types with radial or multipole magnetization are not implemented at this point). The magnetization vector is described by the `magnetization` attribute which is always a 3D-array indicating direction and magnitude. The magnetization vector is always given with respect to the INIT ORIENTATION of the magnet. The current sources represent line currents. They require a scalar `current` input. The moment class represents a magnetic dipole moment which requires a `moment` (3D-array) input.

```eval_rst
.. note::
  For convenience `magnetization`, `current`, `dimension`, `position` are initialized through the keywords *mag*, *curr*, *dim* and *pos*.
```

The following code shows how to initialize a source object, a D4H5 permanent magnet cylinder with diagonal magnetization, positioned with the center in the origin, standing upright with axis in z-direction.

```python
from magpylib.source.magnet import Cylinder

s = Cylinder( mag = [500,0,500], # The magnetization vector in mT.
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

<i><p align="center" style="font-weight: 100; font-size: 10pt"> <b>Figure:</b> Magnet geometry created by above code: A cylinder which stands upright with geometric center at the origin. </p></i>

### Methods for Geometric Manipulation

In most cases we want to move the magnet to a designated position, orient it in a desired way or change its dimension dynamically. There are several ways to achieve this:

**At initialization:**
When initializing the source we can set all attributes as desired. So instead of 'moving' one source around one could create a new one for each set of parameters of interest.

**Manipulation after initialization**
We initialize the source and manipulate it afterwards as desired by

 1. directly setting the source attributes.
 2. using provided methods of manipulation.

The latter is often the most practical and intuitive way. To this end the source class provides a set of methods for convenient geometric manipulation. The methods include `setPosition` and `move` for translation of the objects as well as `setOrientation` and `rotate` for rotation operations. Upon application to source objects they will simply modify the object attributes accordingly.

- `s.setPosition(newPos)`: Moves the source to the position given by the argument vector (*newPos*. *s.position -> newPos*)
- `s.move(displacement)`: Moves the source by the argument vector *displacement*. (*s.position -> s.position + displacement*) 
- `s.setOrientation(angle,axis)`: This method sets a new source orientation given by *angle* and *axis*. (*s.angle -> angle, s.axis -> axis*)
- `s.rotate(angle,axis,anchor=self.position)`: Rotates the source object by *angle* about the axis *axis* which passes through a position given by *anchor*. As a result position and orientation attributes are modified. If no value for anchor is specified, the anchor is set to the object position, which means that the object rotates about itself.

The following videos show the application of the four methods for geometric manipulation.

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

The following example code shows how geometric operations are applied to source objects.

```python
from magpylib.source.magnet import Cylinder

s = Cylinder( mag = [500,0,500], dim = [4,5])

print(s.position)       # Output: [0. 0. 0.]

s.move([1,2,3])
print(s.position)       # Output: [1. 2. 3.]

s.move([1,2,3])
print(s.position)       # Output: [2. 4. 6.]
```

### Calculating the Magnetic Field

Once a source object `s` is defined one can calculate the magnetic field generated by it using the two methods `getB` and `getBsweep`. Here the call `s.getB(pos)` simply returns the value of the field which is generated by the source `s` at the sensor position `pos`.

```python
from magpylib.source.magnet import Cylinder

s = Cylinder( mag = [500,0,500], dim = [4,5])
print(s.getB([4,4,4]))       

# Output: [ 7.69869084 15.407166    6.40155549]
```

In most cases, however, one will be interested to determine the field for a set of sensor positions, or for different magnet positions and orientations. While this can manually be achieved by looping `getB`, magpylib also provides the advanced method `s.getBsweep(INPUT)` for ease of use and for the possibility of multiprocessing. Here *INPUT* can have two possible formats:

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


```eval_rst
.. note::
    With getBsweep's keyword argument `multiprocessing=True`, it is possible to utilize the host computer's multiple cores to calculate points in parallel.
    
    Please check out our :doc:`2_guideExamples` page for more details on multiprocessing.    
```

Please check out our [Guide and Examples](2_guideExamples.md) page for more details.
The following example code shows how to quickly calculate the magnetic field using `getBsweep` with *INPUT TYPE 1*:

```eval_rst

.. plot:: pyplots/doku/getBsweep.py
   :include-source:

:download:`getBsweep.py <../pyplots/doku/getBsweep.py>`
```

<i><p align="center" style="font-weight: 100; font-size: 10pt"> <b>Figure:</b> Output of the above code. Three components of the field in millitesla along a linear stroke above the magnet. </p></i>

To calculate the fields, magpylib uses mostly analytical expressions that can be found in the literature. A detailed analysis of the precision and applicability of these solutions can be found in the [Physics section](9_physics.md). In a nutshell for the magnet classes: The analytical solution fixes the magnetization and can therefore not treat magnetization effects based on interactions. This means that Hysteresis effects, demagnetization and soft magnetic materials cannot be treated. But for typical hard ferromagnets like Ferrite, Neodyme and SmCo the accuracy of the solution easily exceeds 98%.

```eval_rst
.. note::
  It is critical to note that the analytical solution does not treat interaction between the sources. This means that even if multiple sources are defined, `s.getB` will return only the unperturbed field from the source `s`. The total magnetic field is simply given by the superposition of the fields of all sources.
```

## The Collection Class

The idea behind the collection class is to group multiple source objects for common manipulation and evaluation of the fields.

#### Constructing Collections

In principle a collection `c` is simply a list of source objects that are collected in the attribute `c.sources`.

Collections can be constructed at initialization by simply giving the sources objects as arguments. It is possible to add single sources, lists of multiple sources and even other collection objects. All sources are simply added to the `sources` attribute of the target collection.

With the collection kwarg `dupWarning=True`, adding multiples of the same source will be blocked, and a warning will be displayed informing the user that a source object is already in the collection's `source` attribute. This can be unblocked by providing the `dupWarning=False` kwarg.

In addition, the collection class features methods to add and remove sources for command line like manipulation. The method `c.addSources(*sources)` will add all sources given to it to the collection `c`. The method `c.removeSource(ref)` will remove the referenced source from the collection. Here the `ref` argument can be either a source or an integer indicating the reference position in the collection, and it defaults to the latest added source in the Collection.

```Python
import magpylib as magpy

#define some magnet objects
mag1 = magpy.source.magnet.Box(mag=[1,2,3],dim=[1,2,3])
mag2 = magpy.source.magnet.Box(mag=[1,2,3],dim=[1,2,3],pos=[5,5,5])
mag3 = magpy.source.magnet.Box(mag=[1,2,3],dim=[1,2,3],pos=[-5,-5,-5])

#create/manipulate collection and print source positions
c = magpy.Collection(mag1,mag2,mag3)
print([s.position for s in c.sources])
#OUTPUT: [array([0., 0., 0.]), array([5., 5., 5.]), array([-5., -5., -5.])]

c.removeSource(1)
print([s.position for s in c.sources])
#OUTPUT: [array([0., 0., 0.]), array([-5., -5., -5.])]

c.addSources(mag2)
print([s.position for s in c.sources])
#OUTPUT: [array([0., 0., 0.]), array([-5., -5., -5.]), array([5., 5., 5.])]
```

#### Common Manipulation and total Magnetic Field

All methods for geometric operations (`setPosition`, `move`, `setOrientation` and `rotate`) are also methods of the collection class. A geometric operation applied to a collection is directly applied to each object within that collection individually. In practice this means e.g. that a whole group of magnets can be rotated about a common pivot point with a single command.

For calculating the magnetic field that is generated by a whole collection the methods `getB` and `getBsweep` are also available. Just as for geometric operations these methods will be applied to all sources individually, and the total magnetic field will be given by the sum of all parts. For obvious reasons the `getBsweep` method works only for INPUT TYPE 1 with collections.

```eval_rst

|Collection| |total Field|

.. |Collection| image:: ../_static/images/documentation/collectionExample.gif
   :width: 45%

.. |total Field| image:: ../_static/images/documentation/collectionAnalysis.png
   :width: 50%
```

<i><p align="center" style="font-weight: 100; font-size: 10pt"> <b>Figure:</b> Collection Example. Circular current sources are grouped into a collection to form a coil. The whole coil is then geometrically manipulated and the total magnetic field is calculated and shown in the xz-plane </p></i>

#### Complex Magnet Geometries

As a result of the superposition principle complex magnet shapes and inhomogeneous magnetizations can be generated by combining multiple sources. Specifically, when two magnets overlap in this region a *vector union* applies. This means that in the overlap the magnetization vector is given by the sum of the two vectors of each object.

```eval_rst
.. image:: ../_static/images/documentation/superposition.JPG
   :align: center
   :scale: 30 %
```
<i><p align="center" style="font-weight: 100; font-size: 10pt"> <b>Figure:</b> Schematic of the *vector union* principle. </p></i>

Geometric addition and subtration operations can be the result when the magnetization vectors are opposed to each other which allows to "cut out" a small volume from a larger one, example THE HOLLOW CYLINDER.

#### Display Collection Graphically

Finally, the collection class provides the method `displaySystem` to quickly check the geometry of the source assembly. It uses the matplotlib package and its limited capabilities of 3D plotting which often results in bad object overlapping.

`displaySystem()` comes with three keyword arguments:

- `markers=listOfPos` for displaying reference positions. By default a marker is set at the origin. By giving *[a,b,c,'text']* instead of just a simple 3vector *'text'* is displayed with the marker.
- `suppress=True` for suppressing the figure output. To suppress the output it is necessary to deactivate the interactive mode by calling *pyplot.ioff()*.
- `direc=True` for displaying current and magnetization directions in the figure.
- `subplotAx=None` for displaying the plot on a designated figure subplot instance.

The following example code shows how a collection is initialized and displayed.

```eval_rst

.. plot:: pyplots/doku/displaySys.py
   :include-source:

:download:`displaySys.py <../pyplots/doku/displaySys.py>`
```

<i><p align="center" style="font-weight: 100; font-size: 10pt"> <b>Figure:</b> Output of the above code demonstrating the `displaySystem()` method. </p></i>

## Math Package

The math package provides some functions for easier use of the angle-axis (Quaternion) rotation used in magpylib. 

- `anglesFromAxis(axis)`: This function takes an arbitrary `axis` argument (3-vector) and returns its orientation given by the angles (PHI,THETA) that are defined as in spherical coordinates. PHI is the azimuth angle and THETA is the polar angle.
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

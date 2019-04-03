# Getting Started with MagPylib

>Note: This is a Work in Progress
- [Getting Started with MagPylib](#getting-started-with-magpylib)
    - [Summary of the Library Structure](#summary-of-the-library-structure)
    - [Input Types](#input-types)
    - [Defining Sources](#defining-sources)
    - [Calculating Fields](#calculating-fields)
    - [Creating Collections and Visualizing Geometry](#creating-collections-and-visualizing-geometry)
    - [Translations and Rotations](#translations-and-rotations)
    - [Multipoint Field Calculations](#multipoint-field-calculations)

### Summary of the Library Structure 

Magpylib is defined by three main modules:

```eval_rst

The core module of magpylib is :mod:`~magpylib.source`, whose subpackages
offer the primitive building blocks for creating our simulation data.

The top level serves the :class:`magpylib.Collection`
class, which offers easy grouping of :mod:`~magpylib.source` objects,
allowing for the display and combination of fields.

The :mod:`~magpylib.math` module provides on-hand methods for handling axis
and angle information as well as rotation of position vectors.

.. currentmodule:: magpylib

.. autosummary::

   Collection
   source
   math

```

### Input Types

MagPyLib utilizes a few arbitrary input types which are currently unchecked. 
They are as follows:


```eval_rst
+------------+---------------------------------------------------------+-----------------------------------------------------------------------------+------------------------------------------+
|  Type Name | Description                                             | Example                                                                     | Actual Type                              |
+============+=========================================================+=============================================================================+==========================================+
| scalar     | A field element.                                        | scalar = 90.0                                                               |  numeric (instances of `int` or `float`) |
+------------+---------------------------------------------------------+-----------------------------------------------------------------------------+------------------------------------------+
| vec3       |  A Vector of 3 scalars.                                 | vec3 = (1,2.5,-3)                                                           | Tuple(numeric,numeric,numeric)           |
+------------+---------------------------------------------------------+-----------------------------------------------------------------------------+------------------------------------------+
| listVec3   |  A List Containing an N amount of Vectors of 3 scalars. | listOfVec3 = [(1,2,3),              (4.0,5.0,6.0),              (-7,-8,-9)] |  List[Tuple(numeric,numeric,numeric)]    |
+------------+---------------------------------------------------------+-----------------------------------------------------------------------------+------------------------------------------+
```

### Defining Sources

```eval_rst
The :class:`magpylib.source` module contains objects that represent electromagnetic sources. These objects are created in a unique 3D space with cartesian positioning, and generate different fields depending on their geometry and magnetization vectors.

As an example we will define a :mod:`~magpylib.source.magnet.Box` source object from the :mod:`magpylib.source.magnet` module.
```


```python
import magpylib


b = magpylib.source.magnet.Box( mag = [1,2,3],   # The magnetization vector in microTesla.
                                dim = [4,5,6],   # The length, width and height of our box in mm.
                                pos = [7,8,9],   # The center position of this magnet 
                                                 # in a cartesian plane.
                                angle = 90,      # The angle of orientation around the given
                                                 # axis upon.
                                axis = (0,0,1))  # The axis for orientation, 
                                                 # (x,y,z) respectively.)


print(b.magnetization)  # Output: [1. 2. 3.]
print(b.dimension)      # Output: [4. 5. 6.]
print(b.position)       # Output: [7. 8. 9.]
print(b.angle)          # Output: 90.0
print(b.axis)           # Output: [0. 0. 1.]
print(b)                # Output: Memory location of our Box Object

```
### Calculating Fields

The magnetic field at a single point may be calculated by entering a position vector into the source object's getB() method. This method is available in all source classes.

```python
import magpylib         

pointToCalculate = [-3,-2,-1] # Position Vector of the field point to calculate
b = magpylib.source.magnet.Box( mag = [1,2,3],   
                                dim = [4,5,6],  
                                pos = [7,8,9],  
                                angle = 90,     
                                axis = (0,0,1))

fieldPoint = b.getB(pointToCalculate) # Get the B field at given point
print(fieldPoint) # Output: [ 0.00730574  0.00181691 -0.00190384] 
```

This is most effective when paired with matplotlib, allowing you to visualize your simulation data.

### Creating Collections and Visualizing Geometry

```eval_rst
Top Level holds the :class:`magpylib.Collection` class, which represents a collection of source objects. 

This means that you may define a space where multiple source objects interact, and acquire the resulting magnetic fields of multiple sources. 

A :class:`~magpylib.Collection` object also allows you to manipulate the listed source objects and show them in a 3D display.

Let's create a collection and visualize our :mod:`~magpylib.source.magnet.Box`.

.. plot:: pyplots/guide/collection1.py
   :include-source:

We can set markers to help us identify certain points in the 3D plane. By default, there is a marker at position `[0,0,0]`.

To set a marker or more, we define a list of positions and utilize the marker keyword argument in the displaySystem method.

Collections also allow us to retrieve the getB field of all magnets in the position with its getB() method. 

Let's mark the position and retrieve the B field from our :mod:`~magpylib.source.magnet.Box` object.

.. plot:: pyplots/guide/collection2.py
   :include-source:


```

### Translations and Rotations

All Objects, be it a Source Object or a Collection Object, have a set of methods that allow for Translations and Rotation 

To be Completed

### Multipoint Field Calculations

To be Defined
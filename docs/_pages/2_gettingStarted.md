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
      - [Displacement Input](#displacement-input)

### Summary of the Library Structure 

```eval_rst

.. image:: ../../_static/images/summary/lib.png
   :align: center
   :scale: 50 %

Magpylib is defined by three main modules:

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
Here's a short table with further details:


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

Summary:

- All input is either `scalar`, `vector`, or a list-like object of either.
- `scalar` can be of any numeric data type (`int` , `float` , `np.float64`,...)
- `vector` can be any iterable, list-like object (`list`, `tuple`, `np.array`,...) of arbitrary data type.
  
The formulas are set up such that the input and output variables are given in the units of: 
- *Millimeter* [`mm`]
- *Millitesla* [`mT`]
- *Ampere* [`A`]
-  *Degree* [`deg`]


### Defining Sources

```eval_rst
The :class:`magpylib.source` module contains objects that represent electromagnetic sources. 

These objects are created in an isolated, unique 3D frame with cartesian positioning, and generate different fields depending on their geometry and magnetization vectors.

As an example we will define a :mod:`~magpylib.source.magnet.Box` source object from the :mod:`magpylib.source.magnet` module.
```


```python
import magpylib


b = magpylib.source.magnet.Box( mag = [1,2,3],  # The magnetization vector in microTesla.
                                dim = [4,5,6],  # Length, width and height of our box in mm.
                                pos = [7,8,9],  # The center position of this magnet 
                                                # in a cartesian plane.
                                angle = 90,     # The angle of orientation 
                                                # around the given axis.
                                axis = (0,0,1)) # The axis for orientation, 
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

We can set markers with labels to help us identify certain points in the 3D plane. By default, there is a marker at position `[0,0,0]`.

```

---


```eval_rst

Collections also allow us to add Source Objects into a shared frame so they can interact, and to retrieve samples of the resulting B field sample of this interaction with its :func:`~magpylib.Collection.getB` method. 

Let's **retrieve the B field** sample from our :mod:`~magpylib.source.magnet.Box` **interacting** with a :mod:`~magpylib.source.magnet.Sphere` object, and show it in the display.

.. plot:: pyplots/guide/collection2.py
   :include-source:


```

### Translations and Rotations

All Objects, be it a Source Object or a Collection Object, have a set of methods that allow for Translations and Rotation.

```python
from magpylib import source

neutral = [0,0,0]

b = source.magnet.Box(mag=[1,2,3],
                      dim=[2,2,2],
                      pos=neutral)

b.setPosition([2,0,0]) ## Place object in [2,0,0]
print(b.position)      ## [2,0,0]

b.move([5,0,0])        ## Move 5 units in X
print(b.position)      ## [7,0,0]

b.setPosition(neutral) ## Place object in [0,0,0]
print(b.position)      ## [0,0,0]

b.move([5,0,0])        ## Move 5 units in X
print(b.position)      ## [5,0,0]

```

```eval_rst

.. note::
   Source Objects within a Collection will have their coordinates modified within the Collection frame, Collections do not create copies.
   If you'd like to avoid this, create `a deep copy <https://docs.python.org/3/library/copy.html>`_. of the source object and add the copy to the Collection instead.


Source Objects may be rotated in respect to themselves or an anchored pivot point.

The result of :func:`~magpylib.source.magnet.Cylinder.rotate` is affected relative to the current position of the Source Object.

If you'd like to set a position that is absolute to the Source's frame, use :func:`~magpylib.source.magnet.Cylinder.setPosition` instead as the manipulation is always the same. This is not available for Collection.

.. plot:: pyplots/guide/rotate1.py
   :include-source:

```

Rotations may also be done with an anchored pivot point. The following code adds two Objects to a Collection, and only moves one of them.

```eval_rst

.. plot:: pyplots/guide/rotate2.py
   :include-source:

```

---

Collections may be rotated using the previous logic as well. Keep in mind if an anchor is not provided, all objects will rotate relative to their own center.


```eval_rst

.. plot:: pyplots/guide/rotate3.py
   :include-source:

```

---

Ultimately, Collections can be added to other Collections, and rotated independently.

```eval_rst

.. plot:: pyplots/guide/rotate4.py
   :include-source:

```
### Multipoint Field Calculations

One of the greatest strengths of the analytical approach is that all desired points of a field computation may be done in parallel, reducing computation overhead.

```eval_rst
.. warning::

    Due to how multiprocessing works on **Windows Systems, the following structure for your code is mandatory**:

    .. code::
    
       from multiprocessing import freeze_support

       def your_code():
           ## Your code

       if __name__ == "__main__":
            freeze_support()
            your_code()

    Failure to comply to this will cause your code **to not yield any results.**
```
Here is an example calculating several marked points in sequence.

```eval_rst
.. plot::  pyplots/guide/multiprocessing1.py
   :include-source:

```

#### Displacement Input

The parallel function may also be utilized to calculate samples of several setups in parallel.

Field sample position, Source object orientation and positioning may be adjusted in every setup, like the following structure:
```eval_rst

.. image:: ../../_static/images/user_guide/multiprocessing.gif
   :align: center

.. image:: ../../_static/images/user_guide/sweep.png
   :align: center
   :scale: 50 % 

```

```python
from magpylib import source, Collection

def setup_creator(sensorPos,magnetPos,angle):
    axis = (0,0,1) # Z axis
    setup = [sensorPos,
            magnetPos,
            (angle,axis)]
    return setup

## Define information for 8 setups
sensors = [ [-1,-6,6], [-1,-5,5], 
            [-1,-4,4.5],[-1,-6,3.5], 
            [-1,-5,2.5], [-1,-4,1.5],
            [-1,-5,-0.5], [-1,-4,-1.0] ]

angles = [0,30,60,90,120,180,210,270]

positions = [ [3,-4,6],[3,-4,5],
              [3,-4,4],[3,-4,3,],
              [3,-4,2],[3,-4,1],
              [3,-4,0],[3,-4,-1] ]

b = source.magnet.Box([1,2,3],
                      [1,1,1])

setups = [setup_creator(sensors[i],
                        positions[i],
                        angles[i]) for i in range(0,7)]

results = b.getBsweep(setups)

print(results)
## Result for each of the 8 setups:
# [array([ 0.0033804 ,  0.00035464, -0.00266834]), 
#  array([ 0.00151226, -0.00219274, -0.00340392]), 
#  array([-0.00427052, -0.00226601, -0.00292288]), 
#  array([-0.00213505, -0.00281333, -0.00213425]), 
#  array([-0.00567799, -0.00189228, -0.00231176]), 
#  array([-0.00371514,  0.00242773, -0.00302629]), 
#  array([-0.00030278,  0.00243991, -0.00334978])]
```









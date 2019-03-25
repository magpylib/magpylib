# Getting Started with MagPylib

>Note: This is a Work in Progress

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



### Defining Sources

```eval_rst
The :class:`magpylib.source` module contains objects that represent electromagnetic sources. These objects are created in a unique 3D space with cartesian positioning, and generate different fields depending on their geometry and magnetization vectors.

As an example we will define a :mod:`~magpylib.source.magnet.Box` source object from the :mod:`magpylib.source.magnet` module.
```


```python
import magpylib
mag = [1,2,3]   # The magnetization vector in microTesla.
dim = [4,5,6]   # The length, width and height of our box in mm.
pos = [7,8,9]   # The position of this magnet 
                # in a cartesian plane.
angle = 90      # The angle of orientation around the given
                # axis upon.
axis = (0,0,1)  # The axis for orientation, 
                # (x,y,z) respectively.

b = magpylib.source.magnet.Box(mag,dim,pos,angle,axis)

print(b.magnetization)  # The magnetization vector of this object
print(b.dimension)      # The dimensions of this box
print(b.position)       # The center position of the object in the cartesian plane
print(b.angle)          # The angle of orientation of the object
print(b.axis)           # The axis of orientation of the object
print(b)                # Location in memory of our object

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
print(fieldPoint) # Result 
```

This is most effective when paired with matplotlib, allowing you to visualize your simulation data.

### Creating Collections and Visualizing Geometry

```eval_rst
Top Level holds the :class:`magpylib.Collection` class, which represents a collection of source objects. 

This means that you may define a space where multiple source objects interact, and acquire the resulting magnetic fields of multiple sources. 

A :class:`~magpylib.Collection` object also allows you to manipulate the listed source objects and show them in a 3D display.

Let's create a collection and visualize our :mod:`~magpylib.source.magnet.Box`.
```

```python
import magpylib         
b = magpylib.source.magnet.Box( mag = [1,2,3],   
                                dim = [4,5,6],  
                                pos = [7,8,9],  
                                angle = 90,     
                                axis = (0,0,1))

col = magpylib.Collection(b)
col.displaySystem()
```

```eval_rst
We can set markers to help us identify certain points in the 3D plane. By default, there is a marker at position `[0,0,0]`.

To set a marker or more, we define a list of positions and utilize the marker keyword argument in the displaySystem method.

Collections also allow us to retrieve the getB field of all magnets in the position with its getB() method. 

Let's mark the position and retrieve the B field from our :mod:`~magpylib.source.magnet.Box` object.
```

```python
import magpylib 
pointToCalculate = [-3,-2,-1] # Position Vector of the field point to calculate        

b = magpylib.source.magnet.Box( mag = [1,2,3],   
                                dim = [4,5,6],  
                                pos = [7,8,9],  
                                angle = 90,     
                                axis = (0,0,1))

col = magpylib.Collection(b)
print(col.getB(pointToCalculate))

markerPosition = pointToCalculate
col.displaySystem(markers=[ markerPosition ])

```

To be Completed

### Multipoint Field Calculations

To be Defined
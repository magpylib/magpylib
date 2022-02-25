---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.6
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Basic functionality

+++

## Just compute the field

The most fundamental functionality of the library - compute the field (B in \[mT\], H in \[kA/m\]) of a
source (here Cuboid magnet) at the observer position (1,2,3)mm.

```{code-cell} ipython3
from magpylib.magnet import Cuboid

# create a cuboid magnet object instance, position is at x,y,z=(0,0,0) by default
cuboid = Cuboid(magnetization=(222, 333, 444), dimension=(2, 2, 2))

# reposition the cuboid
cuboid.move((10, 10, 10))
print("Cuboid position: ", cuboid.position)

# compoute the magnetic field at position x,y,z=(1,2,3)
B = cuboid.getB((1, 2, 3))
print("B-Field from cuboid at observer position (1,2,3): ", B)
```

+++ {"tags": []}

## Group objects into a `Collection`

+++

Magpylib objects can be grouped into `Collection` objects to allow easy manipulation and field computation. The Collection acts in many ways as a single object and can be moved or rotated, while children keep their relative positions and orientations.

```{code-cell} ipython3
from magpylib import Collection
from magpylib.magnet import Cuboid, Cylinder, Sphere

cuboid = Cuboid(magnetization=(1000, 0, 0), dimension=(2, 2, 2), position=(-5,0,0))
cylinder = Cylinder(magnetization=(1000, 0, 0), dimension=(2, 2), position=(0,0,0))
sphere = Sphere(magnetization=(1000, 0, 0), diameter=2, position=(5,0,0))

# group sources into a Collection 
collection = Collection(cuboid, cylinder, sphere)

# move the collection as a whole
print(
    'Object positions before moving\n',
    '- cuboid: ', cuboid.position,
    '- cylinder: ', cylinder.position,
    '- sphere: ', sphere.position
)
collection.move((10,10,10))

print(
    'Object positions after moving\n',
    '- cuboid: ', cuboid.position,
    '- cylinder: ', cylinder.position,
    '- sphere: ', sphere.position
)

# Compute the total field at observer position x,y,z = (100,0,0)
Bsum = collection.getB([100,0,0])
print('Total B-field at observer position (100,0,0)\n', Bsum)
```

## Field values of a path

The position and orientation properties of any Magpylib object can also hold a series of values. This allows to construct paths for moving and/or rotating objects and to efficiently evaluate the magnetic field for intermediate steps.
In this example the field of a spherical magnet is evaluated for a moving observer rotating 360° with 45° steps around the source along the z-axis and a radius of 5\[mm\].

```{code-cell} ipython3
import numpy as np
import plotly.graph_objects as go
from magpylib import Sensor
from magpylib.magnet import Sphere

sphere = Sphere(magnetization=(1000, 0, 0), diameter=1)
sensor = Sensor(position=(5, 0, 0))
angle_array = np.linspace(0.0, 360, 12, endpoint=False)
sensor.rotate_from_angax(angle_array, "z", anchor=sphere.position, start=0)
B = sensor.getB(sphere)
print("B-field value over a path:\n", B)

fig = go.Figure()
fig.update_layout(
    title="Magnetic field from a sensor rotating around a cylinder magnet",
    xaxis_title="Angle [deg]",
    yaxis_title="Magnitude [mT]",
)
for i, k in enumerate("xyz"):
    fig.add_scatter(x=angle_array, y=B.T[i], name=f"B{k}")
fig.show()
```



```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy

# define pyplot figure
fig, [ax1,ax2] = plt.subplots(1, 2, figsize=(10,5))

# define Magpylib source
src = magpy.magnet.Cuboid(magnetization=(500,0,500), dimension=(2,2,2))

# create a grid in the xz-symmetry plane
ts = np.linspace(-3, 3, 30)
grid = np.array([[(x,0,z) for x in ts] for z in ts])

# compute B field on grid using a source method
B = src.getB(grid)
ampB = np.linalg.norm(B, axis=2)

# compute H-field on grid using the top-level function
H = magpy.getH(src, grid)
ampH = np.linalg.norm(H, axis=2)

# display field with Pyplot
ax1.streamplot(grid[:,:,0], grid[:,:,2], B[:,:,0], B[:,:,2],
    density=2, color=np.log(ampB), linewidth=1, cmap='autumn')

ax2.streamplot(grid[:,:,0], grid[:,:,2], H[:,:,0], H[:,:,2],
    density=2, color=np.log(ampH), linewidth=1, cmap='winter')

# outline magnet boundary
for ax in [ax1,ax2]:
    ax.plot([1,1,-1,-1,1], [1,-1,-1,1,1], 'k--')

plt.tight_layout()
plt.show()
```
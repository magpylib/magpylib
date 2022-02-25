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

(docu)=

# Introduction to Magpylib

This section provides an introduction to the Magpylib API. Detailed behavior is documented in the docstrings. Many practical examples how to use Magpylib can be found in the example galleries.

## Contents

- {ref}`docu-idea`
- {ref}`docu-when-to-use`
- {ref}`docu-magpylib-objects`
- {ref}`docu-position-and-orientation`
- {ref}`docu-grouping-objects`
- {ref}`docu-graphic-output`
- {ref}`docu-field-computation`
- {ref}`docu-getB_dict-getH_dict`
- {ref}`docu-direct-access`


(docu-idea)=

## The idea behind Magpylib

Magpylib provides fully tested and properly implemented **analytical solutions** to permanent magnet and current problems. It gives quick access to extremely fast and accurate magnetic field computation. Details on how these solutions are mathematically obtained can be found in the {ref}`physComp` section.

The central API of Magpylib is object oriented and couples the field computation to a position/orientation interface. The idea is that users can create Magpylib objects like sensors, magnets, currents, etc. with defined position and orientation in a global coordinate system. These objects can then be easily manipulated, displayed, grouped and used for user-friendly field computation. For users who would like to avoid the object oriented interface, the field implementations can also be accessed directly.

(docu-when-to-use)=

## When can you use Magpylib ?

The analytical solutions are exact when there is no material response and natural boundary conditions can be assumed. For permanent **magnets**, when (remanent) permeabilities are below $$\mu_r < 1.1$$ the error is typically below 1-5 % (long magnet shapes are better, large distance from magnet is better). Demagnetization factors are not automatically included at this point. Error estimation as a result of the material response is evaluated in more detail in the appendix of [Malagò2020](https://www.mdpi.com/1424-8220/20/23/6873). The line **current** solutions give the exact same field as outside of a wire that carries a homogenous current. In general, Magpylib is at its best when dealing with air-coils (no eddy currents) and high grade permanent magnet assemblies (Ferrite, NdFeB, SmCo or similar materials). For more details check out the {ref}`physComp` section.

Magpylib only provides solutions for simple geometric forms. However, in magnetostatics the **superposition principle** holds, i.e. the total magnetic field is given by the sum of all the fields of all sources. For magnets this means that complex magnet shapes can be constructed from simple forms. **EXAMPLE_CONSTRUCTING_A_COMPLEX_FORM** Specifically, it is possible to cut-out a part of a magnet simply by placing a second magnet with opposite magnetization inside the first one. **EXAMPLE_HOLLOW_CYLINDER**

(docu-magpylib-objects)=

## The Magpylib objects

The most convenient way for working with Magpylib is through the object oriented interface. Magpylib objects represent magnetic field sources, sensors and collections with various defining attributes and methods. With Version 4.0.0 the following classes are implemented:

**Magnets**

All magnet objects have the `magnetization` attribute which must be of the format $(m_x, m_y, m_z)$ and denotes the homogeneous magnetization vector in units of millitesla. This is often referred to as the remanence ($B_r=\mu_0 M$) in material data sheets. All magnets can be used as Magpylib `sources` input.

- **`Cuboid`**`(magnetization, dimension, position, orientation, style)` represents a magnet with cuboid shape. `dimension` has the format $(a,b,c)$ and denotes the sides of the cuboid in units of millimeters. By default (`position=(0,0,0)`, `orientation=None`) the center of the cuboid lies in the origin of the global coordinates, and the sides are parallel to the coordinate axes.

- **`Cylinder`**`(magnetization, dimension, position, orientation, style)` represents a magnet with cylindrical shape. `dimension` has the format $(d,h)$ and denotes diameter and height of the cylinder in units of millimeters. By default (`position=(0,0,0)`, `orientation=None`) the center of the cylinder lies in the origin of the global coordinates, and the cylinder axis coincides with the z-axis.

- **`CylinderSegment`**`(magnetization, dimension, position, orientation, style)` represents a magnet with the shape of a cylindrical ring section. `dimension` has the format $(r_1,r_2,h,\varphi_1,\varphi_2)$ and denotes inner radius, outer radius and height in units of milimeters and the two section angles $\varphi_1<\varphi_2$ in degrees. By default (`position=(0,0,0)`, `orientation=None`) the center of the full cylinder lies in the origin of the global coordinates, and the cylinder axis coincides with the z-axis.

- **`Sphere`**`(magnetization, diameter, position, orientation, style)` represents a magnet of spherical shape. `diameter` is the sphere diameter $d$ in units of millimeters. By default (`position=(0,0,0)`, `orientation=None`) the center of the sphere lies in the origin of the global coordinates.

**Currents**

All current objects have the `current` attribute which must be a scalar $i_0$ and denotes the electrical current in units of Ampere. All currents can be used as Magpylib `sources` input.

- **`Loop`**`(current, diameter, position, orientation, style)` represents a circular current loop where `diameter` is the loop diameter $d$ in units of millimeters. By default (`position=(0,0,0)`, `orientation=None`) the loop lies in the xy-plane with its center in the origin of the global coordinates.

- **`Line`**`(current, vertices, position, orientation, style)` represents electrical current segments that flow in a straight line from vertex to vertex. By default (`position=(0,0,0)`, `orientation=None`) the locally defined vertices have the same position in the global coordinates.

**Other**

- **`Dipole`**`(moment, position, orientation, style)` represents a magnetic dipole moment with moment $(m_x,m_y,m_z)$ given in $mT\times mm^3$. For homogeneous magnets the relation moment=magnetization$\times$volume holds. Can be used as Magpylib `sources` input.

- **`CustomSource`**`(field_B_lambda, field_H_lambda, position, orientation, style)` can be used to create user defined custom sources. Can be used as Magpylib `sources` input.

- **`Sensor`**`(position, pixel, orientation)` represents a 3D magnetic field sensor. The field is evaluated at the given pixel positions, by default `pixel=(0,0,0)`. Can be used as Magpylib `observers` input.

- **`Collection`**`(*children, position, orientation)` is a group of source and sensor objects (children) that is used for common manipulation. Depending on the children, a collection can be used as Magpylib `sources` and/or `observers` input.

```{code-cell} ipython3
import magpylib as magpy

# magnets
src1 = magpy.magnet.Cuboid()
src2 = magpy.magnet.Cylinder()
src3 = magpy.magnet.CylinderSegment()
src4 = magpy.magnet.Sphere()

# currents
src5 = magpy.current.Loop()
src6 = magpy.current.Line()

# other
src7 = magpy.misc.Dipole()
src8 = magpy.misc.CustomSource()
sens = magpy.Sensor()
col = magpy.Collection()

# print object representation
for obj in [src1, src2, src3, src4, src5, src6, src7, src8, sens, col]:
    print(obj)
```

(docu-position-and-orientation)=

## Position and orientation

All Magpylib objects have the `position` and `orientation` attributes that refer to position and orientation in the global Cartesian coordinate system. The `position` attribute is a numpy `ndarray, shape (3,) or (m,3)` and denotes `(x,y,z)` coordinates in units of millimeter. By default every object is created at `position=(0,0,0)`. The `orientation` attribute is a scipy [Rotation object](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html). By default the orientation of a Magpylib object is the unit rotation, generated by `orientation=None`.

```{note}
The attributes `position` and `orientation` can be either of **"scalar"** nature, i.e. a single position or a single rotation, or **"vectors"** when they are arrays of such scalars. The two attributes together define an object **"path"**.
```

```{code-cell} ipython3
import magpylib as magpy
from scipy.spatial.transform import Rotation as R

sens = magpy.Sensor()

print(sens.position)
print(sens.orientation.as_euler('xyz', degrees=True))
```

Set and manipulate position and orientation at object initialization,

```{code-cell} ipython3
sens = magpy.Sensor(position=[(1,1,1), (2,2,2), (3,3,3)])

print(sens.position)
print(sens.orientation.as_euler('xyz', degrees=True))
```

or after initialization using the setter methods,

```{code-cell} ipython3
sens = magpy.Sensor()
sens.orientation=R.from_rotvec([(0,0,45), (0,0,90)], degrees=True)

print(sens.position)
print(sens.orientation.as_euler('xyz', degrees=True))
```

or by making use of the `move` and `rotate` methods:

```{code-cell} ipython3
sens = magpy.Sensor()
sens.move((1,2,3))
sens.rotate_from_angax(45, 'z')

print(sens.position)
print(sens.orientation.as_euler('xyz', degrees=True))
```

Notice that, when one of the `position` and `orientation` attributes is manipulated, the other is automatically adjusted to the same size.

```{note}
The underlying Magpylib logic is to always pad the edge entries of a path when the length is increased, and to slice from the end of the path when the length is reduced by an operation.
```

The **`move()`** and **`rotate()`** methods provide a lot of additional functionality that makes it easy to move objects around and to generate more complex paths. They obey the following rules:

```{note}
- Scalar input is applied to the whole object path, starting with path index `start`. With the default `start='auto'` the index is set to `start=0` and the functionality is *moving objects around*.
- Vector input of length $n$ applies the individual $n$ operations to $n$ object path entries, starting with path index `start`. Padding applies when the input exceeds the existing path. With the default `start='auto'` the index is set to `start=len(object path)` and the functionality is *appending paths*.
```

The following example demonstrates this functionality (works similarly for rotations):

```{code-cell} ipython3
import magpylib as magpy

sens = magpy.Sensor(position=((0,0,0), (1,1,1)))

# scalar input is by default applied to the whole path
sens.move((1,1,1))
print(f"{sens.position}\n")

# vector input is by default appended
sens.move([(1,1,1)])
print(f"{sens.position}\n")

# scalar input and start=1 is applied to whole path starting at index 1
sens.move((1,1,1), start=1)
print(f"{sens.position}\n")

# vector input and start=1 merges the input with the existing path starting at index 1
sens.move([(0,0,10)]*3, start=1)
print(sens.position)
```

A more detailed example how complex paths are generated through merging, slicing and padding is given in **EXAMPLE_PATH_MERGE_SLICE_PAD**.

(docu-graphic-output)=

## Graphic output

When all Magpylib objects and their paths have been created, `show()` provides a convenient way to graphically display the geometric arrangement using the Matplotlib and Plotly graphic libraries. By installation default, objects are displayed with the Matplotlib backend. Matplotlib comes with Magpylib installation, Plotly must be installed by hand.

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

src1 = magpy.magnet.Cuboid(magnetization=(-100,0,0), dimension=(1,1,1), position=(0,0,-7))
src2 = magpy.magnet.Cylinder(magnetization=(0,0,100), dimension=(1,1), position=(-5,0,0))
src3 = magpy.magnet.CylinderSegment(magnetization=(0,0,100), dimension=(.3,1,1,0,140), position=(-3,0,0))
src4 = magpy.magnet.Sphere(magnetization=(0,0,100), diameter=1, position=(-1,0,0))
src5 = magpy.current.Loop(current=1, diameter=1, position=(1,0,0))
src6 = magpy.current.Line(current=1, vertices=[(1,0,0), (0,1,0), (-1,0,0), (0,-1,0), (1,0,0)], position=(3,0,7))
src7 = magpy.misc.Dipole(moment=(0,0,100), position=(5,0,0))
sens = magpy.Sensor(pixel=[(0,0,z) for z in (-.5,0,.5)], position=(7,0,0))

src6.move(np.linspace((0,0,0), (0,0,-7), 20))
src1.rotate_from_angax(np.linspace(0, 90, 20), 'y', anchor=0)

magpy.show(src1, src2, src3, src4, src5, src6, src7, sens)
```

Objects can be styled individually by:
1. Providing a style dictionary at object initialization.
2. Making use of style underscore_magic at initialization.
3. Manipulating the style properties.

```{code-cell} ipython3
import magpylib as magpy

src1 = magpy.magnet.Sphere((0,0,1), 1, (0,0,0), style={'color':'r'})
src2 = magpy.magnet.Sphere((0,0,1), 1, (1,0,0), style_color='g')
src3 = magpy.magnet.Sphere((0,0,1), 1, (2,0,0))
src3.style.color='b'

magpy.show(src1, src2, src3)
```

There are multiple hierarchy levels that descide about the final object style that is displayed:
1. When no input is given, the default style from `magpy.defaults.display.style` will be applied.
2. Individual object styles will take precedence over the default values.
3. Setting a global style in `show()` will overwrite all other inputs.

```{code-cell} ipython3
import magpylib as magpy

# default
magpy.defaults.display.style.base.color='red'
magpy.defaults.display.style.base.path.line.style=':'

# individual styles
src1 = magpy.magnet.Sphere((0,0,1), 1, [(0,0,0), (0,0,2)])
src2 = magpy.magnet.Sphere((0,0,1), 1, [(1,0,0), (1,0,2)], style_path_line_style='-')
src3 = magpy.magnet.Sphere((0,0,1), 1, [(2,0,0), (2,0,2)], style_color='g')

# overwrite globally
magpy.show(src1, src2, src3, style_path_line_style='--')
```

Detailed information on the show features including how to animate paths and how to set up the more powerful plotly graphic backend can be found in example gallery [**XXXX**](examples/02_display_features).


(docu-grouping-objects)=

## Collections

The top level class `magpylib.Collection` allows a user to group sources for common manipulation. All operations acting on a collection are individually applied to all child objects.

```python
import magpylib as magpy

src1 = magpy.magnet.Sphere((1,2,3), 1, position=(2,0,0))
src2 = magpy.current.Loop(1, 1, position=(-2,0,0))
col = magpy.Collection(src1, src2)
col.move(((0,0,2)))
print(src1.position)   # out: [2. 0. 2.]
print(src2.position)   # out: [-2.  0.  2.]
print(col.position)    # out: [0. 0. 2.]
```

Collections function as **groups**: Objects can be part of multiple collections, and after being added to a collection, it is still possible to manipulate the individual objects by name, and also by collection index:

```python
src1.move((2,0,0))
col[1].move((-2,0,0))
print(src1.position)   # out: [4. 0. 2.]
print(src2.position)   # out: [-4.  0.  2.]
print(col.position)    # out: [0. 0. 2.]
```

But collections also function as **compound objects**: For magnetic field computation a collection that contains sources functions like a single source (that returns the total field of all children). Collections have their own `position` and `orientation` attributes. Geometric operations (`move`and `rotate` methods, `position` and `orientation` setter) acting on a collection object are individually applied to all child objects - but in such a way that the geometric compound sturcture is maintained. For example, `rotate()` with `anchor=None` rotates all children about `collection.position` (instead of each individual child position) which is demonstrated below:

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

src1 = magpy.magnet.Cuboid((0,0,1), (1,1,1), (4,0,0))
src2 = src1.copy(position=(-4,0,0))

col = src1 + src2
col.rotate_from_angax(np.linspace(0, 120, 50), 'y')
col.show(style_path_frames=15, style_magnetization_show=False)
```

Notice that, when Magpylib objects are added up they will form a collection. We can make use of the `path.frames` property to show our objects at several path positions, and when a collection is displayed, it will apply the same color to all its children.

(docu-field-computation)=

## Magnetic field computation

Magnetic field computation in Magpylib is achieved through:

- **`getB`**`(sources, observers)` computes the B-field seen by `observers` generated by `sources` in units of \[mT\]
- **`getH`**`(sources, observers)` computes the H-field seen by `observers` generated by `sources` in units of \[kA/m\]

The argument `sources` can be any Magpylib source object or a list thereof. The argument `observers` can be an array_like of position vectors with shape $(n_1,n_2,n_3,...,3)$, Magpylib observer objects or a list thereof. `getB()` and `getH()` are top-level functions, but can also be called as Magpylib object methods.

A most fundamental field computation example is shown below where the B-field generated by a cylinder magnet is computed at position $(1,2,3)$:

```{code-cell} ipython3
import magpylib as magpy

src = magpy.magnet.Cylinder(magnetization=(222,333,444), dimension=(2,2))
B = src.getB((1,2,3))
print(B)
```

When computing the field of multiple sources, or multiple observers or long paths, `getB` and `getH` will automatically return the field for all inputs as one big array. In the following example B- and H-field of a cuboid magnet are computed on a grid and displayed using Matplotlib:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy

# define pyplot figure
fig, [ax1,ax2] = plt.subplots(1, 2, figsize=(10,5))

# create an observer grid in the xz-symmetry plane
ts = np.linspace(-3, 3, 30)
grid = np.array([[(x,0,z) for x in ts] for z in ts])

# compute B- and H-fields of a cuboid magnet on the grid
src = magpy.magnet.Cuboid(magnetization=(500,0,500), dimension=(2,2,2))
B = src.getB(grid)
H = src.getH(grid)

# display field with Pyplot
ax1.streamplot(grid[:,:,0], grid[:,:,2], B[:,:,0], B[:,:,2], density=2,
    color=np.log(np.linalg.norm(B, axis=2)), linewidth=1, cmap='autumn')

ax2.streamplot(grid[:,:,0], grid[:,:,2], H[:,:,0], H[:,:,2], density=2,
    color=np.log(np.linalg.norm(B, axis=2)), linewidth=1, cmap='winter')

# outline magnet boundary
for ax in [ax1,ax2]:
    ax.plot([1,1,-1,-1,1], [1,-1,-1,1,1], 'k--')

plt.tight_layout()
plt.show()
```

Magpylib automatically vectorizes the computation so that maximal performance is achieved.

```{note}
The output of the most general field computation `getB(sources, observers)` is an ndarray of shape `(l,m,k,n1,n2,n3,...,3)` where `l` is the number of input sources, `m` the maximal object path length, `k` the number of sensors, `n1,n2,n3,...` the sensor pixel shape or the shape of a position vector input and `3` the three magnetic field components $(B_x,B_y,B_z)$.
```

When input objects have different path lengths, all shorter paths are padded (=static objects) beyond their end. In the following example we compute the field of three sources, two observers with pixel shape (4,5) and one object with path length 11:

```{code-cell} ipython3
import magpylib as magpy

# 3 sources
s1 = magpy.misc.Dipole(moment=(0,0,100))
s2 = magpy.current.Loop(current=1, diameter=3)
col = s1 + s2

# 2 observers with 4x5 pixel
pix = [[(1,2,3)]*4]*5
sens1 = magpy.Sensor(pixel=pix)
sens2 = magpy.Sensor(pixel=pix)

# length 11 path
s1.move([(1,1,1)]*11, start=0)

B = magpy.getB([s1,s2,col], [sens1, sens2])
print(B.shape)
```


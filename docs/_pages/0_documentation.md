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

This section provides an introduction to the Magpylib API. Details are documented in the docstrings.

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

The analytical solutions are exact when there is no material response and natural boundary conditions can be assumed.

For permanent **magnets**, when (remanent) permeabilities are below $\mu_r < 1.1$ the error is typically below 1-5 % (long magnet shapes are better, large distance from magnet is better). Demagnetization factors are not automatically included at this point. Error estimation as a result of the material response is evaluated in more detail in the appendix of [Malagò2020](https://www.mdpi.com/1424-8220/20/23/6873). The line **current** solutions give the exact same field as outside of a wire that carries a homogenous current. In general, Magpylib is at its best when dealing with air-coils (no eddy currents) and high grade permanent magnet assemblies (Ferrite, NdFeB, SmCo or similar materials). For more details check out the {ref}`physComp` section.

Magpylib only provides solutions for simple geometric forms. However, in magnetostatics the **superposition principle** holds, i.e. the total magnetic field is given by the sum of all the fields of all sources. For magnets this means that complex magnet shapes can be constructed from simple forms. Specifically, it is possible to cut-out a part of a magnet simply by placing a second magnet with opposite magnetization inside the first one. REFERENCE TO EXAMPLES

(docu-magpylib-objects)=

## The Magpylib objects

The most convenient way for working with Magpylib is through the object oriented interface. Magpylib objects represent magnetic field sources, sensors and collections with various defining attributes and methods. With Version 4.0.0 the following classes are implemented:

**Magnets**

All magnet objects have the `magnetization` attribute which must be of the format $(m_x, m_y, m_z)$ and denotes the homogeneous magnetization vector in units of millitesla. This is often referred to as the material remanence (=$\mu_0 M$). All magnets can be used as Magpylib `sources` input.

- `Cuboid(magnetization, dimension, position, orientation, style)` represents a cuboid magnet where `dimension` has the format $(a,b,c)$ and denotes the sides of the cuboid in units of millimeters. By default (`position=(0,0,0)`, `orientation=None`) the center of the cuboid lies in the origin of the global coordinates, and the sides are parallel to the coordinate axes.

- `Cylinder(magnetization, dimension, position, orientation, style)` represents a cylindrical magnet where `dimension` has the format $(d,h)$ and denotes diameter and height of the cylinder in units of millimeters. By default (`position=(0,0,0)`, `orientation=None`) the center of the cylinder lies in the origin of the global coordinates, and the cylinder axis coincides with the z-axis.

- `CylinderSegment(magnetization, dimension, position, orientation, style)` represents a magnet with the shape of a cylindrical ring section. `dimension` has the format $(r1,r2,h,\varphi_1,\varphi_2)$ and denotes inner radius, outer radius and height in units of milimeters and the two section angles $\varphi_1<\varphi_2$ in degrees. By default (`position=(0,0,0)`, `orientation=None`) the center of the full cylinder lies in the origin of the global coordinates, and the cylinder axis coincides with the z-axis.

- `Sphere(magnetization, diameter, position, orientation, style)` represents a spherical magnet. `diameter` is the sphere diameter $d$ in units of millimeters. By default (`position=(0,0,0)`, `orientation=None`) the center of the sphere lies in the origin of the global coordinates.

**Currents**

All current objects have the `current` attribute which must be a scalar $i_0$ and denotes the electrical current in units of Ampere. All currents can be used as Magpylib `sources` input.

- `Loop(current, diameter, position, orientation, style)` represents a circular current loop where `diameter` is the loop diameter $d$ in units of millimeters. By default (`position=(0,0,0)`, `orientation=None`) the loop lies in the xy-plane with its center in the origin of the global coordinates.

- `Line(current, vertices, position, orientation, style)` represents electrical current segments that flow in a straight line from vertex to vertex. By default (`position=(0,0,0)`, `orientation=None`) the locally defined vertices have the same position in the global coordinates.

**Other**

- `Dipole(moment, position, orientation, style)` represents a magnetic dipole moment with moment $(m_x,m_y,m_z)$ given in $mT\times mm^3$. For homogeneous magnets the relation moment=magnetization$\times$volume holds. Can be used as Magpylib `sources` input.

- `CustomSource(field_B_lambda, field_H_lambda, position, orientation, style)` can be used to create user defined custom sources. Can be used as Magpylib `sources` input.

- `Sensor(position, pixel, orientation)` represents a magnetic field sensor. The field is evaluated at the given pixel positions, by default `pixel=(0,0,0)`. Can be used as Magpylib `observers` input.

- `Collection(*children, position, orientation)` represents a group of source and sensor objects (children) for common manipulation. Depending on the children, a collection can be used as Magpylib `sources` and `observers` input.

```python
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

# out: Cuboid(id=2097455834016)
# out: Cylinder(id=2097455834256)
# out: CylinderSegment(id=2097455837040)
# out: Sphere(id=2097455834352)
# out: Loop(id=2097455833920)
# out: Line(id=2097455834688)
# out: Dipole(id=2097455835504)
# out: CustomSource(id=2097455836320)
# out: Sensor(id=2097455836272)
# out: Collection(id=2097455836656)
```

<!-- #region -->
(docu-position-and-orientation)=


## Position and orientation, `move()` and `rotate()`

All Magpylib objects have the `position` and `orientation` attributes that refer to object position and orientation in the global Cartesian coordinate system. The `position` attribute is a numpy `ndarray, shape (3,) or (m,3)` and denotes `(x,y,z)` coordinates in units of millimeter. By default every object is created at `position=(0,0,0)`. The `orientation` attribute is a scipy [Rotation object](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html). By default the orientation of an object is the unit rotation, given by `rotation=None`.

Both attributes can be either of "scalar" nature, i.e. a single position or a single rotation, or "vectors" when they are arrays of such scalars. The `position` and `orientation` attributes together form an object "path".
<!-- #endregion -->

```python
import magpylib as magpy
sens = magpy.Sensor()

print(sens.position)
# out: [0. 0. 0.]

print(sens.orientation.as_euler('xyz', degrees=True))
# out: [0. 0. 0.]
```

Set and manipulate position and orientation at object initialization,

```python
import magpylib as magpy
sens = magpy.Sensor(position=[(1,1,1), (2,2,2), (3,3,3)])

print(sens.position)
# out: [[1. 1. 1.]  [2. 2. 2.]  [3. 3. 3.]]

print(sens.orientation.as_euler('xyz', degrees=True))
# out: [[0. 0. 0.]  [0. 0. 0.]  [0. 0. 0.]]
```

or after initialization using the setter methods

```python
import magpylib as magpy
from scipy.spatial.transform import Rotation as R

sens = magpy.Sensor()
sens.orientation=R.from_rotvec([(0,0,45), (0,0,90)], degrees=True)

print(sens.position)
# out: [[0. 0. 0.]  [0. 0. 0.]]

print(sens.orientation.as_euler('xyz', degrees=True))
# out: [[ 0.  0. 45.]  [ 0.  0. 90.]]
```

or by making use of the `move` and `rotate` methods.

```python
import magpylib as magpy

sens = magpy.Sensor()
sens.move((1,2,3))
sens.rotate_from_angax(45, 'z')

print(sens.position)
# out: [1. 2. 3.]

print(sens.orientation.as_euler('xyz', degrees=True))
# out: [ 0.  0. 45.]
```

Notice that, when one of the `position` and `orientation` attributes is manipulated, the other is automatically adjusted to the same size. The underlying logic is to pad the edge entries of the path when the length is increased, and to slice from the end of the path when the length is reduced.

The `move` and `rotate` methods provide a lot of functionality that makes it easy to generate more complex motions. They obey the following rules:

1. **Scalar input** is applied to the whole object path, starting with path index `start`.
2. **Vector input** of length n applies the individual n operations to n object path entries, starting with path index `start`.

By default (`start='auto'`) the index is set to `start=0` for scalar input (=move whole object path), and to `start=len(object path)` for vector input (=append to existing object path).

The following example demonstrates this functionality (works similarly for rotations):

```python
import magpylib as magpy

sens = magpy.Sensor(position=((0,0,0), (1,1,1)))

sens.move((1,1,1))                  # scalar input is by default applied to the whole path
print(sens.position)
# out: [[1. 1. 1.]  [2. 2. 2.]]

sens.move([(1,1,1)])                # vector input is by default appended
print(sens.position)
# out: [[1. 1. 1.]  [2. 2. 2.]  [3. 3. 3.]]

sens.move((1,1,1), start=1)         # applied to whole path starting at index 1
print(sens.position)
# out: [[1. 1. 1.]  [3. 3. 3.]  [4. 4. 4.]]

sens.move([(0,0,10)]*3, start=1)    # merges with existing path starting at index 1
print(sens.position)
# out: [[ 1.  1.  1.]  [ 3.  3. 13.]  [ 4.  4. 14.]  [ 4.  4. 14.]]
```

(docu-graphic-output)=

## Graphic output with `show()` and the `style` attribute

When all source and sensor objects are created and all paths are defined `show()` (top level function and method of all Magpylib objects) provides a convenient way to graphically view the geometric arrangement through Matplotlib and Plotly backends.

```{code-cell} ipython3
import magpylib as magpy

src1 = magpy.magnet.Cuboid(magnetization=(0,0,100), dimension=(1,1,1), position=(-10,0,0))
src2 = magpy.magnet.Cylinder(magnetization=(0,0,100), dimension=(1,1), position=(-8,0,0))
src3 = magpy.magnet.CylinderSegment(magnetization=(0,0,100), dimension=(.3,1,1,0,140), position=(-6,0,0))
src4 = magpy.magnet.Sphere(magnetization=(0,0,100), diameter=1, position=(-4,0,0))
src5 = magpy.current.Loop(current=1, diameter=1, position=(-2,0,0))
src6 = magpy.current.Line(current=1, vertices=[(1,0,0), (0,1,0), (-1,0,0), (0,-1,0), (1,0,0)])
src7 = magpy.misc.Dipole(moment=(0,0,100), position=(2,0,0))
sens = magpy.Sensor(pixel=[(0,0,z) for z in (-.1,0,.1)], position=(4,0,0))
magpy.show(src1, src2, src3, src4, src5, src6, src7, sens)

```

Objects can be styled individually by
1. providing a style dictionary at object initialization
2. by making use of style underscore_magic at initialization
3. by manipulating the style properties

```{code-cell} ipython3
import magpylib as magpy

src1 = magpy.magnet.Sphere((0,0,1), 1, (0,0,0), style={'color':'r'})
src2 = magpy.magnet.Sphere((0,0,1), 1, (1,0,0), style_color='g')
src3 = magpy.magnet.Sphere((0,0,1), 1, (2,0,0))
src3.style.color='b'

magpy.show(src1, src2, src3)
```

There are multiple hierarchy levels that descide about the final object style that is displayed:
1. When no input is given, then the default style from `magpy.defaults.display.style` will be applied.
2. Giving an individual object style will overwrite the default value.
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

With the more powerful plotly graphic backend the magnetization is displayed by default through a color gradient and object paths can easily be animated:

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

src = magpy.magnet.Cuboid(
    magnetization=(0,0,1),
    dimension=(1,1,1),
    position=np.linspace((3,0,0), (3,0,5), 80))
src.rotate_from_angax(np.linspace(0, 1000, 80), 'z', anchor=0, start=0)

src.show(backend='plotly', animation=True)
```

(docu-grouping-objects)=

## Grouping objects with collections

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

On one hand, a collection functions like a group. After being added to a collection, it is still possible to manipulate individual objects by name and also by collection index:

```python
src1.move((2,0,0))
col[1].move((-2,0,0))
print(src1.position)   # out: [4. 0. 2.]
print(src2.position)   # out: [-4.  0.  2.]
print(col.position)    # out: [0. 0. 2.]
```

On the other hand, collections can not only be used for grouping, but they function like "compound-objects" themselves. For magnetic field computation a collection that contains sources functions like a single source. When the collection contains sensors it functions like a list of all its sensors.

Collections have their own `position` and `orientation` attributes. Geometric operations (`move`and `rotate` methods, `position` and `orientation` setter) acting on a collection object are individually applied to all child objects - but in such a way that the geometric compound sturcture is maintained. For example, `rotate()` with `anchor=None` rotates all children about `collection.position` (instead of each individual child position) which is demonstrated below:

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

src1 = magpy.magnet.Cuboid((0,0,1), (1,1,1))
src2 = src1.copy(position=(2,0,0))
src3 = src1.copy(position=(-2,0,0))

col = src1 + src2 + src3

col.move(np.linspace((0,0,0), (0,5,0), 30))
col.rotate_from_angax(np.linspace(0, 360, 30), 'y')

col.show(backend='plotly', animation=True, style_path_show=False)
```

(docu-field-computation)=

## Magnetic field computation

Field computation is done through the functions `getB(sources, observers)` and `getH(sources, observers)` that return the B-field and H-field in units of \[mT\] and \[kA/m\] respectively. The argument `sources` can be any Magpylib source object or a list thereof. The argument `observers` can be an array_like of position vectors with shape $(n_1,n_2,n_3,...,3)$ or observer objects (sensors, sensor collections) or a list thereof. `getB` and `getH` are top-level functions, but can also be called as Magpylib object methods.

A most fundamental field computation example is shown below where the magnetic field generated by a cylinder magnet is computed at position $(1,2,3)$:

```python
import magpylib as magpy

src = magpy.magnet.Cylinder(magnetization=(222,333,444), dimension=(2,2))
B = src.getB((1,2,3))
print(B)

# out: [2.52338575 6.12608223 9.49772059]
```

The magnetization input is in units of \[mT\], the B-field is returned in \[mT\], the H-field in \[kA/m\]. Field computation is also valid inside of the magnets.

```python
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

```{eval-rst}
.. plot:: _codes/doc_fieldBH.py
```

The output of the most general field computation through the top level function `magpylib.getB(sources, observers)` is an ndarray of shape `(l,m,k,n1,n2,n3,...,3)` where `l` is the number of input sources, `m` the path length, `k` the number of sensors, `n1,n2,n3,...` the sensor pixel shape or shape of position vector and `3` the three magnetic field components `(Bx,By,Bz)`.

```python
import magpylib as magpy

# three sources
s1 = magpy.misc.Dipole(moment=(0,0,100))
s2 = magpy.current.Loop(current=1, diameter=3)
col = s1 + s2

# two observers with 4x5 pixel
pix = [[(1,2,3)]*4]*5
sens1 = magpy.Sensor(pixel=pix)
sens2 = magpy.Sensor(pixel=pix)

# path of length 11
s1.move([(1,1,1)]*11)

B = magpy.getB([s1,s2,col], [sens1, sens2])
print(B.shape)

# out: (3, 11, 2, 5, 4, 3)
```

The object-oriented interface automatically vectorizes the computation for the user. Similar source types of multiple input-objects are automatically tiled up.

(docu-getB_dict-getH_dict)=

## Functional vs object-oriented `getB` and `getH` use

The `magpylib.getB` and `magpylib.getH` top-level functions also allow the user to avoid the object oriented interface, yet enable usage of the position/orientation implementations via a more functional programming paradigm. The input arguments must be shape `(n,x)` vectors/lists/tuple. Static inputs e.g. of shape `(x,)` are automatically tiled up to shape `(n,x)`. Depending on the source type defined by a string instead of a magpylib source object, different input arguments are expected (see docstring for details).

```python
import magpylib as magpy

# observer positions
observer_pos = [(0,0,x) for x in range(5)]

# magnet dimensions
dim = [(d,d,d) for d in range(1,6)]

# functional-oriented getB computation - magnetization is automatically tiled
B = magpy.getB(
    'Cuboid',
    observer_pos,
    magnetization=(0,0,1000),
    dimension=dim,
)
print(B)

# out: [[  0.           0.         666.66666667]
#       [  0.           0.         435.90578315]
#       [  0.           0.         306.84039675]
#       [  0.           0.         251.12200327]
#       [  0.           0.         221.82226656]]
```

The `getB` and `getH` functions used this way can be up to 2 times faster than the object oriented interface. However, this requires that the user knows how to properly generate the vectorized input.

(docu-direct-access)=

## Direct access to field implementations

For users who do not want to use the position/orientation interface, Magpylib offers direct access to the vectorized analytical implementations that lie at the bottom of the library through the `magpylib.lib` subpackage. Details on the implementations can be found in the respective function docstrings.

```python
import numpy as np
import magpylib as magpy

mag = np.array([(100,0,0)]*5)
dim = np.array([(1,2,45,90,-1,1)]*5)
obs_pos = np.array([(0,0,0)]*5)

B = magpy.lib.magnet_cylinder_section_core(mag, dim, obs_pos)
print(B)

# out: [[   0.           0.        -186.1347833]
#       [   0.           0.        -186.1347833]
#       [   0.           0.        -186.1347833]
#       [   0.           0.        -186.1347833]
#       [   0.           0.        -186.1347833]]
```

As all input checks, coordinate transformations and position/orientation implementation are avoided, this is the fastest way to compute fields in Magpylib.

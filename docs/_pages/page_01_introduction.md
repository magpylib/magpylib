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

(intro)=

# Introduction to Magpylib

This section gives an introduction to the Magpylib API. Many practical examples how to use Magpylib can be found in the example galleries.
Detailed package, class, method and function documentations are found in the library docstrings {ref}`genindex`.

## Contents

- {ref}`intro-idea`
- {ref}`intro-when-to-use`
- {ref}`intro-magpylib-objects`
- {ref}`intro-position-and-orientation`
- {ref}`intro-graphic-output`
- {ref}`intro-collections`
- {ref}`intro-field-computation`
- {ref}`intro-direct-interface`
- {ref}`intro-core-functions`


(intro-idea)=

## The idea behind Magpylib

Magpylib provides fast and accurate magnetic field computation through **analytical solutions** to permanent magnet and current problems. Details on how these solutions are mathematically obtained can be found in the {ref}`physComp` section.

Magpylib couples the field computation to a position and orientation interface. The idea is that users can create Magpylib objects like sensors, magnets, currents, etc. with defined position and orientation in a global coordinate system. These objects can then be easily manipulated, displayed, grouped and used for magnetic field computation. For users who would like to avoid the object oriented interface, the field implementations can also be accessed directly.

(intro-when-to-use)=

## When can you use Magpylib ?

The analytical solutions are exact when there is no material response and natural boundary conditions can be assumed. In general, Magpylib is at its best when dealing with air-coils (no eddy currents) and high grade permanent magnet assemblies (Ferrite, NdFeB, SmCo or similar materials).

When **magnet** permeabilities are below $\mu_r < 1.1$ the error typically undercuts 1-5 % (long magnet shapes are better, large distance from magnet is better). Demagnetization factors are not automatically included at this point. The line **current** solutions give the exact same field as outside of a wire that carries a homogenous current. For more details check out the {ref}`physComp` section.

Magpylib only provides solutions for simple geometric forms (cuboids, clinders, lines, ...). How complex shapes can be constructed from these simple base shapes is described in {ref}`examples-complex-forms`.

(intro-magpylib-objects)=

## The Magpylib objects

The most convenient way for working with Magpylib is through the **object oriented interface**. Magpylib objects represent magnetic field sources, sensors and collections with various defining attributes and methods. The following classes are implemented:

**Magnets**

All magnet objects have the `magnetization` attribute which must be of the format $(m_x, m_y, m_z)$ and denotes the homogeneous magnetization vector in units of \[mT\]. It is often referred to as the remanence ($B_r=\mu_0 M$) in material data sheets. All magnets can be used as Magpylib `sources` input.

- **`Cuboid`**`(magnetization, dimension, position, orientation, style)` represents a magnet with cuboid shape. `dimension` has the format $(a,b,c)$ and denotes the sides of the cuboid in units of \[mm\]. By default (`position=(0,0,0)`, `orientation=None`) the center of the cuboid lies in the origin of the global coordinates, and the sides are parallel to the coordinate axes.

- **`Cylinder`**`(magnetization, dimension, position, orientation, style)` represents a magnet with cylindrical shape. `dimension` has the format $(d,h)$ and denotes diameter and height of the cylinder in units of \[mm\]. By default (`position=(0,0,0)`, `orientation=None`) the center of the cylinder lies in the origin of the global coordinates, and the cylinder axis coincides with the z-axis.

- **`CylinderSegment`**`(magnetization, dimension, position, orientation, style)` represents a magnet with the shape of a cylindrical ring section. `dimension` has the format $(r_1,r_2,h,\varphi_1,\varphi_2)$ and denotes inner radius, outer radius and height in units of \[mm\] and the two section angles $\varphi_1<\varphi_2$ in \[deg\]. By default (`position=(0,0,0)`, `orientation=None`) the center of the full cylinder lies in the origin of the global coordinates, and the cylinder axis coincides with the z-axis.

- **`Sphere`**`(magnetization, diameter, position, orientation, style)` represents a magnet of spherical shape. `diameter` is the sphere diameter $d$ in units of \[mm\]. By default (`position=(0,0,0)`, `orientation=None`) the center of the sphere lies in the origin of the global coordinates.

**Currents**

All current objects have the `current` attribute which must be a scalar $i_0$ and denotes the electrical current in units of \[A\]. All currents can be used as Magpylib `sources` input.

- **`Loop`**`(current, diameter, position, orientation, style)` represents a circular current loop where `diameter` is the loop diameter $d$ in units of \[mm\]. By default (`position=(0,0,0)`, `orientation=None`) the loop lies in the xy-plane with its center in the origin of the global coordinates.

- **`Line`**`(current, vertices, position, orientation, style)` represents electrical current segments that flow in a straight line from vertex to vertex. By default (`position=(0,0,0)`, `orientation=None`) the locally defined vertices have the same position in the global coordinates.

**Other**

- **`Dipole`**`(moment, position, orientation, style)` represents a magnetic dipole moment with moment $(m_x,m_y,m_z)$ given in \[mT mm³]. For homogeneous magnets the relation moment=magnetization$\times$volume holds. Can be used as Magpylib `sources` input.

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

(intro-position-and-orientation)=

## Position and orientation

All Magpylib objects have the `position` and `orientation` attributes that refer to position and orientation in the global Cartesian coordinate system. The `position` attribute is a numpy ndarray, shape (3,) or (m,3) and denotes the coordinates $(x,y,z)$ in units of millimeter. By default every object is created at `position=(0,0,0)`. The `orientation` attribute is a scipy [Rotation object](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html) and denotes the object rotation (relative to its initial state), e.g. in terms of Euler angles $(\phi, \psi, \theta)$. By default the orientation of a Magpylib object is the unit rotation, generated by `orientation=None`.

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

Set `position` and `orientation` attributes at object initialization,

```{code-cell} ipython3
sens = magpy.Sensor(position=[(1,1,1), (2,2,2), (3,3,3)])

print(sens.position)
print(sens.orientation.as_euler('xyz', degrees=True))
```

or after initialization using the setter methods:

```{code-cell} ipython3
sens = magpy.Sensor()
sens.orientation=R.from_rotvec([(0,0,45), (0,0,90)], degrees=True)

print(sens.position)
print(sens.orientation.as_euler('xyz', degrees=True))
```

Add relative position and orientation to the existing path with the `move` and `rotate` methods:

```{code-cell} ipython3
sens.move((1,2,3))
sens.rotate_from_angax(45, 'z')

print(sens.position)
print(sens.orientation.as_euler('xyz', degrees=True))
```

Notice that, when one of the `position` and `orientation` attributes is manipulated, the other is automatically adjusted to the same size.

```{note}
The underlying Magpylib logic is to always pad the edge entries of a path when the length is increased, and to slice from the end of the path when the length is reduced by an operation. More details can be found in the example gallery {ref}`examples-edge-padding-end-slicing`.
```

The **`move`** and **`rotate`** methods provide a lot of additional functionality that makes it easy to move objects around and to generate more complex paths. They obey the following rules:

```{note}
- Scalar input is applied to the whole object path, starting with path index `start`. With the default `start='auto'` the index is set to `start=0` and the functionality is *moving objects around*.
- Vector input of length $n$ applies the individual $n$ operations to $n$ object path entries, starting with path index `start`. Padding applies when the input exceeds the existing path. With the default `start='auto'` the index is set to `start=len(object path)` and the functionality is *appending paths*.
```

The following example demonstrates this functionality (works similarly for rotations):

```python
import magpylib as magpy

sens = magpy.Sensor()

sens.move((1,1,1))                                      # scalar input is by default applied
print(sens.position)                                    # to the whole path
# out: [1. 1. 1.]

sens.move([(1,1,1), (2,2,2)])                           # vector input is by default appended
print(sens.position)                                    # to the existing path
# out: [[1. 1. 1.]  [2. 2. 2.]  [3. 3. 3.]]

sens.move((1,1,1), start=1)                             # scalar input and start=1 is applied
print(sens.position)                                    # to whole path starting at index 1
# out: [[1. 1. 1.]  [3. 3. 3.]  [4. 4. 4.]]

sens.move([(0,0,10), (0,0,20)], start=1)                # vector input and start=1 merges
print(sens.position)                                    # the input with the existing path
# out: [[ 1.  1.  1.]  [ 3.  3. 13.]  [ 4.  4. 24.]]    # starting at index 1.
```

Deeper insights into how paths are generated through merging, slicing and padding are given in the example gallery {ref}`examples-paths`.

(intro-graphic-output)=

## Graphic output

When all Magpylib objects and their paths have been created, `show()` provides a convenient way to graphically display the geometric arrangement using the **Matplotlib** and **Plotly** graphic libraries. By installation default, objects are displayed with the Matplotlib backend. Matplotlib comes with Magpylib installation, Plotly must be installed by hand. Details on how to switch between graphic backends using the `backend` setting is given in the example gallery {ref}`examples-backend`.

When `show` is called, it generates a new figure which is then automatically displayed. To bring the output to a given, user-defined figure, the `canvas` kwarg can be used. This is explained in detail in the example gallery {ref}`examples-canvas`.

The following example shows the graphical representation of various Magpylib objects and their paths using the default Matplotlib graphic backend.

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

src1 = magpy.magnet.Cuboid(magnetization=(0,100,0), dimension=(1,1,1), position=(-7,0,0))
src2 = magpy.magnet.Cylinder(magnetization=(0,0,100), dimension=(1,1), position=(-5,0,0))
src3 = magpy.magnet.CylinderSegment(magnetization=(0,0,100), dimension=(.3,1,1,0,140), position=(-3,0,0))
src4 = magpy.magnet.Sphere(magnetization=(0,0,100), diameter=1, position=(-1,0,0))
src5 = magpy.current.Loop(current=1, diameter=1, position=(1,0,0))
src6 = magpy.current.Line(current=1, vertices=[(1,0,0), (0,1,0), (-1,0,0), (0,-1,0), (1,0,0)], position=(3,0,0))
src7 = magpy.misc.Dipole(moment=(0,0,100), position=(5,0,0))
sens = magpy.Sensor(pixel=[(0,0,z) for z in (-.5,0,.5)], position=(7,0,0))

src6.move(np.linspace((0,0,0), (0,0,7), 20))
src1.rotate_from_angax(np.linspace(0, -90, 20), 'y', anchor=0)

magpy.show(src1, src2, src3, src4, src5, src6, src7, sens)
```

Notice that, objects and their paths are automatically assigned different colors, the magnetization vector, current directions and dipole objects are indicated by arrows and sensors are shown as tri-colored coordinate cross with pixel as markers.

How objects are represented graphically (color, line thickness, ect.) is defined by the `style` properties. An exhaustive review of object styling can be found in the example gallery {ref}`examples-graphic-styles`. What you see above is the **default style**, with all settings stored in `magpy.defaults.display.style`.

It is possible to set **individual styles** for each object by:
1. Providing a style dictionary at object initialization.
2. Making use of style underscore_magic at initialization.
3. Manipulating the object style properties.

```{code-cell} ipython3
import magpylib as magpy

# using a style dictionary
src1 = magpy.magnet.Sphere((0,0,1), 1, (0,0,0), style={'color':'r'})

# using style underscore_magic
src2 = magpy.magnet.Sphere((0,0,1), 1, (1,0,0), style_color='g')

# manipulating the object style properties
src3 = magpy.magnet.Sphere((0,0,1), 1, (2,0,0))
src3.style.color='b'

magpy.show(src1, src2, src3)
```

A list of all style attribues can be found in {ref}`examples-list-of-styles`.
```{note}
For practical purposes make use of the Magpylib error messages that will tell you which style arguments are allowed when an unknown input is provided.
```

There is a strict **style hierarchy** that descide about the final graphic object representation:
1. When no input is given, the default style from `magpy.defaults.display.style` will be applied.
2. Individual object styles will take precedence over the default values.
3. Setting a global style in `show()` will override all other inputs.

```{code-cell} ipython3
import magpylib as magpy

src1 = magpy.magnet.Sphere((0,0,1), 1, [(0,0,0), (0,0,2)])
src2 = magpy.magnet.Sphere((0,0,1), 1, [(1,0,0), (1,0,2)])
src3 = magpy.magnet.Sphere((0,0,1), 1, [(2,0,0), (2,0,2)])

# default style
magpy.defaults.display.style.base.color='red'
magpy.defaults.display.style.base.path.line.style=':'

# individual style
src1.style.color = 'g'
src2.style.path.line.style = '-'

# global style override
magpy.show(src1, src2, src3, style_path_line_style='--')
```

Finally, the remaining arguments of `show(*objects, zoom=0, animation=False, markers=None, backend=None, canvas=None, **kwargs)` include,

- `zoom` which is a positive number that defines the figure zoom level.
- `markers` which displays position markers, given as array_like of shape (n,3).
- `animation` which enables path animation when using the Plotly backend, see {ref}`examples-animation`.
- `kwargs` which are handed directly to the respective graphic backends.

More information on graphic output can be found in the example gallery {ref}`example-gallery-show`.

(intro-collections)=

## Collections

The top level class `magpylib.Collection` allows users to group sources by reference for common manipulation. Objects that are part of a collection are called **children** of that collection. All operations acting on a collection are individually applied to its children.

```python
import magpylib as magpy

src1 = magpy.magnet.Sphere(magnetization=(1,2,3), diameter=1, position=(2,0,0))
src2 = magpy.current.Loop(current=1, diameter=1, position=(-2,0,0))
coll = magpy.Collection(src1, src2)

print(src1.position)   # out: [ 2.  0.  0.]
print(src2.position)   # out: [-2.  0.  0.]
print(coll.position)    # out: [ 0.  0.  0.]

coll.move(((0,0,2)))

print(src1.position)   # out: [ 2.  0.  2.]
print(src2.position)   # out: [-2.  0.  2.]
print(coll.position)    # out: [ 0.  0.  2.]
```

Collections function primarily like **groups**. Magpylib objects can be part of multiple collections. After being added to a collection, it is still possible to manipulate the individual objects by reference, and also by collection index:

```python
src1.move((2,0,0))
coll[1].move((-2,0,0))

print(src1.position)   # out: [ 4.  0.  2.]
print(src2.position)   # out: [-4.  0.  2.]
print(coll.position)    # out: [ 0.  0.  2.]
```

A detailed review of collection properties and construction is provided in the example gallery {ref}`examples-collections-construction`.

For advanced applications, collections also follow the **compound object** philosophy, which means that a collection behaves itself like a single object. Notice in the examples above, that `coll` has its own `position` and `orientation` attributes. Geometric operations acting on a collection object are individually applied to all child objects - but in such a way that the geometric compound structure is maintained. For example, applying `rotate` with `anchor=None` rotates all children about `collection.position` instead of the individual child positions. This is demonstrated in the following example:

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

src1 = magpy.magnet.Cuboid((0,0,1), (1,1,1), (2,0,0))
src2 = src1.copy(position=(-2,0,0))
coll = src1 + src2

coll.move(np.linspace((0,0,0), (0,2,0), 30))
coll.rotate_from_angax(np.linspace(0, 180, 30), 'y')

coll.show(animation=True, style_path_show=False, backend='plotly')
```

Here the Plotly graphic backend is used to animate the path. It should be noted that collections have their own `style` attributes. Their paths are displayed in `show` and all children are automatically given their parent color. More insights on the collection compound functionality is provided in the example gallery {ref}`examples-collections-compound`.

(intro-field-computation)=

## Magnetic field computation

Magnetic field computation in Magpylib is achieved through:

- **`getB`**`(sources, observers)` computes the B-field seen by `observers` generated by `sources` in units of \[mT\]
- **`getH`**`(sources, observers)` computes the H-field seen by `observers` generated by `sources` in units of \[kA/m\]

The argument `sources` can be any Magpylib **source object** or a list thereof. The argument `observers` can be an array_like of position vectors with shape $(n_1,n_2,n_3,...,3)$, any Magpylib **observer object** or a list thereof. `getB` and `getH` will automatically determine all source-position combinations, where positions will depend on all objects and their attributes (paths, pixels), compute the field and return it in the input format.

```{note}
The output of the most general field computation `getB(sources, observers)` is an ndarray of shape `(l, m, k, n1, n2, n3, ..., 3)` where `l` is the number of input sources, `m` the maximal object path length, `k` the number of sensors, `n1,n2,n3,...` the sensor pixel shape or the shape of a position vector input and `3` the three magnetic field components $(B_x, B_y, B_z)$.
```

A most fundamental field computation example is shown below where the B-field generated by a cylinder magnet is computed at position $(1,2,3)$:

```{code-cell} ipython3
import magpylib as magpy

src = magpy.magnet.Cylinder(magnetization=(222,333,444), dimension=(2,2))
B = magpy.getB(src, (1,2,3))
print(B)
```

When dealing with multiple observer positions, `getB` and `getH` will return the field in the shape of the observer input. In the following example B- and H-field of a cuboid magnet are computed on a grid with single function calls, and then displayed using Matplotlib:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy

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


When input objects have different path lengths, all shorter paths are padded (= objects remain "static") beyond their end, following the {ref}`examples-edge-padding-end-slicing` logic. In the following example we compute the field of three sources, one with a length 11 path, and two sensors, each with pixel shape (4,5):

```{code-cell} ipython3
import magpylib as magpy

# 3 sources, one with length 11 path
s1 = magpy.misc.Dipole(moment=(0,0,100), position=[(1,1,1)]*11)
s2 = magpy.current.Loop(current=1, diameter=3)
coll = s1 + s2

# 2 observers with 4x5 pixel
pix = [[(1,2,3)]*4]*5
sens1 = magpy.Sensor(pixel=pix)
sens2 = magpy.Sensor(pixel=pix)

B = magpy.getB([s1,s2,coll], [sens1, sens2])
print(B.shape)
```

In terms of **performance** it must be noted that Magpylib automatically vectorizes all computations when `getB` and `getH` are called. This reduces the compuation time dramically for large inputs. For maximal performance try to make all field computations with as few calls to `getB` and `getH` as possible.

Finally, the remaining arguments of the field computation functions are
- `sumup=False` which allows users to sum over the field of all sources.
- `squeeze=True` which will squeeze output dimensions of size 1 rather than returning the complete shape (l,m,k,n1,n2...,3).


(intro-direct-interface)=

## Direct interface

The direct interface allows users to bypass the object oriented functionality of Magpylib. The magnetic field is computed for a set of arbitrary input instances by providing the top level functions `getB` and `getH` with 

1. a string denoting the source type for the `sources` argument,
2. an array_like of shape (3,) or (n,3) giving the positions for the `observers` argument,
3. a dictionary with array_likes of shape (x,) or (n,x) for all other inputs.

All "scalar" inputs of shape (x,) are automatically tiled up to shape (n,x), and for every of the n given instances the field is computed and returned with shape (n,3). The allowed source types are similar to the Magpylib source class names, and the required dictionary inputs are the respective class inputs. Details can be found in the respective docstrings.

In the following example we compute the cuboid field for 5 different position and dimension input instances and "constant" magnetization:

```{code-cell} ipython3
import magpylib as magpy

obs = [(0,0,x) for x in range(5)]
dim = [(d,d,d) for d in range(1,6)]

B = magpy.getB(
    'Cuboid',
    obs,
    magnetization=(0,0,1000),
    dimension=dim)

print(B)
```

The direct interface is convenient for users who work with complex inputs or favor a more functional programming paradigm. It is typically faster than the object oriented interface, but it also requires that users know how to generate the inputs efficiently with numpy (e.g. `np.arange`, `np.linspace`, `np.tile`, `np.repeat`, ...).

(intro-core-functions)=

## Core functions

At the heart of Magpylib lies a set of core functions that are our implementations of the analytical field expressions, see {ref}`physcomp`. For users who are not interested in the position/orientation interface, the `magpylib.core` subpackage gives direct access to these functions. Inputs are ndarrays of shape (n,x). Details can be found in the respective function docstrings.

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

mag = np.array([(100,0,0)]*5)
dim = np.array([(1,2,3,45,90)]*5)
pos = np.array([(0,0,0)]*5)

B = magpy.core.magnet_cylinder_segment_field(mag, dim, pos, field='B')
print(B)
```

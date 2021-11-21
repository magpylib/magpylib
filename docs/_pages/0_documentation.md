(docu)=

# Documentation

Brief overview and some critical information.

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

Magpylib provides fully tested and properly implemented analytical solutions to permanent magnet and current problems, and thus gives quick access to extremely fast and accurate magnetic field computation. Details on how these solutions are mathematically obtained can be found in the {ref}`physComp` section.

The core API of Magpylib is object oriented and couples the field computation to a position/orientation interface. The idea is that users can create Magpylib objects like sensors, magnets, currents, etc. with defined position and orientation in a global coordinate system. These objects can be easily manipulated, displayed, grouped and used for user-friendly field computation. For users who would like to avoid the object oriented interface, the field implementations can also be accessed directly.

(docu-when-to-use)=

## When can you use Magpylib ?

The analytical solutions are exact when there is no material response and natural boundary conditions can be assumed.

For permanent magnets, when (remanent) permeabilities are below $\mu_r < 1.1$ the error is typically below 1-5 % (long magnet shapes are better, large distance from magnet is better). Demagnetization factors are not automatically included at this point. With these factors the precision can be increased to below 1 % error. Error estimation as a result of the material response is evaluated in more detail in the appendix of [Malagò2020](https://www.mdpi.com/1424-8220/20/23/6873).

The line-current solutions give the exact same field as outside of a wire which carries a homogenous current.

Magpylib is at its best when dealing with air-coils (no eddy currents) and high grade permanent magnet assemblies (Ferrite, NdFeB, SmCo or similar materials). For more details check out the {ref}`physComp` section.

Magpylib only provides solutions for simple forms. However, in Magnetostatics the superposition principle holds: the total magnetic field is given by the (vector-)sum of all the fields of all sources. For magnets this means that complex magnet shapes can be constructed from simple forms. Specifically, it is possible to cut-out a part of a magnet simply by placing a second magnet with opposite magnetization inside the first one.

(docu-magpylib-objects)=

## Magpylib objects

The most convenient way to compute magnetic fields is through the object oriented interface. Magpylib objects represent magnetic field sources and sensors with various defining attributes.

```python
import magpylib as magpy

# magnets
src1 = magpy.magnet.Cuboid(magnetization=(0,0,1000), dimension=(1,2,3))
src2 = magpy.magnet.Cylinder(magnetization=(0,1000,0), dimension=(1,2))
src3 = magpy.magnet.CylinderSegment(magnetization=(0,1000,0), dimension=(1,2,2,45,90))
src4 = magpy.magnet.Sphere(magnetization=(1000,0,0), diameter=1)

# currents
src5 = magpy.current.Loop(current=15, diameter=3)
src6 = magpy.current.Line(current=15, vertices=[(0,0,0), (1,2,3)])

# misc
src7 = magpy.misc.Dipole(moment=(100,200,300))

# sensor
sens = magpy.Sensor()

# print object representation
for obj in [src1, src2, src3, src4, src5, src6, src7, sens]:
    print(obj)

# out: Cuboid(id=1331541150016)
# out: Cylinder(id=1331541148672)
# out: CylinderSegment(id=1331541762784)
# out: Sphere(id=1331541762448)
# out: Loop(id=1331543166304)
# out: Line(id=1331543188720)
# out: Dipole(id=1331543189632)
# out: Sensor(id=1331642701760)
```

(docu-position-and-orientation)=

## Position and orientation

All Magpylib objects are endowed with `position` `(ndarray, shape (m,3))` and `orientation` `(` [scipy Rotation object](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html) `, shape (m,3))` attributes that describe their state in a global coordinate system. Details on default object position (0-position) and alignment (unit-rotation) are found in the respective docstrings.

```python
import magpylib as magpy
sens = magpy.Sensor()
print(sens.position)
print(sens.orientation.as_euler('xyz', degrees=True))

# out: [0. 0. 0.]
# out: [0. 0. 0.]
```

Manipulate position and orientation attributes directly through source attributes, or by using built-in `move`, `rotate` or `rotate_from_angax` methods.

```python
import magpylib as magpy
from scipy.spatial.transform import Rotation as R

sens = magpy.Sensor(position=(1,1,1))
print(sens.position)

sens.move((1,1,1))
print(sens.position)

# out: [1. 1. 1.]
# out: [2. 2. 2.]

sens = magpy.Sensor(orientation=R.from_euler('x', 10, degrees=True))
print(sens.orientation.as_euler('xyz'))

sens.rotate(R.from_euler('x', 10, degrees=True)))
print(sens.orientation.as_euler('xyz'))

sens.rotate_from_angax(angle=10, axis=(1,0,0))
print(sens.orientation.as_euler('xyz'))

# out: [10 0. 0.]
# out: [20 0. 0.]
# out: [30 0. 0.]
```

Source position and orientation attributes can also represent complete source paths in the global coordinate system. Such paths can be generated conveniently using the `move` and `rotate` methods.

```python
import magpylib as magpy

src = magpy.magnet.Cuboid(magnetization=(1,2,3), dimension=(1,2,3))
src.move([(1,1,1),(2,2,2),(3,3,3),(4,4,4)], start='append')
print(src.position)

# out: [[0. 0. 0.]  [1. 1. 1.]  [2. 2. 2.]  [3. 3. 3.]  [4. 4. 4.]]
```

Details on rotation arguments, and how to conveniently generate complex paths are found in the docstings and some examples below.

(docu-grouping-objects)=

## Grouping objects with `Collection`

The top level class `magpylib.Collection` allows a user to group sources for common manipulation. A Collection functions like a list of source objects extended by Magpylib source methods: all operations applied to a Collection are applied to each source individually. Specific sources in the Collection can still be accessed and manipulated individually.

```python
import magpylib as magpy

src1 = magpy.magnet.Cuboid(magnetization=(0,0,11), dimension=(1,2,3))
src2 = magpy.magnet.Cylinder(magnetization=(0,22,0), dimension=(1,2))
src3 = magpy.magnet.Sphere(magnetization=(33,0,0), diameter=2)

col = magpy.Collection(src1, src2, src3)
col.move((1,2,3))
src1.move((1,2,3))

for src in col:
    print(src.position)

# out: [2. 4. 6.]
# out: [1. 2. 3.]
# out: [1. 2. 3.]
```

Magpylib sources have addition and subtraction methods defined, adding up to a Collection, or removing a specific source from a Collection.

```python
import magpylib as magpy

src1 = magpy.misc.Dipole(moment=(1,2,3))
src2 = magpy.current.Loop(current=1, diameter=2)
src3 = magpy.magnet.Sphere(magnetization=(1,2,3), diameter=1)

col = src1 + src2 + src3

for src in col:
    print(src)

# out: Dipole(id=2158565624128)
# out: Loop(id=2158565622784)
# out: Sphere(id=2158566236896)

col - src1

for src in col:
    print(src)

# out: Loop(id=2158565622784)
# out: Sphere(id=2158566236896)
```

(docu-graphic-output)=

## Graphic output with `display`

When all source and sensor objects are created and all paths are defined `display` (top level function and method of all Magpylib objects) provides a convenient way to graphically view the geometric arrangement through Matplotlib.

```python
import magpylib as magpy

# create a Collection of three sources
s1 = magpy.magnet.Sphere(magnetization=(0,0,100), diameter=3, position=(3,0,0))
s2 = magpy.magnet.Cuboid(magnetization=(0,0,100), dimension=(2,2,2), position=(-3,0,0))
col = s1 + s2

# generate a spiral path
s1.move([(.2,0,0)]*100, increment=True)
s2.move([(-.2,0,0)]*100, increment=True)
col.rotate_from_angax([5]*100, 'z', anchor=0, increment=True, start=0)

# display
col.display(path=10)
```

```{eval-rst}
.. plot:: _codes/doc_collection_display.py
```

```{seealso} Various
    arguments like `path, markers, canvas, zoom, backend` and `style` can be used to customize the output and are described in the docstrings. More detail is also available in the examples gallery {ref}`display_styling_example`
```

(docu-field-computation)=

## Field computation

Field computation is done through the `getB` and `getH` function/methods. They always require `sources` and `observers` inputs. Sources are single Magpylib source objects, Collections or lists thereof.  Observers are arbitrary tensors of position vectors `(shape (n1,n2,n3,...,3))`, sensors or lists thereof. A most fundamental field computation example is

```python
from magpylib.magnet import Cylinder

src = Cylinder(magnetization=(222,333,444), dimension=(2,2))
B = src.getB((1,2,3))
print(B)

# out: [-2.74825633  9.77282601 21.43280135]
```

The magnetization input is in units of \[mT\], the B-field is returned in \[mT\], the H-field in \[kA/m\]. Field computation is also valid inside of the magnets.

```python
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy

# define Pyplot figure
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

The output of the most general field computation through the top level function `magpylib.getB(sources, observers)` is an ndarray of shape `(l,m,k,n1,n2,n3,...,3)` where `l` is the number of input sources, `m` the pathlength, `k` the number of sensors, `n1,n2,n3,...` the sensor pixel shape or shape of position vector and `3` the three magnetic field components `(Bx,By,Bz)`.

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

## getB_dict and getH_dict

The `magpylib.getB_dict` and `magpylib.getH_dict` top-level functions avoid the object oriented interface, yet enable usage of the position/orientation implementations. The input arguments must be shape `(n,x)` vectors/lists/tuple. Static inputs e.g. of shape `(x,)` are automatically tiled up to shape `(n,x)`. Depending on the `source_type`, different input arguments are expected (see docstring for details).

```python
import magpylib as magpy

# observer positions
poso = [(0,0,x) for x in range(5)]

# magnet dimensions
dim = [(d,d,d) for d in range(1,6)]

# getB_dict computation - magnetization is automatically tiled
B = magpy.getB_dict(
    source_type='Cuboid',
    magnetization=(0,0,1000),
    dimension=dim,
    observer=poso)
print(B)

# out: [[  0.           0.         666.66666667]
#       [  0.           0.         435.90578315]
#       [  0.           0.         306.84039675]
#       [  0.           0.         251.12200327]
#       [  0.           0.         221.82226656]]
```

The `getBH_dict` functions can be up to 2 times faster than the object oriented interface. However, this requires that the user knows how to properly generate the vectorized input.

(docu-direct-access)=

## Direct access to field implementations

For users who do not want to use the position/orientation interface, Magpylib offers direct access to the vectorized analytical implementations that lie at the bottom of the library through the `magpylib.lib` subpackage. Details on the implementations can be found in the respective function docstrings.

```python
import numpy as np
import magpylib as magpy

mag = np.array([(100,0,0)]*5)
dim = np.array([(1,2,45,90,-1,1)]*5)
poso = np.array([(0,0,0)]*5)

B = magpy.lib.magnet_cyl_tile_H_Slanovc2021(mag, dim, poso)
print(B)

# out: [[   0.           0.        -186.1347833]
#       [   0.           0.        -186.1347833]
#       [   0.           0.        -186.1347833]
#       [   0.           0.        -186.1347833]
#       [   0.           0.        -186.1347833]]
```

As all input checks, coordinate transformations and position/orientation implementation are avoided, this is the fastest way to compute fields in Magpylib.

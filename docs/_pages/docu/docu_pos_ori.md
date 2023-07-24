---
orphan: true
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(docu-position)=

# Position, Orientation, Paths

(intro-position-and-orientation)=

## Position and orientation

All Magpylib objects have the `position` and `orientation` attributes that refer to position and orientation in the global Cartesian coordinate system. The `position` attribute is a numpy ndarray, shape (3,) or (m,3) and corresponds to the coordinates $(x,y,z)$ in units of mm. By default every object is created in the origin of the global coordinate system, at `position=(0,0,0)`. The `orientation` attribute is a scipy [Rotation object](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html) and corresponds to the object orientation relative to its initial state. By default objects are created with initial orientation, `orientation=None`, which is the unit rotation. The initial orientation of every object, e.g. current loop lies in the x-y plane, is described in the respective class docstrings.

```python
import magpylib as magpy

# init object with default values
sensor = magpy.Sensor()
print(sensor.position)                                     # out: [0. 0. 0.]
print(sensor.orientation.as_euler('xyz', degrees=True))    # out: [0. 0. 0.]
```

Set absolute object position and orientation attributes at object initialization or directly through the properties.

```python
import magpylib as magpy
from scipy.spatial.transform import Rotation as R

# set attributes at initialization
sensor = magpy.Sensor(position=(1,1,1))
print(sensor.position)                                     # out: [1. 1. 1.]
print(sensor.orientation.as_euler('xyz', degrees=True))    # out: [0. 0. 0.]

# set properties directly
sensor.orientation = R.from_rotvec((0,0,45), degrees=True)
print(sensor.position)                                     # out: [1. 1. 1.]
print(sensor.orientation.as_euler('xyz', degrees=True))    # out: [ 0.  0. 45.]
```

The `move` and `rotate` methods are powerful tools to change the relative position and orientation of an existing object. **Move** the object by a `displacement` vector. **Rotate** an object by specifying the angle of rotation `angle` (scalar), an axis of rotation `axis` (vector or `'x'`, `'y'`, `'z'`) and an anchor point `anchor` (vector) through which the rotation axis passes through. By default `anchor=self.position`, meaning that the object rotates about itself.

```python
import magpylib as magpy

# init object with default values
sensor = magpy.Sensor()

# move
sensor.move(displacement=(1,1,3))
print(sensor.position)                                     # out: [1. 1. 3.]
print(sensor.orientation.as_euler('xyz', degrees=True))    # out: [ 0.  0.  0.]

# rotate about self
sensor.rotate_from_angax(angle=45, axis='z')
print(sensor.position)                                     # out: [1. 1. 3.]
print(sensor.orientation.as_euler('xyz', degrees=True))    # out: [ 0.  0. 45.]

# rotate with anchor
sensor.rotate_from_angax(angle=90, axis='z', anchor=(0,0,0))
print(sensor.position)                                     # out: [-1. 1. 3.]
print(sensor.orientation.as_euler('xyz', degrees=True))    # out: [ 0.  0. 135.]
```


(intro-paths)=

## Paths

The attributes `position` and `orientation` can be either of **"scalar"** nature, i.e. a single position or a single rotation like in the examples above, or **"vectors"** when they are arrays of such scalars. The two attributes together define an object **"path"**. Paths should always be used when modeling object motion as the magnetic field is computed on the whole path with increased performance.

With vector inputs, the `move` and `rotate` methods provide *append* and *merge* functionality.  The following example shows how a path `path1` is assigned to a magnet object, how `path2` is appended with `move` and how `path3` is merged on top starting at path index 25.

```{code-cell} ipython3
import numpy as np
from magpylib.magnet import Cylinder

magnet = Cylinder(magnetization=(100,0,0), dimension=(2,2))

# assign path
path1 = np.linspace((0,0,0), (0,0,5), 20)
magnet.position = path1

# append path
path2 = np.linspace((0,0,0), (0,10,0), 40)
magnet.move(path2[1:])

# merge path
path3 = np.linspace(0, 360, 20)
magnet.rotate_from_angax(angle=path3, axis='z', anchor=0, start=25)

magnet.show(backend='plotly')
```


Notice that when one of the `position` and `orientation` attributes are modified in length, the other is automatically adjusted to the same length. A detailed outline of the functionality of `position`, `orientation`, `move`, `rotate` and paths is given in examples `examples-paths`.

# paths

Magpylib objects all have `position` and `orientation` attributes in the global reference frame. These attributes can be single instances describing a single position and a single orientation of an object, but they can also be vectors of positions and orientations. In this case the attributes represent multiple object locations and orientaions and are referred to as a **path**. When the field of an object is computed, it is automatically computed for the whole path. The reason is that computation is much more efficient when all instances are computed in a single operation. The introduction to `position` and `orientation` attributes and path can be found in {ref}`intro-position-and-orientation`. Here we show how to work with paths properly.


(examples-assign-absolute-path)=
## Assigning absolute paths

Absolute object paths are assigned at initialization or through the object properties.

```{code-cell} ipython3
import numpy as np
from scipy.spatial.transform import Rotation as R
import magpylib as magpy

ts = np.linspace(0, 10, 51)
pos = np.array([(t, 0, np.sin(t)) for t in ts])
ori = R.from_rotvec(np.array([(0, -np.cos(t)*0.785, 0) for t in ts]))

# set path at initialization
sensor = magpy.Sensor(position=pos, orientation=ori)

# set path through properties
cube = magpy.magnet.Cuboid(magnetization=(0,0,1), dimension=(.5,.5,.5))
cube.position = pos + np.array((0,0,3))
cube.orientation = ori

magpy.show(sensor, cube, style_path_frames=10)
```

(examples-move-and-rotate)=
## Move and rotate

The attributes position and orientation can be either of “scalar” nature, i.e. a single position or a single rotation, or “vectors” when they are arrays of such scalars. The two attributes together define an object “path”.

The move and rotate methods obey the following rules:

- Scalar input is applied to the whole object path, starting with path index `start`. With the default `start='auto'` the index is set to `start=0` and the functionality is *moving objects around*.
- Vector input of length $n$ applies the individual $n$ operations to $n$ object path entries, starting with path index `start`. Padding applies when the input exceeds the existing path. With the default `start='auto'` the index is set to `start=len(object path)` and the functionality is *appending paths*.

```python
import magpylib as magpy

sensor = magpy.Sensor()

sensor.move((1,1,1))                                      # scalar input is by default applied
print(sensor.position)                                    # to the whole path
# out: [1. 1. 1.]

sensor.move([(1,1,1), (2,2,2)])                           # vector input is by default appended
print(sensor.position)                                    # to the existing path
# out: [[1. 1. 1.]  [2. 2. 2.]  [3. 3. 3.]]

sensor.move((1,1,1), start=1)                             # scalar input and start=1 is applied
print(sensor.position)                                    # to whole path starting at index 1
# out: [[1. 1. 1.]  [3. 3. 3.]  [4. 4. 4.]]

sensor.move([(0,0,10), (0,0,20)], start=1)                # vector input and start=1 merges
print(sensor.position)                                    # the input with the existing path
# out: [[ 1.  1.  1.]  [ 3.  3. 13.]  [ 4.  4. 24.]]      # starting at index 1.
```

(examples-relative-paths)=

## Relative paths

`move` and `rotate` input is interpreted relative to the existing path. Vector input is by default appended:

```{code-cell} ipython3
import numpy as np
from magpylib.magnet import Sphere

x_path = np.linspace((0,0,0), (10,0,0), 10)[1:]
z_path = np.linspace((0,0,0), (0,0,10), 10)[1:]

sphere = Sphere(magnetization=(0,0,1), diameter=3)

for _ in range(3):
    sphere.move(x_path).move(z_path)

sphere.show()
```

(examples-merging-paths)=

## Merging paths

Complex paths can be created by merging multiple path operations. This is done with vector input for the `move` and `rotate` methods, and choosing values for `start` that will make the paths overlap. In the following example we combine a linear path with a rotation about self (`anchor=None`) until path index 30. Thereon, a second rotation about the origin is applied, creating a spiral motion.

```{code-cell} ipython3
import numpy as np
from magpylib.magnet import Cuboid

cube =  Cuboid(magnetization=(0,0,100), dimension=(2,2,2))
cube.position = np.linspace((0,0,0), (10,0,0), 60)
cube.rotate_from_rotvec(np.linspace((0,0,0), (0,0,360), 30), start=0)
cube.rotate_from_rotvec(np.linspace((0,0,0), (0,0,360), 30), anchor=0, start=30)

cube.show(backend='plotly', animation=True)
```

## Reset path

The `reset_path()` method allows users to reset an object path to `position=(0,0,0)` and `orientation=None`.

```{code-cell} ipython3
import magpylib as magpy

# create sensor object with complex path
sensor=magpy.Sensor()
sensor.rotate_from_angax([1,2,3,4,5], (1,2,3), anchor=(0,3,5))

# reset path
sensor.reset_path()

print(sensor.position)
print(sensor.orientation.as_quat())
```

(examples-edge-padding-end-slicing)=

## Edge-padding and end-slicing

Magpylib will always make sure that object paths are in the right format, i.e. `position` and `orientation` attributes are of the same length. In addition, when objects with different path lengths are combined, e.g. when computing the field, the shorter paths are treated as static beyond their end to make the computation sensible. Internally, Magpylib follows a philosophy of edge-padding and end-slicing when adjusting paths.

The idea behind **edge-padding** is, that whenever path entries beyond the existing path length are needed the edge-entries of the existing path are returned. This means that the object is considered to be "static" beyond its existing path.

In the following example the orientation attribute is padded by its edge value `(0,0,.2)` as the position attribute length is increased.

```{code-cell} ipython3
from scipy.spatial.transform import Rotation as R
import magpylib as magpy

sensor = magpy.Sensor(
    position=[(0,0,0), (1,1,1)],
    orientation=R.from_rotvec([(0,0,.1), (0,0,.2)]),
)
sensor.position=[(i,i,i) for i in range(4)]
print(sensor.position)
print(sensor.orientation.as_rotvec())
```

When the field is computed of `loop1` with path length 4 and `loop2` with path length 2, `loop2` will remain in position (= edge padding) while the other object is still in motion.

```{code-cell} ipython3
from magpylib.current import Loop

loop1 = Loop(current=1, diameter=1, position=[(0,0,i) for i in range(4)])
loop2 = Loop(current=1, diameter=1, position=[(0,0,i) for i in range(2)])

B = magpy.getB([loop1,loop2], (0,0,0))
print(B)
```

The idea behind **end-slicing** is that, whenever a path is automatically reduced in length, Magpylib will slice to keep the ending of the path. While this occurs rarely, the following example shows how the `orientation` attribute is automatically end-sliced, keeping the values `[(0,0,.3), (0,0,.4)]`, when the `position` attribute is reduced in length:

```{code-cell} ipython3
from scipy.spatial.transform import Rotation as R
from magpylib import Sensor

sensor = Sensor(
    position=[(0,0,0), (1,1,1), (2,2,2), (3,3,3)],
    orientation=R.from_rotvec([(0,0,.1), (0,0,.2), (0,0,.3), (0,0,.4)]),
)
sensor.position=[(1,2,3), (2,3,4)]
print(sensor.position)
print(sensor.orientation.as_rotvec())
```

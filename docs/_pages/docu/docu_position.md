
(docu-position)=

# Position, Orientation and Paths

This section covers key elements of Magpylib. A tutorial {ref}`gallery-tutorial-paths` shows good practice examples.

## Position and Orientation

All Magpylib objects lie in a global Cartesian coordinate system. Their position and orientation is given by the attributes

* `position`: A point $(x,y,z)$ in the global coordinates, or a set of such points $(P_1, P_2, ...)$ given in units of mm. By default objects are created with `position=(0,0,0)`.
* `orientation`: A [Scipy Rotation object](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html) which describes the object rotation relative to its default orientation. The `Rotation` object can include a set of such rotations. For each class, the default orientation is described in {ref}`docu-classes`. By default objects are created with unit rotation `orientation=None`.

The `position` and `orientation` attributes can be either of scalar nature, i.e. a single position or a single rotation, or vectors when they are arrays of such scalars. The two attributes together define the "path" of an object. When the field of an object is computed, it is automatically computed for the whole path.

```{warning}
Paths should always be used when modeling multiple object positions to enable vectorized field computation. Avoid using Python loops for that purpose !
```

## Move and Rotate

Magpylib offers two powerful methods for object manipulation:

* `move(displacement, start)`: Move object by displacment input. `displacement` is a position vector (scalar input) or a set of position vectors (vector input).
* `rotate(rotation, anchor, start)`: Rotate object about an anchor point. `rotation` is a Scipy Rotation Object, and `anchor` is a position vector. Both inputs can be scalar or vector inputs.

Several deviations of the `rotate` method enable easier user input

* `rotate_from_angax`
* `rotate_from_rotvec`
* `rotate_from_euler`
* `rotate_from_quat`
* `rotate_from_matrix`
* `rotate_from_mrp`

The move and rotate methods obey the following rules:

- Scalar input is applied to the whole object path, starting with path index `start`. With the default `start='auto'` the index is set to `start=0` and the functionality is *moving objects around*.
- Vector input of length $n$ applies the individual $n$ operations to $n$ object path entries, starting with path index `start`. Padding applies when the input exceeds the existing path. With the default `start='auto'` the index is set to `start=len(object path)` and the functionality is *appending paths*.

This seemingly abstract behavior is best demonstrated by the following program

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

## Edge-padding and end-slicing

Magpylib will always make sure that object paths are in the right format, i.e. position and orientation attributes are of the same length. In addition, when objects with different path lengths are combined, e.g. when computing the field, the shorter paths are treated as static beyond their end to make the computation sensible. Internally, Magpylib follows a philosophy of edge-padding and end-slicing when adjusting paths.

* The idea behind edge-padding is, that whenever path entries beyond the existing path length are needed the edge-entries of the existing path are returned. This means that the object is considered to be “static” beyond its existing path.

* The idea behind end-slicing is that, whenever a path is automatically reduced in length, Magpylib will slice to keep the ending of the path.





<!-- 

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


Notice that when one of the `position` and `orientation` attributes are modified in length, the other is automatically adjusted to the same length. A detailed outline of the functionality of `position`, `orientation`, `move`, `rotate` and paths is given in examples `examples-paths`. -->



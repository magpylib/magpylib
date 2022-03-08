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

(examples-paths)=

# Paths

(examples-assign-absolute-path)=
## Assigning absolute paths

Absolute object paths are assigned at initialization or with the getter and setter methods.:

```{code-cell} ipython3
import numpy as np
from scipy.spatial.transform import Rotation as R
import magpylib as magpy

ts = np.linspace(0, 10, 51)
pos = np.array([(t, 0, np.sin(t)) for t in ts])
ori = R.from_rotvec(np.array([(0, -np.cos(t)*0.785, 0) for t in ts]))

# set path at initialization
sens1 = magpy.Sensor(position=pos, orientation=ori)

# set path through setter methods
sens2 = magpy.Sensor()
sens2.position = pos + np.array((0,0,3))
sens2.orientation = ori

magpy.show(sens1, sens2, style_path_frames=10)
```

(examples-move-and-rotate)=
## Move and rotate

The attributes position and orientation can be either of “scalar” nature, i.e. a single position or a single rotation, or “vectors” when they are arrays of such scalars. The two attributes together define an object “path”.

The move and rotate methods obey the following rules:

- Scalar input is applied to the whole object path, starting with path index `start`. With the default `start='auto'` the index is set to `start=0` and the functionality is *moving objects around*.
- Vector input of length $n$ applies the individual $n$ operations to $n$ object path entries, starting with path index `start`. Padding applies when the input exceeds the existing path. With the default `start='auto'` the index is set to `start=len(object path)` and the functionality is *appending paths*.

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

(examples-relative-paths)=

## Relative paths

`move` and `rotate` input is interpreted relative to the existing path. Vector input is by default appended to the existing path. The combination leads to the following behavior,

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

pos1 = np.linspace((0,0,0), (10,0,0), 10)[1:]
pos2 = np.linspace((0,0,0), (0,0,10), 10)[1:]

sens = magpy.Sensor()

for _ in range(3):
    sens.move(pos1).move(pos2)

magpy.show(sens)
```

(examples-merging-paths)=

## Merging paths

Complex paths can be created by merging multiple path operations using vector input for the `move` and `rotate` methods and choosing values for `start` that will make the paths overlap. In the following example we combine a linear path with a rotation about self (`anchor=None`) until path index 30. Thereon, a second rotation is applied about the origin, which creates a spiral motion.

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

src =  magpy.magnet.Cuboid(magnetization=(0,0,100), dimension=(2,2,2))
src.position = np.linspace((0,0,0), (10,0,0), 60)
src.rotate_from_rotvec(np.linspace((0,0,0), (0,0,360), 30), start=0)
src.rotate_from_rotvec(np.linspace((0,0,0), (0,0,360), 30), anchor=0, start=30)

src.show(backend='plotly', animation=True)
```

## Reset path

The `reset_path()` method allows users to reset an object path to `position=(0,0,0)` and `orientation=None`.

```{code-cell} ipython3
import magpylib as magpy

# create sensor object with complex path
sens=magpy.Sensor()
sens.rotate_from_angax([1,2,3,4,5], (1,2,3), anchor=(0,3,5))

# reset path
sens.reset_path()

print(sens.position)
print(sens.orientation.as_quat())
```

(examples-edge-padding-end-slicing)=

## Edge-padding and end-slicing

Magpylib will always make sure that object paths are in the right format, i.e. `position` and `orientation` attributes are of the same length. In addition, when objects with different path lengths are combined, e.g. when computing the field, the shorter paths are adjusted in length to make the computation sensible. Internally, Magpylib follows a philosohpy of edge-padding and end-slicing when adjusting paths.

The idea behind **edge-padding** is, that whenever path entries beyond the existing path length are needed the edge-entries of the existing path are returned. This means that the object is considered to be "static" beyond its existing path.

In the following example the orientation attribute is padded by its edge value `(0,0,.2)` as the position attribute length is increased.

```{code-cell} ipython3
from scipy.spatial.transform import Rotation as R
import magpylib as magpy

sens = magpy.Sensor(
    position=[(0,0,0), (1,1,1)],
    orientation=R.from_rotvec([(0,0,.1), (0,0,.2)]),
)
sens.position=[(i,i,i) for i in range(4)]
print(sens.position)
print(sens.orientation.as_rotvec())
```

When the field is computed of `src1` with path length 4 and `src2` with path length 2, `src2` will remain in position (= edge padding) while the other object is still in motion.

```{code-cell} ipython3
import magpylib as magpy

src1 = magpy.current.Loop(current=1, diameter=1, position=[(0,0,i) for i in range(4)])
src2 = magpy.current.Loop(current=1, diameter=1, position=[(0,0,i) for i in range(2)])

print(magpy.getB([src1,src2], (0,0,0)))
```

The idea behind **end-slicing** is that, whenever a path is automatically reduced in length, Magplyib will slice to keep the ending of the path. While this occurs rarely, the following example shows how the `orientation` attribute is automatically end-sliced, keeping the values `[(0,0,.3), (0,0,.4)]`, when the `position` attribute is reduced in length:

```{code-cell} ipython3
from scipy.spatial.transform import Rotation as R
import magpylib as magpy

sens = magpy.Sensor(
    position=[(0,0,0), (1,1,1), (2,2,2), (3,3,3)],
    orientation=R.from_rotvec([(0,0,.1), (0,0,.2), (0,0,.3), (0,0,.4)]),
)
sens.position=[(1,2,3), (2,3,4)]
print(sens.position)
print(sens.orientation.as_rotvec())
```

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

## Assigning an object path

When the user already knows the **absolute path** of an object in the form of a position array (shape (n,3)) and a scipy `Rotation` object (len n) he can assign this path to an object either at initialization, or by using the setter and getter methods:

```{code-cell} ipython3
import numpy as np
from scipy.spatial.transform import Rotation as R
import magpylib as magpy

ts = np.linspace(0, 10, 51)
pos = np.array([(t, 0, np.sin(t)) for t in ts])
ori = R.from_rotvec(np.array([(0, -np.cos(t), 0) for t in ts]))

# set path at initialization
sens1 = magpy.Sensor(position=pos, orientation=ori)

# set path through setter methods
sens2 = magpy.Sensor()
sens2.position = pos + np.array((0,0,3))
sens2.orientation = ori

magpy.show(sens1, sens2, style_path_frames=10)
```

When the user wants apply a **relative path** with respect to the existing object path he should make use of the `move` and `rotate` methods. They are specifically useful when appending relative positions to an existing path:

```{code-cell} ipython3
import numpy as np
from scipy.spatial.transform import Rotation as R
import magpylib as magpy

pos1 = np.linspace((0,0,0), (10,0,0), 10)[1:]
pos2 = np.linspace((0,0,0), (0,0,10), 10)[1:]

# set path at initialization
sens = magpy.Sensor(position=pos1)

# append relative paths to the existing one
sens.move(pos2)
sens.move(pos1)

magpy.show(sens)
```

## Merging paths

Complex paths can be created by merging multiple path operations using vector input for the `move` and `rotate` methods and choosing values for `start` that will make the paths overlap. In the following example we combine a linear path with a rotation after path index 20, to create a spiral path:

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

src =  magpy.magnet.Cuboid(magnetization=(0,0,100), dimension=(2,2,2))
src.move(np.linspace((0,0,0), (10,0,0), 60)[1:])
src.rotate_from_rotvec(np.linspace((0,0,0), (0,0,720), 60), anchor=0, start=20)

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

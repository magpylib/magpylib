---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
orphan: true
---

(examples-tutorial-paths)=

# Working with Paths

The position and orientation attributes are key elements of Magpylib. The documentation section {ref}`docs-position` describes how they work in detail. Wile these definitions can seem abstract, the interface was constructed as intuitively as possible which is demonstrated in this tutorial.

```{important}
Always make use of paths when computing with multiple Magpylib object position and orientation instances. This enables vectorized computation. Avoid Python loops at all costs!
```

In this tutorial we show some good practice examples.

## Assigning Absolute Paths

Absolute object paths are assigned at initialization or through the object properties.

```{code-cell} ipython3
import numpy as np
from scipy.spatial.transform import Rotation as R
import magpylib as magpy

# Create paths
ts = np.linspace(0, 10, 31)
pos = np.array([(0.1 * t, 0, 0.1 * np.sin(t)) for t in ts])
ori = R.from_rotvec(np.array([(0, -0.1 * np.cos(t) * 0.785, 0) for t in ts]))

# Set path at initialization
sensor = magpy.Sensor(position=pos, orientation=ori)

# Set path through properties
cube = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(0.01, 0.01, 0.01))
cube.position = pos + np.array((0, 0, 0.3))
cube.orientation = ori

# Display as animation
magpy.show(sensor, cube, animation=True, backend="plotly")
```

## Relative Paths

`move` and `rotate` input is interpreted relative to the existing path. When the input is scalar the whole existing path is moved.

```{code-cell} ipython3
import numpy as np
from scipy.spatial.transform import Rotation as R
import magpylib as magpy

# Create paths
ts = np.linspace(0, 10, 21)
pos = np.array([(0.1 * t, 0, 0.1 * np.sin(t)) for t in ts])
ori = R.from_rotvec(np.array([(0, -0.1 * np.cos(t) * 0.785, 0) for t in ts]))

# Set path at initialization
sens1 = magpy.Sensor(position=pos, orientation=ori, style_label="sens1")

# Apply move operation to whole path with scalar input
sens2 = sens1.copy(style_label="sens2")
sens2.move((0, 0, 0.05))

# Apply rotate operation to whole path with scalar input
sens3 = sens1.copy(style_label="sens3")
sens3.rotate_from_angax(angle=90, axis="y", anchor=0)

# Display paths
magpy.show(sens1, sens2, sens3)
```

When the input is a vector, the path is by default appended.

```{code-cell} ipython3
import numpy as np
from magpylib.magnet import Sphere

# Create paths
x_path = np.linspace((0, 0, 0), (0.1, 0, 0), 10)[1:]
z_path = np.linspace((0, 0, 0), (0, 0, 0.1), 10)[1:]

# Create sphere object
sphere = Sphere(polarization=(0, 0, 1), diameter=0.03)

# Apply paths subsequently
for _ in range(3):
    sphere.move(x_path).move(z_path)

# Display paths
sphere.show()
```

## Merging paths

Complex paths can be created by merging multiple path operations. This is done with vector input for the `move` and `rotate` methods and choosing values for `start` that will make the paths overlap. In the following example we combine a linear path with a rotation about self (`anchor=None`) until path index 30. Thereon, a second rotation about the origin is applied, creating a spiral.

```{code-cell} ipython3
import numpy as np
from magpylib.magnet import Cuboid

# Create cube and set linear path
cube = Cuboid(polarization=(0, 0, 0.1), dimension=(0.02, 0.02, 0.02))
cube.position = np.linspace((0, 0, 0), (0.1, 0, 0), 60)

# Apply rotation about self - starting at index 0
cube.rotate_from_rotvec(np.linspace((0, 0, 0), (0, 0, 360), 30), start=0)

# Apply rotation about origin - starting at index 30
cube.rotate_from_rotvec(np.linspace((0, 0, 0), (0, 0, 360), 30), anchor=0, start=30)

# Display paths as animation
cube.show(backend="plotly", animation=True)
```

## Reset path

The `reset_path()` method allows users to reset an object path to `position=(0,0,0)` and `orientation=None`.

```{code-cell} ipython3
import magpylib as magpy

# Create sensor object with complex path
sensor = magpy.Sensor().rotate_from_angax(
    [1, 2, 3, 4], (1, 2, 3), anchor=(0, 0.03, 0.05)
)

# Reset path
sensor.reset_path()

print(sensor.position)
print(sensor.orientation.as_quat())
```

(examples-tutorial-paths-edge-padding-end-slicing)=
## Edge-padding and end-slicing

Magpylib will always make sure that object paths are in the right format, i.e., `position` and `orientation` attributes are of the same length. In addition, when objects with different path lengths are combined, e.g., when computing the field, the shorter paths are treated as static beyond their end to make the computation sensible. Internally, Magpylib follows a philosophy of edge-padding and end-slicing when adjusting paths.

The idea behind **edge-padding** is that, whenever path entries beyond the existing path length are needed, the edge-entries of the existing path are returned. This means that the object is static beyond its existing path.

In the following example the orientation attribute is padded by its edge value `(0,0,.2)` as the position attribute length is increased.

```{code-cell} ipython3
from scipy.spatial.transform import Rotation as R
import magpylib as magpy

sensor = magpy.Sensor(
    position=[(0, 0, 0), (0.01, 0.01, 0.01)],
    orientation=R.from_rotvec([(0, 0, 0.1), (0, 0, 0.2)]),
)
sensor.position = [(i / 100, i / 100, i / 100) for i in range(4)]
print(sensor.position)
print(sensor.orientation.as_rotvec())
```

When the field is computed of `loop1` with path length 4 and `loop2` with path length 2, `loop2` will remain in position (= edge padding) while the other object is still in motion.

```{code-cell} ipython3
from magpylib.current import Circle

loop1 = Circle(current=1, diameter=1, position=[(0, 0, i) for i in range(4)])
loop2 = Circle(current=1, diameter=1, position=[(0, 0, i) for i in range(2)])

B = magpy.getB([loop1, loop2], (0, 0, 0))
print(B)
```

The idea behind **end-slicing** is that, whenever a path is automatically reduced in length, Magpylib will slice to keep the ending of the path. While this occurs rarely, the following example shows how the `orientation` attribute is automatically end-sliced, keeping the values `[(0,0,.3), (0,0,.4)]`, when the `position` attribute is reduced in length:

```{code-cell} ipython3
from scipy.spatial.transform import Rotation as R
from magpylib import Sensor

sensor = Sensor(
    position=[(0, 0, 0), (0.01, 0.01, 0.01), (0.02, 0.02, 2), (0.03, 0.03, 0.03)],
    orientation=R.from_rotvec([(0, 0, 0.1), (0, 0, 0.2), (0, 0, 0.3), (0, 0, 0.4)]),
)
sensor.position = [(0.01, 0.02, 0.03), (0.02, 0.03, 0.04)]
print(sensor.position)
print(sensor.orientation.as_rotvec())
```

```{code-cell} ipython3

```

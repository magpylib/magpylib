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

(examples-tutorial-field-computation)=

# Computing the Field

## Most basic Example

The v2 slogan was *"The magnetic field is only three lines of code away"*, which is demonstrated by the following most fundamental and self-explanatory example,

```{code-cell} ipython3
import magpylib as magpy
loop = magpy.current.Circle(current=1, diameter=1)
B = loop.getB((0, 0, 0))

print(B)
```

## Field on a Grid

There are four field computation functions: `getB` will compute the B-field in T. `getH` computes the H-field in A/m. `getJ` computes the magnetic polarization in units of T. `getM` computes the magnetization in units of A/m.

All these functions will return the field in the shape of the input. In the following example, BHJM-fields of a diametrically magnetized cylinder magnet are computed on a position grid in the symmetry plane and are then displayed using Matplotlib.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
import magpylib as magpy

fig, [[ax1,ax2], [ax3,ax4]] = plt.subplots(2, 2, figsize=(10, 10))

# Create an observer grid in the xy-symmetry plane
grid = np.mgrid[-50:50:100j, -50:50:100j, 0:0:1j].T[0]
X, Y, _ = np.moveaxis(grid, 2, 0)

# Compute BHJM-fields of a cylinder magnet on the grid
cyl = magpy.magnet.Cylinder(polarization=(0.5, 0.5, 0), dimension=(40, 20))

B = cyl.getB(grid)
Bx, By, _ = np.moveaxis(B, 2, 0)

H = cyl.getH(grid)
Hx, Hy, _ = np.moveaxis(H, 2, 0)

J = cyl.getJ(grid)
Jx, Jy, _ = np.moveaxis(J, 2, 0)

M = cyl.getM(grid)
Mx, My, _ = np.moveaxis(M, 2, 0)

# Display field with Pyplot
ax1.streamplot(X, Y, Bx, By, color=np.log(norm(B, axis=2)), cmap="spring_r")
ax2.streamplot(X, Y, Hx, Hy, color=np.log(norm(H, axis=2)), cmap="winter_r")
ax3.streamplot(X, Y, Jx, Jy, color=norm(J, axis=2), cmap="summer_r")
ax4.streamplot(X, Y, Mx, My, color=norm(M, axis=2), cmap="autumn_r")

ax1.set_title("B-Field")
ax2.set_title("H-Field")
ax3.set_title("J-Field")
ax4.set_title("M-Field")

for ax in [ax1,ax2,ax3,ax4]:
    ax.set(
        xlabel="x-position",
        ylabel="y-position",
        aspect=1,
        xlim=(-50,50),
        ylim=(-50,50),
    )
    # Outline magnet boundary
    ts = np.linspace(0, 2 * np.pi, 50)
    ax.plot(20 * np.sin(ts), 20 * np.cos(ts), "k--")

plt.tight_layout()
plt.show()
```

(examples-tutorial-field-computation-sensors)=

## Using Sensors

The `Sensor` class enables relative positioning of observer grids in the global coordinate system. The observer grid is stored in the `pixel` parameter of the sensor object which is `(0,0,0)` by default (sensor position = observer position).

The following example shows a moving and rotating sensor with two pixels. At the same time, the source objects are moving to demonstrate the versatility of the field computation.

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

# Reset defaults set in previous example
magpy.defaults.reset()


# Define sensor with path
sensor = magpy.Sensor(pixel=[(0, 0, -0.0005), (0, 0, 0.0005)], style_size=1.5)
sensor.position = np.linspace((0, 0, -0.003), (0, 0, 0.003), 37)

angles = np.linspace(0, 360, 37)
sensor.rotate_from_angax(angles, "z", start=0)

# Define source with path
cyl1 = magpy.magnet.Cylinder(
    polarization=(0.1, 0, 0), dimension=(0.001, 0.002), position=(0.003, 0, 0)
)
cyl2 = cyl1.copy(position=(-0.003, 0, 0))
coll = magpy.Collection(cyl1, cyl2)
coll.rotate_from_angax(-angles, "z", start=0)

# Display system and field at sensor
with magpy.show_context(sensor, coll, animation=True, backend="plotly"):
    magpy.show(col=1)
    magpy.show(output="Bx", col=2, pixel_agg=None)
```

## Multiple Inputs

When `getBHJM` receive multiple inputs for sources and observers they will compute all possible combinations. It is still beneficial to call the field computation only a single time, because similar sources will be grouped, and the computation will be vectorized automatically.

```{code-cell} ipython3
import magpylib as magpy

# Three sources
cube1 = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(0.1, 0.1, 0.1))
cube2 = cube1.copy()
cube3 = cube1.copy()

# Two sensors with 4x5 pixel each
pixel = [[[(i / 1000, j / 1000, 0)] for i in range(4)] for j in range(5)]
sens1 = magpy.Sensor(pixel=pixel)
sens2 = sens1.copy()

# Compute field
B = magpy.getB([cube1, cube2, cube3], [sens1, sens2])

# The result includes all combinations
B.shape
```

Select the second cube (first index), the first sensor (second index), pixel 3-4 (index three and four) and the Bz-component of the field (index five)

```{code-cell} ipython3
# Continuation from above - ensure previous code is executed

B[1, 0, 2, 3, 2]
```

A path will add another index. Every higher pixel dimension will add another index as well.

## Field as Pandas Dataframe

Instead of a NumPy `ndarray`, the field computation can also return a [pandas](https://pandas.pydata.org/).[dataframe](https://pandas.pydata.org/docs/user_guide/dsintro.html#dataframe) using the `output='dataframe'` kwarg.

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

cube = magpy.magnet.Cuboid(
    polarization=(0, 0, 1), dimension=(0.01, 0.01, 0.01), style_label="cube"
)
loop = magpy.current.Circle(
    current=200,
    diameter=0.02,
    style_label="loop",
)
sens1 = magpy.Sensor(
    pixel=[(0, 0, 0), (0.005, 0, 0)],
    position=np.linspace((-0.04, 0, 0.02), (0.04, 0, 0.02), 30),
    style_label="sens1",
)
sens2 = sens1.copy(style_label="sens2").move((0, 0, 0.01))

B = magpy.getB(
    [cube, loop],
    [sens1, sens2],
    output="dataframe",
)

B
```

Plotting libraries such as [plotly](https://plotly.com/python/plotly-express/) or [seaborn](https://seaborn.pydata.org/introduction.html) can take advantage of this feature, as they can deal with `dataframes` directly.

```{code-cell} ipython3
# Continuation from above - ensure previous code is executed

import plotly.express as px

fig = px.line(
    B,
    x="path",
    y="Bx",
    color="pixel",
    line_group="source",
    facet_col="source",
    symbol="sensor",
)
fig.show()
```

(examples-tutorial-field-computation-functional-interface)=

## Functional Interface

All above computations demonstrate the convenient object-oriented interface of Magpylib. However, there are instances when it is better to work with the functional interface instead.

1. Reduce overhead of Python objects
2. Complex computation instances

In the following example we show how complex instances are computed using the functional interface.

```{important}
The functional interface will only outperform the object oriented interface if you use NumPy operations for input array creation, such as `tile`, `repeat`, `reshape`, ... !
```

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

# Two different magnet dimensions
dim1 = (0.02, 0.04, 0.04)
dim2 = (0.04, 0.02, 0.02)
DIM = np.vstack(
    (
        np.tile(dim1, (6, 1)),
        np.tile(dim2, (6, 1)),
    )
)

# Sweep through different polarizations for each magnet type
pol = np.linspace((0, 0, 0.5), (0, 0, 1), 6)
POL = np.tile(pol, (2, 1))

# Airgap must stay the same
pos1 = (0, 0, 0.03)
pos2 = (0, 0, 0.02)
POS = np.vstack(
    (
        np.tile(pos1, (6, 1)),
        np.tile(pos2, (6, 1)),
    )
)

# Compute all instances with the functional interface
B = magpy.getB(
    sources="Cuboid",
    observers=POS,
    polarization=POL,
    dimension=DIM,
)

B.round(decimals=2)
```

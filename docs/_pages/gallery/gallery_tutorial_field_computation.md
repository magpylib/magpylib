---
orphan: true
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(gallery-tutorial-field-computation)=

# Computing the Field

## Most basic Example

The v2 slogan was *"The magnetic field is only three lines of code away"*, which is demonstrated by the most fundamental use of Magpylib.

```{code-cell} ipython3
import magpylib as magpy                          # Import Magpylib
loop = magpy.current.Circle(current=1, diameter=1)  # Create magnetic source
B = magpy.getB(loop, observers=(0,0,0))           # Compute field

print(B.round(decimals=3))
```

## Field on a Grid

When handed multiple observer positions, `getB` and `getH` will return the field in the shape of the input. In the following example, B- and H-field of a diametrically magnetized cylinder magnet are computed on a position grid in the symmetry plane, and are then displayed using Matplotlib.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy

fig, [ax1,ax2] = plt.subplots(1, 2, figsize=(10,5))

# Create an observer grid in the xz-symmetry plane
X, Y = np.mgrid[-5:5:100j, -5:5:100j].transpose((0, 2, 1))
grid = np.stack([X, Y, np.zeros((100, 100))], axis=2)

# Compute B- and H-fields of a cylinder magnet on the grid
cyl = magpy.magnet.Cylinder(magnetization=(500,500,0), dimension=(4,2))
B = cyl.getB(grid)
H = cyl.getH(grid)

# Display field with Pyplot
ax1.streamplot(grid[:,:,0], grid[:,:,1], B[:,:,0], B[:,:,1], density=1.5,
    color=np.log(np.linalg.norm(B, axis=2)), linewidth=1, cmap='spring_r')

ax2.streamplot(grid[:,:,0], grid[:,:,1], H[:,:,0], H[:,:,1], density=1.5,
    color=np.log(np.linalg.norm(B, axis=2)), linewidth=1, cmap='winter_r')

# Outline magnet boundary
for ax in [ax1,ax2]:
    ts = np.linspace(0, 2*np.pi, 50)
    ax.plot(2*np.sin(ts), 2*np.cos(ts), 'k--')

plt.tight_layout()
plt.show()
```

(gallery-tutorial-field-computation-sensors)=

## Using Sensors

The `Sensor` class enables relative positioning of observer grids in the global coordinate system. The observer grid is stored in the `pixel` parameter of the sensor object which is `(0,0,0)` by default (sensor position = observer position).

The following example shows a moving and rotating sensor with two pixels. At the same time, the source objects are moving to demonstrate the versatility of the field computation.

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

# Reset defaults set in previous example
magpy.defaults.reset()


# Define sensor with path
sensor = magpy.Sensor(pixel=[(0,0,-.5), (0,0,.5)], style_size=1.5)
sensor.position = np.linspace((0,0,-3), (0,0,3), 37)

angles = np.linspace(0, 360, 37)
sensor.rotate_from_angax(angles, 'z', start=0)

# Define source with path
cyl1 = magpy.magnet.Cylinder(magnetization=(100,0,0), dimension=(1,2), position=(3,0,0))
cyl2 = cyl1.copy(position=(-3,0,0))
coll = magpy.Collection(cyl1, cyl2)
coll.rotate_from_angax(-angles, 'z', start=0)

# Display system and field at sensor
with magpy.show_context(sensor, coll, animation=True, backend='plotly'):
    magpy.show(col=1)
    magpy.show(output='Bx', col=2, pixel_agg=None)
```

## Multiple Inputs

When `getB` and `getH` receive multiple inputs for sources and observers they will compute all possible combinations. It is still beneficial to call the field computation only a single time, because similar sources will be grouped and the computation will be vectorized automatically.

```{code-cell} ipython3
import magpylib as magpy

# Three sources
cube1 = magpy.magnet.Cuboid(magnetization=(0,0,1000), dimension=(1,1,1))
cube2 = cube1.copy()
cube3 = cube1.copy()

# Two sensors with 4x5 pixel each
pixel = [[[(i,j,0)] for i in range(4)] for j in range(5)]
sens1 = magpy.Sensor(pixel=pixel)
sens2 = sens1.copy()

# Compute field
B = magpy.getB([cube1, cube2, cube3], [sens1, sens2])

# The result includes all combinations
B.shape
```

Select the second cube (first index), the first sensor (second index), pixel 3-4 (index three and four) and the Bz-component of the field (index five)

```{code-cell} ipython3
B[1, 0, 2, 3, 2]
```

A path will add another index. Every higher pixel dimension will add another index as well.

## Field as Pandas Dataframe

Instead of a Numpy `ndarray`, the field computation can also return a [pandas](https://pandas.pydata.org/).[dataframe](https://pandas.pydata.org/docs/user_guide/dsintro.html#dataframe) using the `output='dataframe'` kwarg.

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

cube = magpy.magnet.Cuboid(
    magnetization=(0, 0, 1000),
    dimension=(1, 1, 1),
    style_label='cube'
)
loop = magpy.current.Circle(
    current=200,
    diameter=2,
    style_label='loop',
)
sens1 = magpy.Sensor(
    pixel=[(0,0,0), (.5,0,0)],
    position=np.linspace((-4, 0, 2), (4, 0, 2), 30),
    style_label='sens1'
)
sens2 = sens1.copy(style_label='sens2').move((0,0,1))

B = magpy.getB(
    [cube, loop],
    [sens1, sens2],
    output='dataframe',
)

B
```

Plotting libraries such as [plotly](https://plotly.com/python/plotly-express/) or [seaborn](https://seaborn.pydata.org/introduction.html) can take advantage of this feature, as they can deal with `dataframes` directly.

```{code-cell} ipython3
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

(gallery-tutorial-field-computation-direct-interface)=

## Direct Interface

All above computations demonstrate the convenient object oriented interface of Magpylib. However, there are instances when it is better to work with the direct interface instead.

1. Reduce overhead of Python objects
2. Complex computation instances

In the following example we show how complex instances are computed using the direct interface.

```{important}
Use numpy operations for input array creation as shown in the example !
```

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

# Two different magnet dimensions
dim1 = (2,4,4)
dim2 = (4,2,2)
DIM = np.vstack((
    np.tile(dim1, (6,1)),
    np.tile(dim2, (6,1)),
    ))

# Sweep through different magnetization for each magnet type
mags = np.linspace((0,0,500), (0,0,1000), 6)
MAG = np.tile(mags, (2,1))

# Airgap must stay the same
pos1 = (0,0,3)
pos2 = (0,0,2)
POS = np.vstack((
    np.tile(pos1, (6,1)),
    np.tile(pos2, (6,1)),
    ))

# Compute all instances with the direct interface
B = magpy.getB(
    sources='Cuboid',
    observers=POS,
    magnetization=MAG,
    dimension=DIM,
)

B.round(decimals=2)
```

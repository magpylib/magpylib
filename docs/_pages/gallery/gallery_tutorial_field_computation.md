---
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

# Field computation - How to

**Example 1:** As expressed by the old v2 slogan *"The magnetic field is only three lines of code away"*, this example demonstrates the most fundamental field computation:

```{code-cell} ipython3
import magpylib as magpy
loop = magpy.current.Loop(current=1, diameter=2)
B = magpy.getB(loop, (1,2,3))
print(B)
```

**Example 2:** When handed with multiple observer positions, `getB` and `getH` will return the field in the shape of the observer input. In the following example, B- and H-field of a cuboid magnet are computed on a position grid, and then displayed using Matplotlib:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy

fig, [ax1,ax2] = plt.subplots(1, 2, figsize=(10,5))

# create an observer grid in the xz-symmetry plane
ts = np.linspace(-3, 3, 30)
grid = np.array([[(x,0,z) for x in ts] for z in ts])

# compute B- and H-fields of a cuboid magnet on the grid
cube = magpy.magnet.Cuboid(magnetization=(500,0,500), dimension=(2,2,2))
B = cube.getB(grid)
H = cube.getH(grid)

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

**Example 3:** The following example code shows how the field in a position system is computed with a sensor object. Both, magnet and sensor are moving. The 3D system and the field along the path are displayed with Plotly:

```{code-cell} ipython3
import numpy as np
import plotly.graph_objects as go
import magpylib as magpy

# reset defaults set in previous example
magpy.defaults.reset()

# setup plotly figure and subplots
fig = go.Figure().set_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]])

# define sensor and source
sensor = magpy.Sensor(pixel=[(0,0,-.2), (0,0,.2)], style_size=1.5)
magnet = magpy.magnet.Cylinder(magnetization=(100,0,0), dimension=(1,2))

# define paths
sensor.position = np.linspace((0,0,-3), (0,0,3), 40)
magnet.position = (4,0,0)
magnet.rotate_from_angax(angle=np.linspace(0, 300, 40)[1:], axis='z', anchor=0)

# display system in 3D
temp_fig = go.Figure()
magpy.show(magnet, sensor, canvas=temp_fig, backend='plotly')
fig.add_traces(temp_fig.data, rows=1, cols=1)

# compute field and plot
B = magpy.getB(magnet, sensor)
for i,plab in enumerate(['pixel1', 'pixel2']):
    for j,lab in enumerate(['_Bx', '_By', '_Bz']):
        fig.add_trace(go.Scatter(x=np.arange(40), y=B[:,i,j], name=plab+lab))

fig.show()
```


**Example 4:** The last example demonstrates the most general form of a `getB` computation with multiple source and sensor inputs. Specifically, 3 sources, one with path length 11, and two sensors, each with pixel shape (4,5). Note that, when input objects have different path lengths, objects with shorter paths are treated as static beyond their path end.

```{code-cell} ipython3
import magpylib as magpy

# 3 sources, one with length 11 path
pos_path = [(i,0,1) for i in range(-1,1)]
source1 = magpy.misc.Dipole(moment=(0,0,100), position=pos_path)
source2 = magpy.current.Loop(current=10, diameter=3)
source3 = source1 + source2

# 2 observers, each with 4x5 pixel
pixel = [[[(i,j,0)] for i in range(4)] for j in range(5)]
sensor1 = magpy.Sensor(pixel=pixel, position=(-1,0,-1))
sensor2 = sensor1.copy().move((2,0,0))

sources = [source1, source2, source3]
sensors = [sensor1, sensor2]
# compute field
B = magpy.getB(sources, sensors)
print(B.shape)
```


Instead of a Numpy `ndarray`, the field computation can also return a [pandas](https://pandas.pydata.org/).[dataframe](https://pandas.pydata.org/docs/user_guide/dsintro.html#dataframe) using the `output='dataframe'` kwarg.

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

cube = magpy.magnet.Cuboid(
    magnetization=(0, 0, 1000),
    dimension=(1, 1, 1),
    style_label='cube'
)
loop = magpy.current.Loop(
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

B_as_df = magpy.getB(
    [cube, loop],
    [sens1, sens2],
    output='dataframe',
)

B_as_df
```


Plotting libraries such as [plotly](https://plotly.com/python/plotly-express/) or [seaborn](https://seaborn.pydata.org/introduction.html) can take advantage of this feature, as they can deal with `dataframes` directly.

```{code-cell} ipython3
import plotly.express as px
fig = px.line(
    B_as_df,
    x="path",
    y="Bx",
    color="pixel",
    line_group="source",
    facet_col="source",
    symbol="sensor",
)
fig.show()
```

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

# Display backends

+++

For easy inspection of sources, sensors or collection thereof, the Magpylib package has default graphical representations for every magnet, current and sensor of the library. Objects can be displayed via the `magpylib.show` function or directly via the `show` method available for all objects (see examples below).

Additionally, objects can be rendered via different plottting backends libraries, which to date includes:

- matplotlib (by default)
- plotly

+++

Note that the default plotting backend can be changed to plotly by setting it a the top of script:
```python
import magpylib as magpy
magpy.defaults.display.backend = 'plotly'
```

All the following calls to the `show` function or method without specifying a backend will call the `'plotly'` plotting backend. If you explicitly specify `'matplotlib'` at `show` call, it will locally override the set defaults.

+++

## Display multiple objects:

```{code-cell} ipython3
import magpylib as magpy

magnet = magpy.magnet.Cuboid(magnetization=(1, 0, 0), dimension=(1, 2, 3))
sens = magpy.Sensor(position=(0, 0, 3))
magpy.show(magnet, sens, zoom=1)
```

## Display multiple objects with paths

```{code-cell} ipython3
import magpylib as magpy
import matplotlib.pyplot as plt
import numpy as np

# define sources
src1 = magpy.magnet.Sphere(magnetization=(0, 0, 1), diameter=1)
src2 = magpy.magnet.Cylinder(magnetization=(0, 0, 1), dimension=(1, 2))

# manipulate first source to create a path
src1.move(np.linspace((0, 0, 0.1), (0, 0, 8), 20))

# manipulate second source
src2.move(np.linspace((0.1, 0, 0.1), (5, 0, 5), 50))
src2.rotate_from_angax(angle=np.linspace(10, 600, 50), axis="z", anchor=0, start=1)

# display the system
magpy.show(src1, src2)
```

+++ {"tags": []}

## Display objects with a different plotting backend

```{code-cell} ipython3
import magpylib as magpy
import matplotlib.pyplot as plt
import numpy as np

# define sources
src1 = magpy.magnet.Sphere(magnetization=(0, 0, 1), diameter=1)
src2 = magpy.magnet.Cylinder(magnetization=(0, 0, 1), dimension=(1, 2))

# manipulate first source to create a path
src1.move(np.linspace((0, 0, 0.1), (0, 0, 8), 20))

# manipulate second source
src2.move(np.linspace((0.1, 0, 0.1), (5, 0, 5), 50))
src2.rotate_from_angax(angle=np.linspace(10, 600, 50), axis="z", anchor=0, start=1)

# display the system
magpy.show(src1, src2, backend="plotly")
```

## Display figure on your own canvas

### With a matplotlib canvas

```{code-cell} ipython3
import magpylib as magpy
import matplotlib.pyplot as plt
import numpy as np

# define sources
src1 = magpy.magnet.Sphere(magnetization=(0, 0, 1), diameter=1)
src2 = magpy.magnet.Cylinder(magnetization=(0, 0, 1), dimension=(1, 2))

# manipulate first source to create a path
src1.move(np.linspace((0, 0, 0.1), (0, 0, 8), 20))

# manipulate second source
src2.move(np.linspace((0.1, 0, 0.1), (5, 0, 5), 50))
src2.rotate_from_angax(angle=np.linspace(10, 600, 50), axis="z", anchor=0, start=1)

# setup matplotlib figure and subplots
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(131, projection="3d")  # 3D-axis
ax2 = fig.add_subplot(132, projection="3d")  # 3D-axis
ax3 = fig.add_subplot(133, projection="3d")  # 3D-axis

# draw the objects
magpy.show(src1, canvas=ax1)
magpy.show(src2, canvas=ax2)
magpy.show(src1, src2, canvas=ax3)

# display the system
plt.tight_layout()
plt.show()
```

### With a plotly canvas

```{code-cell} ipython3
import magpylib as magpy
import numpy as np
import plotly.graph_objects as go

# define sources
src1 = magpy.magnet.Sphere(magnetization=(0, 0, 1), diameter=1)
src2 = magpy.magnet.Cylinder(magnetization=(0, 0, 1), dimension=(1, 2))

# manipulate first source to create a path
src1.move(np.linspace((0, 0, 0.1), (0, 0, 8), 20))

# manipulate second source
src2.move(np.linspace((0.1, 0, 0.1), (5, 0, 5), 50))
src2.rotate_from_angax(angle=np.linspace(10, 600, 50), axis="z", anchor=0, start=1)

# setup plotly figure and subplots
fig = go.Figure().set_subplots(rows=1, cols=3, specs=[[{"type": "scene"}] * 3])
temp_fig = go.Figure()

# draw the objects
magpy.show(src1, canvas=temp_fig, backend='plotly')
fig.add_traces(temp_fig.data, rows=1, cols=1)
fig.layout.scene1 = temp_fig.layout.scene
temp_fig = go.Figure()
magpy.show(src2, canvas=temp_fig, backend='plotly')
fig.add_traces(temp_fig.data, rows=1, cols=2)
fig.layout.scene2 = temp_fig.layout.scene
temp_fig = go.Figure()
magpy.show(src1, src2, canvas=temp_fig, backend='plotly')
fig.add_traces(temp_fig.data, rows=1, cols=3)
fig.layout.scene3 = temp_fig.layout.scene

# display the system
fig.show()
```

---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Display backends

+++

The `magpylib` package is shipped with a display function which provides a graphical representation of every magnet, current and sensor of the library. To date the library includes two possible backends:

- matplotlib (by default)
- plotly

+++

Display multiple objects, object paths, markers in 3D using Matplotlib:

```{code-cell} ipython3
import magpylib as magpy
magnet = magpy.magnet.Cuboid(magnetization=(1, 0, 0), dimension=(1, 2, 3))
sens = magpy.Sensor(position=(0, 0, 3))
magpy.display(magnet, sens, zoom=1)
```

Display figure on your own canvas (here Matplotlib 3D axis):

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy

my_axis = plt.axes(projection="3d")
src = magpy.magnet.Sphere(magnetization=(0, 0, 1), diameter=1)
src.move(np.linspace((0.1, 0, 0), (5,0,0), 50))
src.rotate_from_angax(angle=np.linspace(10, 500, 50), axis="z", anchor=0, start=1)
ts = [-0.4, 0, 0.4]
sens = magpy.Sensor(position=(0, 0, 2), pixel=[(x, y, 0) for x in ts for y in ts])
magpy.display(src, sens, canvas=my_axis)
plt.show()
```

The same objects can also be displayed using the `plotly` plotting backend

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

src = magpy.magnet.Sphere(magnetization=(0, 0, 1), diameter=1)
src.move(np.linspace((0.1, 0, 0), (5,0,0), 50))
src.rotate_from_angax(angle=np.linspace(10, 500, 50), axis="z", anchor=0, start=1)
ts = [-0.4, 0, 0.4]
sens = magpy.Sensor(position=(0, 0, 2), pixel=[(x, y, 0) for x in ts for y in ts])
magpy.display(src, sens, backend="plotly")
```

The display function is also available as a class method and can be called for every object separately.

```{code-cell} ipython3
import plotly.graph_objects as go
import magpylib as magpy

fig = go.Figure()
ts = [-0.4, 0, 0.4]
sens = magpy.Sensor(position=(0, 0, 2), pixel=[(x, y, 0) for x in ts for y in ts])
sens.display(canvas=fig, backend="plotly", zoom=1, style_size=5)
fig
```

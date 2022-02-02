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
magpy.show(magnet, sens, zoom=1)
```

Display figure on your own canvas (here Matplotlib 3D axis):

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy

# setup matplotlib figure and subplots
fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(131, projection='3d')  # 3D-axis
ax2 = fig.add_subplot(132, projection='3d')  # 3D-axis
ax3 = fig.add_subplot(133, projection='3d')  # 3D-axis

# define sources
src1 = magpy.magnet.Sphere(magnetization=(0, 0, 1), diameter=1)
src2 = magpy.magnet.Cylinder(magnetization=(0, 0, 1), dimension=(1,2))

# manipulate first source to create a path
src1.move(np.linspace((0,0,0.1), (0,0,8), 20))

# manipulate second source
src2.move(np.linspace((0.1, 0, 0.1), (5,0,5), 50))
src2.rotate_from_angax(angle=np.linspace(10, 600, 50), axis="z", anchor=0, start=1)

# draw the objects
magpy.show(src1, canvas=ax1)
magpy.show(src2, canvas=ax2)
magpy.show(src1, src2, canvas=ax3)

# display the system
plt.tight_layout()
plt.show()
```

The same objects can also be displayed using the `plotly` plotting backend

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy

# define sources
src1 = magpy.magnet.Sphere(magnetization=(0, 0, 1), diameter=1)
src2 = magpy.magnet.Cylinder(magnetization=(0, 0, 1), dimension=(1,2))

# manipulate first source to create a path
src1.move(np.linspace((0,0,0.1), (0,0,8), 20))

# manipulate second source
src2.move(np.linspace((0.1, 0, 0.1), (5,0,5), 50))
src2.rotate_from_angax(angle=np.linspace(10, 600, 50), axis="z", anchor=0, start=1)

# display the system
magpy.show(src1, src2, backend='plotly')
```

The `show` function is also available as a class method and can be called for every object separately.

```{code-cell} ipython3
import plotly.graph_objects as go
import magpylib as magpy

fig = go.Figure()
ts = [-0.4, 0, 0.4]
sens = magpy.Sensor(position=(0, 0, 2), pixel=[(x, y, 0) for x in ts for y in ts])
sens.show(canvas=fig, backend="plotly", zoom=1, style_size=5)
fig
```

---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# This is a test example with magpylib.display

```{code-cell} ipython3
import magpylib as magpy

cuboid = magpy.magnet.Cuboid(magnetization=(1,0,0), dimension=(8, 4 ,6), position=(0,0,0))
cylinder = magpy.magnet.CylinderSegment(dimension=(6,10,4,0,90), position=(15,0,15),
    magnetization=(1,0,0))\
    .rotate_from_angax(axis=(0,0,1), angle= 45),

col = magpy.Collection(cuboid, cylinder)
magpy.defaults.reset()
magpy.defaults.display.backend = 'matplotlib'
#magpy.defaults.display.style.magnet.magnetization.show = False
cuboid.style.magnetization.show = True
col.set_styles(
    magnetization_show=True,
    magnetization_size=1,
)
magpy.display(
    col,
    #style_magnetization_show=True,
    #style_magnetization_size=1,
    backend='plotly',

)
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy

# setup matplotlib figure and subplots
fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(131, projection='3d')  # 3D-axis
ax2 = fig.add_subplot(132, projection='3d')  # 3D-axis
ax3 = fig.add_subplot(133)                   # 2D-axis

# define two sources and display in figure
src1 = magpy.magnet.CylinderSegment(magnetization=(0,0,1000), dimension=(2,3,1,-45,45))
src2 = magpy.current.Circular(current=500, diameter=1)
magpy.display(src1, src2, canvas=ax1, style_magnetization_size=0.3)

# manipulate source position and orientation and display
src2.move((0,0,1))
src1.rotate_from_angax(90, 'y', anchor=0)
magpy.display(src1, src2, canvas=ax2, style_magnetization_size=0.3)

# create a grid
ts = np.linspace(-4,4,30)
grid = np.array([[(x,0,z) for x in ts] for z in ts])

# compute field on grid
B = magpy.getB([src1,src2], grid, sumup=True)
amp = np.linalg.norm(B, axis=2)
```

```{code-cell} ipython3
import plotly.graph_objects as go
fig = go.Figure()
fig.add_scatter3d(x=grid[:,:,0].flatten(), y=grid[:,:,1].flatten(), z=grid[:,:,2].flatten(), surfaceaxis=0, surfacecolor='red')
```

```{code-cell} ipython3

```

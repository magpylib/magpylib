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

# Modelling a Coil

+++

A coil consists of large number of windings that can be modeled using `Loop` sources. The total coil is then a `Collection` of windings. One must be careful to take the line-current approximation into consideration. This means that the field diverges when approaching the current, while the field is correct outside a hypothetical wire with homogeneous current distribution.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy

# create figure using Matplotlib
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121, projection='3d')  # 3D-axis
ax2 = fig.add_subplot(122,)                  # 2D-axis

# create a Magpylib collection of Loop Sources that form a coil
coil = magpy.Collection()
for z in np.linspace(-2,2,20):
    winding = magpy.current.Loop(
        current = 1,
        diameter = 5,
        position = (0,0,z))
    coil += winding

# display the coil on ax1
coil.show(canvas=ax1)

# create a grid
ts = np.linspace(-6,6,30)
grid = np.array([[(x,0,z) for x in ts] for z in ts])

# compute field on grid
B = magpy.getB(coil, grid)
amp = np.linalg.norm(B, axis=2)

# display field in figure with matplotlib
ax2.streamplot(grid[:,:,0], grid[:,:,2], B[:,:,0], B[:,:,2],
    density=2, color=np.log(amp), linewidth=1, cmap='autumn')

plt.tight_layout()
plt.show()
```

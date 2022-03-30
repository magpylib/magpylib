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

# Field of a Coil

In this example we model the magnetic field of a coil. Teh coil consists of multiple circular current loops, the windings, each of which can be modeled using Magpylib `Loop` source objects. The individual windings are combined in a `Collection` which itself behaves like a single magnetic field source.

One must be careful to take the line-current approximation into consideration. This means that the field diverges when approaching the current line, while the field is correct outside a hypothetical wire with homogeneous current distribution.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy

fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(121, projection='3d')  # 3D-axis
ax2 = fig.add_subplot(122)                   # 2D-axis

# create coil compound object
coil = magpy.Collection()
for z in np.linspace(-2, 2, 10):
    winding = magpy.current.Loop(current=1, diameter=5, position=(0,0,z))
    coil.add(winding)

# coil now behaves like a single object
coil.rotate_from_angax(angle=25, axis='y')

# display 3D model
coil.show(canvas=ax1)

# compute and display coil-field on grid
ts = np.linspace(-6,6,30)
grid = np.array([[(x,0,z) for x in ts] for z in ts])
B = magpy.getB(coil, grid)

sp = ax2.streamplot(grid[:,:,0], grid[:,:,2], B[:,:,0], B[:,:,2], density=2,
    color=(np.linalg.norm(B, axis=2)), linewidth=1, cmap='autumn')

ax2.set(
    xlabel='x-position [mm]',
    ylabel='z-position [mm]',
    aspect=1,
)
plt.colorbar(sp.lines, ax=ax2, label='[mT]')

plt.tight_layout()
plt.show()
```

In {ref}`intro-collections` we show a similar example where a Helmholtz coil is modeled using nested collections.

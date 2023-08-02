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

(gallery-viz-mpl-streamplot)=

# Streamplot

In this example we show the B-field of a cuboid magnet using streamlines. Streamlines are not magnetic field lines in the sense that the fiel amplitude cannot be derived from their density. However, matplotlib streamlines can show the field amplitude via color and line thickness. One must be carefult that streamlines can only display two components of the field. In the examples below the third component is always zero.

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

import magpylib as magpy

fig, ax = plt.subplots()

# create an observer grid in the xz-symmetry plane
X, Z = np.mgrid[-5:5:40j, -5:5:40j].transpose((0, 2, 1))
grid = np.stack([X, np.zeros((40, 40)), Z], axis=2)

# compute B-field of a cuboid magnet on the grid
cube = magpy.magnet.Cuboid(magnetization=(500, 0, 500), dimension=(2, 2, 2))
B = cube.getB(grid)
Bamp = np.linalg.norm(B, axis=2)
log_Bamp = np.log(np.linalg.norm(B, axis=2))

# display field with Pyplot
splt = ax.streamplot(
    X,
    Z,
    B[:, :, 0],
    B[:, :, 2],
    color=log_Bamp,
    density=1,
    linewidth=log_Bamp,
    cmap="autumn",
)

# outline magnet boundary
ax.plot([1, 1, -1, -1, 1], [1, -1, -1, 1, 1], "k--", lw=2)

ax.set(
    xlabel="x-position (mm)",
    ylabel="z-position (mm)",
)
fig.colorbar(splt.lines, ax=ax, label="ln|B| (mT)")

plt.tight_layout()
plt.show()
```


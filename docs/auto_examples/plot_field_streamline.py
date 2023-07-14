"""
Streamline field
================

Plot the magnetic field lines with matplotlib streamline
"""
# %%
# In this example we show the B-field of a cuboid magnet in the symmetry plane using
import matplotlib.pyplot as plt
import numpy as np

import magpylib as magpy

fig, ax = plt.subplots()

# create an observer grid in the xz-symmetry plane
ts = np.linspace(-3, 3, 30)
grid = np.array([[(x, 0, z) for x in ts] for z in ts])

# compute B-field of a cuboid magnet on the grid
cube = magpy.magnet.Cuboid(magnetization=(500, 0, 500), dimension=(2, 2, 2))
B = cube.getB(grid)

# display field with Pyplot
ax.streamplot(
    grid[:, :, 0],
    grid[:, :, 2],
    B[:, :, 0],
    B[:, :, 2],
    density=2,
    color=np.log(np.linalg.norm(B, axis=2)),
    linewidth=1,
    cmap="autumn",
)

# outline magnet boundary
ax.plot([1, 1, -1, -1, 1], [1, -1, -1, 1, 1], "k--")

plt.tight_layout()
plt.show()

---
orphan: true
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(examples-app-halbach)=

# Halbach Magnets

Magpylib is an excellent tool to create magnet assemblies. In this example we will show how to model Halbach magnets.

```{note}
In the following examples we make use of the [arbitrary unit convention](guide-docs-io-scale-invariance).
```

The original Halbach-magnetization describes a hollow cylinder with a polarization direction that rotates twice while going around the cylinder once. In reality such polarizations are difficult to fabricate. What is commonly done instead are "Discreete Halbach Arrays", which are magnet assemblies that approximate a Halbach magnetization.

The following code creates a Discreete Halbach Cylinder generated from Cuboids:

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

N = 10
angles = np.linspace(0, 360, N, endpoint=False)

halbach = magpy.Collection()

for a in angles:
    cube = magpy.magnet.Cuboid(
        dimension=(1,1,1),
        polarization=(1,0,0),
        position=(2.3,0,0)
    )
    cube.rotate_from_angax(a, 'z', anchor=0)
    cube.rotate_from_angax(a, 'z')
    halbach.add(cube)

halbach.show(backend='plotly')
```

Next we compute and display the field on an xy-grid in the symmetry plane using the [matplotlib streamplot](examples-vis-mpl-streamplot) example.

```{code-cell} ipython3
# Continuation from above - ensure previous code is executed

import matplotlib.pyplot as plt
fig, ax = plt.subplots()

# Compute and plot field on x-y grid
grid = np.mgrid[-3.5:3.5:100j, -3.5:3.5:100j, 0:0:1j].T[0]
X, Y, _ = np.moveaxis(grid, 2, 0)

B = halbach.getB(grid)
Bx, By, _ = np.moveaxis(B, 2, 0)
Bamp = np.linalg.norm(B, axis=2)

pc = ax.contourf(X, Y, Bamp, levels=50, cmap="coolwarm")
ax.streamplot(X, Y, Bx, By, color="k", density=1.5, linewidth=1)

# Add colorbar
fig.colorbar(pc, ax=ax, label="|B|")

# Figure styling
ax.set(
    xlabel="x-position",
    ylabel="z-position",
    aspect=1,
)

plt.show()
```

```{warning}
Magpylib models magnets with perfect polarization. However, such magnets do not exist in reality due to fabrication tolerances and material response. While fabrication tolerances can be estimated easily, our [tutorial](examples-tutorial-modelling-magnets) explains how to deal with material response.
```

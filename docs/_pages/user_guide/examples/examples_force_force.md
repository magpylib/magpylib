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

(examples-force-force)=

# Magnetic Force and Torque

The `magpylib-force` extension provides force and torque computation between Magpylib objects. A detailed description of the API and how the computation is achieved can be found in the [user guide](docs-force-computation).

In the following example we show how to compute force and torque between two objects and how to represent it graphically.

```{code-cell} ipython3
import pyvista as pv
import magpylib as magpy
from magpylib_force import getFT

# Source
coil = magpy.current.Circle(position=(0,0,1), diameter=3, current=1000)

# Target
cube = magpy.magnet.Cuboid(dimension=(1,1,1), polarization=(1,0,1))
cube.meshing = (20,20,20)

# Compute force and torque
F,T = getFT(coil, cube, anchor=None)

# Plot force and torque in Pyvista with arrows
pl = magpy.show(coil, cube, backend='pyvista', return_fig=True)
arrowF = pv.Arrow(start=(0,0,0), direction=F)
pl.add_mesh(arrowF, color="blue")
arrowT = pv.Arrow(start=(0,0,0), direction=T)
pl.add_mesh(arrowT, color="white")

pl.show()
```
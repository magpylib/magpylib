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

```{warning}
[Scaling invariance](guide-docs-io-scale-invariance) does not hold for force computations! Be careful to provide the inputs in the correct units!
```

In the following example we show how to compute force and torque between two objects and how to represent it graphically.

```{code-cell} ipython3
import pyvista as pv
import magpylib as magpy
from magpylib_force import getFT

# Source
coil = magpy.current.Circle(position=(0,0,-.5), diameter=4, current=1000)
coil.rotate_from_angax(angle=-20, axis='y')

# Target
cube = magpy.magnet.Cuboid(dimension=(.7,.7,.7), polarization=(0,0,1))
cube.meshing = (10,10,10)

# Compute force and torque
F,T = getFT(coil, cube, anchor=None)

print(f"Force (blue):    {[round(f) for f in F]} N")
print(f"Torque (yellow): {[round(t) for t in T]} Nm")
```

Force and torque are really strong in this example, because the magnet and the coil are very large objects. With 0.7 m side length, the magnet has a Volume of ~1/3rd cubic meter :).

```{code-cell} ipython3
# Example continued from above

# Plot force and torque in Pyvista with arrows
pl = magpy.show(coil, cube, backend='pyvista', return_fig=True)
arrowF = pv.Arrow(start=(0,0,0), direction=F)
pl.add_mesh(arrowF, color="blue")
arrowT = pv.Arrow(start=(0,0,0), direction=T)
pl.add_mesh(arrowT, color="yellow")

pl.show()
```
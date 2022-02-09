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

# Basic functionality

+++

## Just compute the field

The most fundamental functionality of the library - compute the field (B in \[mT\], H in \[kA/m\]) of a
source (here Cylinder magnet) at the observer position (1,2,3).

```{code-cell} ipython3
from magpylib.magnet import Cylinder

src = Cylinder(magnetization=(222, 333, 444), dimension=(2, 2))
B = src.getB((1, 2, 3))
print(B)
```

## Field values of a path

In this example the field B in \[mT\] of the Cylinder magnet is evaluated for a moving observer,
rotating 360° with 45° steps around the source along the z-axis and a radius of 5\[mm\].

```{code-cell} ipython3
import numpy as np
from magpylib import Sensor
from magpylib.magnet import Cylinder

cyl = Cylinder(magnetization=(222, 333, 444), dimension=(2, 2))
sens = Sensor(position=(5, 0, 0))
sens.rotate_from_angax(np.linspace(0.0, 270, 12)[1:], "z", anchor=cyl.position)
B = sens.getB(src)
print(B)
```

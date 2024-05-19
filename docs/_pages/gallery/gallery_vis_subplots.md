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

What is shown is the following examples works best in the [plotly backend](examples-backends-canvas).

(gallery-vis-subplots)=

# Subplots

It is very illustrative to combine 2D and 3D subplots when viewing the field along paths. For this, Magpylib offers a built-in functionality:

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

# define sensor and source
sens = magpy.Sensor(pixel=[(-.2, 0, 0), (.2, 0, 0)])
cyl = magpy.magnet.Cylinder(
    polarization=(1, 0, 0),
    dimension=(2, 1),
    position=(4,0,0),
)

# define paths
N = 40
sens.position = np.linspace((0, 0, -3), (0, 0, 3), N)
cyl.rotate_from_angax(angle=np.linspace(0, 300, N), start=0, axis="z", anchor=0)

# combine field plot and 3D-plot
magpy.show(
    dict(objects=[cyl, sens], output="Bx", col=1),
    dict(objects=[cyl, sens], output="model3d", col=2),
    backend='plotly',
)
```

# Subplot animations

This is specifially illustrative as an animation where the field at the respective path position is indicated by a marker:

```{code-cell} ipython3
magpy.show(
    dict(objects=[cyl, sens], output=["Bx", "By", "Bz"], col=1),
    dict(objects=[cyl, sens], output="model3d", col=2),
    backend='plotly',
    animation=True,
)
```

# show_context

Which can be generated with maximal ease using the [show_context](docu-graphics-show_context) context manager:

```{code-cell} ipython3
with magpy.show_context([cyl, sens], backend='plotly', animation=True) as sc:
    sc.show(output="Bx", col=1, row=1)
    sc.show(output="By", col=1, row=2)
    sc.show(output="Bz", col=2, row=1)
    sc.show(output="model3d", col=2, row=2)
```

Read up everything about subplots [here](docu-graphics-subplots).

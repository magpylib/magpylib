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

(gallery-vis-animations)=

# Animations

Magpylib can display the motion of objects along paths in the form of animations. The following example shows how to set up such an animation.

```{hint}
If your browser window opens, but your animation does not load, reload the page (ctrl+r in chrome).
```

Detailed information about how to tune animations can be found in the [graphics documentation](examples-animation). Animations work best in the [plotly backend](examples-backends-canvas). Avoid rendering too many frames.

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

# create sensor with path
sensor=magpy.Sensor().rotate_from_angax(
    angle=35*np.sin(np.linspace(-4,4,36)),
    axis='y',
    anchor=(0,0,-.05),
    start=0,
)

# create magnet with path
magnet = magpy.magnet.Cuboid(
    dimension=(0.02,0.01,0.01),
    polarization=(0.3,0,0),
    position=(0,0,-.03)
).rotate_from_angax(
    angle=np.linspace(0,360,37),
    axis='z',
    start=0,
)

# display as animation - prefers plotly backend
magpy.show(sensor, magnet, animation=True, backend='plotly')
```

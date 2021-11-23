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

# Animate paths

+++

For objects for which a path has been constructed using the path-API, the ``magpylib`` library
enables, users to animate the object movements, in addition to displaying them statically.

+++

```{note}
This feature is only available for the ``plotly`` backend at the moment.
```

+++

While the orientation is static while the animation is running, it can be dynamically set when the
the animation is on pause.

```{code-cell} ipython3
import numpy as np
import plotly.graph_objects as go
import magpylib as magpy

# Define sources and sensor
dim = 2
col = magpy.Collection(
    [
        magpy.magnet.Cuboid(magnetization=(0, 1, 0), dimension=[dim] * 3),
        magpy.magnet.Cylinder(magnetization=(0, 1, 0), dimension=[dim] * 2),
        magpy.magnet.Sphere(magnetization=(0, 1, 0), diameter=dim),
    ]
)

ts = np.arange(-0.6, 0.6, 0.2)
sens = magpy.Sensor(position=(0, 0, 2), pixel=[(x, y, 0) for x in ts for y in ts])

# Create paths
pathlen = 40
for displ, src in zip([(0.1414, 0, 0.1), (-0.1, -0.1, 0.1), (-0.1, 0.1, 0.1)], col):
    src.move([np.array(displ)] * pathlen, increment=True)
    src.rotate_from_angax(
        angle=[10] * pathlen, axis="z", anchor=0, start=0, increment=True
    )
sens.move([(0, 0, 0.1)] * 10, increment=True)

# display animation
fig = go.Figure()
magpy.display(
    col,
    sens,
    canvas=fig,
    path="animate",
    zoom=0,
    animate_time=2,
    animate_fps=20,
    animate_slider=True,
    backend="plotly",
)
fig.update_layout(height=800)
```

```{warning}
Even if some failsafes are implemented such as a maximum frame rate and frame count. There is no
guarantee that the animation will be able to be rendered. This is particularly relevant if the
user tries to animate many objects at the same time.
```

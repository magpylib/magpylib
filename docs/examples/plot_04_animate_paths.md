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

# Animate paths

+++

For objects for which a path has been constructed using the path-API, the Magpylib library enables users to animate their movements, in addition to displaying them statically.

+++

```{note}
This feature is only available for the ``plotly`` backend at the moment.
```

```{code-cell} ipython3
import numpy as np
import plotly.graph_objects as go
import magpylib as magpy

# Define sources and sensor
dim = 2
coll = magpy.Collection(
    magpy.magnet.Cuboid(magnetization=(0, 1, 0), dimension=[dim] * 3),
    magpy.magnet.Cylinder(magnetization=(0, 1, 0), dimension=[dim] * 2),
    magpy.magnet.Sphere(magnetization=(0, 1, 0), diameter=dim),
)

ts = np.linspace(-0.6, 0.6, 5)
sens = magpy.Sensor(position=(0, 0, 2), pixel=[(x, y, 0) for x in ts for y in ts])

# Create paths
pathlen = 50
start_positions = np.array([(1.414, 0, 1), (-1, -1, 1), (-1, 1, 1)])
for pos, src in zip(start_positions, coll):
    src.move(np.linspace(pos, pos*pathlen*0.1, pathlen), start=0)
    src.rotate_from_angax(np.linspace(0., 360., pathlen), 'z', anchor=0, start=0)
sens.move(np.linspace((0.,0.,5.), (0.,0.,-5.), 20), start=0)

# display animation
fig = go.Figure()
magpy.show(
    *coll,
    sens,
    canvas=fig,
    animation=3, #animation time set to 3 seconds
    zoom=0,
    animation_fps=20,
    animation_slider=True,
    backend="plotly",
)
fig.update_layout(height=800)
```

```{note}
While the canvas orientation is static while the animation is running, it can be dynamically set with by draging the mouse when the animation is on pause or finished.
Also note that the animation of the sensor stops earlier since it has a shorter path length.
```

```{warning}
Even if some failsafes are implemented such as a maximum frame rate and frame count. There is no
guarantee that the animation will be able to be rendered. This is particularly relevant if the
user tries to animate many objects and/or many path positions at the same time.
```

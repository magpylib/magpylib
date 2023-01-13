---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(examples-animation)=

# Animate paths

With some backends, paths can automatically be animated with `show(animation=True)`. Animations can be fine-tuned with the following properties:

1. `animation_time` (default=3), must be a positive number that gives the animation time in seconds.
2. `animation_slider` (default=`True`), is boolean and sets if a slider should be displayed in addition.
3. `animation_fps` (default=30), sets the maximal frames per second.

Ideally, the animation will show all path steps, but when e.g. `time` and `fps` are too low, specific equidistant frames will be selected to adjust to the limited display possibilities. For practicality, the input `animation=x` will automatically set `animation=True` and `animation_time=x`.

The following example demonstrates the animation feature,

```{code-cell} ipython3
import numpy as np
import magpylib as magpy
from magpylib.magnet import Cuboid, Cylinder, Sphere

# define objects with paths
coll = magpy.Collection(
    Cuboid(magnetization=(0,1,0), dimension=(2,2,2)),
    Cylinder(magnetization=(0,1,0), dimension=(2,2)),
    Sphere(magnetization=(0,1,0), diameter=2),
)

start_positions = np.array([(1.414, 0, 1), (-1, -1, 1), (-1, 1, 1)])
for pos, src in zip(start_positions, coll):
    src.position = np.linspace(pos, pos*5, 50)
    src.rotate_from_angax(np.linspace(0, 360, 50), 'z', anchor=0, start=0)

ts = np.linspace(-0.6, 0.6, 5)
sensor = magpy.Sensor(pixel=[(x, y, 0) for x in ts for y in ts])
sensor.position = np.linspace((0,0,-5), (0,0,5), 20)

# show with animation
magpy.show(coll, sensor,
    animation=3,
    animation_fps=20,
    animation_slider=True,
    backend='plotly',
    showlegend=False,      # kwarg to plotly
)
```

Notice that the sensor, with the shorter path stops before the magnets do. This is an example where {ref}`examples-edge-padding-end-slicing` is applied.

```{warning}
Even with some implemented failsafes, such as a maximum frame rate and frame count, there is no guarantee that the animation will be rendered properly. This is particularly relevant when the user tries to animate many objects and/or many path positions at the same time.
```

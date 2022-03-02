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

(examples-animation)=

# Animate paths

With the powerful Plotly backend, Magpylib supports animation of the object paths through the `animation` kwarg of the `show` function. Animations can be fine-tuned with the following properties:

1. `animation_time` must be a positive number that gives the animation time in seconds.
2. `animation_slider` is boolean and sets if a slider should be displayed in addition.
3. `animation_fps` sets the frames per second.

Ideally, the animation will show all path steps, but when e.g. `time` and `fps` are too low, specific equidistant frames will be selected to adjust to the limited display possibilities. The `show` input `animation=15` will automatially set `animation=True` and `animation_time=15`.

The following example demonstrates the animation feature,

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

# define objects with paths
coll = magpy.Collection(
    magpy.magnet.Cuboid(magnetization=(0,1,0), dimension=(2,2,2)),
    magpy.magnet.Cylinder(magnetization=(0,1,0), dimension=(2,2)),
    magpy.magnet.Sphere(magnetization=(0,1,0), diameter=2),
)

start_positions = np.array([(1.414, 0, 1), (-1, -1, 1), (-1, 1, 1)])
for pos, src in zip(start_positions, coll):
    src.position = np.linspace(pos, pos*5, 50)
    src.rotate_from_angax(np.linspace(0, 360, 50), 'z', anchor=0, start=0)

ts = np.linspace(-0.6, 0.6, 5)
sens = magpy.Sensor(pixel=[(x, y, 0) for x in ts for y in ts])
sens.position = np.linspace((0,0,-5), (0,0,5), 20)

# show with animation
magpy.show(coll, sens,
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

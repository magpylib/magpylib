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

(examples-vis-animations)=

# Animations

Magpylib can display the motion of objects along paths in the form of animations.

```{hint}
1. Animations work best with the [plotly backend](guide-graphic-backends).

2. If your browser window opens, but your animation does not load, reload the page (ctrl+r in chrome).

3. Avoid rendering too many frames.
```

Detailed information about how to tune animations can be found in the [graphics documentation](guide-graphic-animations).

## Simple Animations

Animations are created with `show` by setting `animation=True`. It is also possible to hand over the animation time with this kwarg.

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

# Define magnet with path
magnet = magpy.magnet.Cylinder(
    polarization=(1, 0, 0),
    dimension=(2, 1),
    position=(4,0,0),
    style_label="magnet",
)
magnet.rotate_from_angax(angle=np.linspace(0, 300, 40), start=0, axis="z", anchor=0)

# Define sensor with path
sensor = magpy.Sensor(
    pixel=[(-.2, 0, 0), (.2, 0, 0)],
    position = np.linspace((0, 0, -3), (0, 0, 3), 40),
    style_label="sensor",
)

# Display as animation - prefers plotly backend
magpy.show(sensor, magnet, animation=True, backend='plotly')
```

(examples-vis-animated-subplots)=

## Animated Subplots

[Subplots](examples-vis-subplots) are a powerful tool to see the field along a path while viewing the 3D models at the same time. This is specifically illustrative as an animation where the field at the respective path position is indicated by a marker.

```{code-cell} ipython3
# Continuation from above - ensure previous code is executed

magpy.show(
    dict(objects=[magnet, sensor], output=["Bx", "By", "Bz"], col=1),
    dict(objects=[magnet, sensor], output="model3d", col=2),
    backend='plotly',
    animation=True,
)
```

It is also possible to use the [show_context](guide-graphics-show_context) context manager.

```{code-cell} ipython3
# Continuation from above - ensure previous code is executed

with magpy.show_context([magnet, sensor], backend='plotly', animation=True) as sc:
    sc.show(output="Bx", col=1, row=1)
    sc.show(output="By", col=1, row=2)
    sc.show(output="Bz", col=2, row=1)
    sc.show(output="model3d", col=2, row=2)
```

## Exporting Animations

Animations are wonderful but can be quite difficult to export when they are needed, for example, in a presentation. Here we show how to create customizable *.gif animations.

### Built-in export

The easiest way is use the Magpylib built-in command `animation_output` in the `show` function. It works only with the Pyvista backend.

```python
import magpylib as magpy
import numpy as np

# Create magnets with Paths
path = [(np.sin(t)+1.5,0,0) for t in np.linspace(0, 2*np.pi, 50)]
cube1 = magpy.magnet.Cuboid(dimension=(1,1,1), polarization=(1,0,0), position= path)
cube2 = cube1.copy(position=-np.array(path), polarization=(-1,0,0))

# Store gif with animation_output using Pyvista
magpy.show(
    cube1,
    cube2,
    style_legend_show=False,
    animation=3,
    animation_output="test4.gif",
    backend="pyvista",
    style_path_show=False,
)
```

### Custom export Pyvista

For customizing videos it is best to work directly in the respective graphic backends.



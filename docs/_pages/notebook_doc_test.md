---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"tags": []}

# This is a test example with magpylib.display

```{code-cell} ipython3
import magpylib as magpy


cuboid = magpy.magnet.Cuboid(magnetization=(1,0,0), dimension=(8, 4 ,6), position=(0,0,0))
cylinder = magpy.magnet.CylinderSegment(dimension=(6,10,4,0,90), position=(15,0,15), magnetization=(1,0,0))\
    .rotate_from_angax(axis=(0,0,1), angle= 45),

col = magpy.Collection(cuboid, cylinder)
magpy.defaults.reset()
magpy.defaults.display.backend = 'matplotlib'
#magpy.defaults.display.style.magnet.magnetization.show = False
cuboid.style.magnetization.show = True
col.set_styles(
    magnetization_show=True,
    magnetization_size=1,
)
magpy.display(
    col,
    #style_magnetization_show=True,
    #style_magnetization_size=1,
    backend='plotly',

)
```

```{warning}
Even if all objects can be represented both by the `matplotlib` and `plotly` plotting backends, there is no 100% feature parity bewteen them.
```

````{tabbed} Matplotlib
```{code-cell} ipython3
import magpylib as magpy

src = magpy.magnet.Sphere(magnetization=(0, 0, 1), diameter=1)
src.move([(0.1, 0, 0)] * 50, increment=True)
src.rotate_from_angax(angle=[10] * 50, axis="z", anchor=0, start=0, increment=True)
ts = [-0.4, 0, 0.4]
sens = magpy.Sensor(position=(0, 0, 2), pixel=[(x, y, 0) for x in ts for y in ts])

magpy.display(src, sens)
```
````

````{tabbed} Plotly
# The same objects can also be displayed using the `plotly` plotting backend
import plotly.graph_objects as go

```{code-cell} ipython3
import magpylib as magpy

src = magpy.magnet.Sphere(magnetization=(0, 0, 1), diameter=1)
src.move([(0.1, 0, 0)] * 50, increment=True)
src.rotate_from_angax(angle=[10] * 50, axis="z", anchor=0, start=0, increment=True)
ts = [-0.4, 0, 0.4]
sens = magpy.Sensor(position=(0, 0, 2), pixel=[(x, y, 0) for x in ts for y in ts])

magpy.display(src, sens, path="animate", backend="plotly")
```
````
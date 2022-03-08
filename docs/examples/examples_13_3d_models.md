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

(examples-3d-models)=

# 3D-models

(examples-own-3d-models)=
## Custom 3D models

Each Magpylib object has a default 3D representation which is displayed with `show`. Users can add a custom 3D model to any Magpylib object with help of the `style.model3d.add_trace(trace, backend)` method. The added trace is then stored in `style.model3d.data`. User-defined traces move with the object just like the default models do. The default trace can be hidden with the command `obj.model3d.showdefault=False`.

The input `trace` of `add_trace` is a dictionary which includes all necessary information for plotting, and `backend` states which graphic backend can interpret the trace. The idea is that, `trace` includes a `type` argument, which is the designation of the 3D plot function to be used. All other trace kwargs are then handed to this function.

The following example for the **Plotly** backend, shows how a `mesh3d` trace and a `scatter3d` trace are constructed:

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

# use mesh3d ##########################################################

trace1 = dict(
    type='mesh3d',
    x=(1, 0, -1, 0),
    y=(-.5, 1.2, -.5, 0),
    z=(-.5, -.5, -.5, 1),
    i=(0, 0, 0, 1),
    j=(1, 1, 2, 2),
    k=(2, 3, 3, 3),
    opacity=0.5,
)

coll = magpy.Collection(position=(4,0,0),style_label="'mesh3d' trace")
coll.style.model3d.add_trace(trace=trace1, backend='plotly')

# use scatter3d #######################################################

ts = np.linspace(0, 2*np.pi, 30)
trace2 = dict(
    type='scatter3d',
    x=np.cos(ts),
    y=np.zeros(30),
    z=np.sin(ts),
    mode='lines',
)

src = magpy.misc.Dipole(moment=(0,0,1), style_label="'scatter3d' trace")
src.style.model3d.add_trace(trace=trace2, backend='plotly')

magpy.show(coll, src, backend='plotly')
```

 It is also possible to have multiple user-defined traces that will be displayed at the same time:

```{code-cell} ipython3
coll.rotate_from_angax(np.linspace(0, 270, 30), 'z', anchor=0)

trace2 = dict(
    type='scatter3d',
    x=np.cos(ts),
    y=np.sin(ts),
    z=np.zeros(30),
    mode='lines',
)
src.style.model3d.add_trace(trace=trace2, backend="plotly")

trace3 = dict(
    type='scatter3d',
    x=np.zeros(30),
    y=np.cos(ts),
    z=np.sin(ts),
    mode='lines',
)
src.style.model3d.add_trace(trace=trace3, backend="plotly")

src.style.label='multiple traces'
magpy.show(coll, src, backend='plotly')
```

**Matplotlib** plotting functions often use positional args for $(x,y,z)$ input, and the input format is not standardized. Positional args are handed over as `args=(x,y,z)` in `trace`. The following examples show how to use `plot`, `plot_surface` and `plot_trisurf` functionality,

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import magpylib as magpy

# using "plot" #########################################################

ts = np.linspace(-10,10,100)
xs = np.cos(ts)
ys = np.sin(ts)
zs = ts/20

trace1 = dict(
    type='plot',
    args=(xs,ys,zs),
    ls='--',
)

obj1 = magpy.misc.Dipole(moment=(0,0,1), style_size=2)
obj1.style.model3d.add_trace(trace=trace1, backend="matplotlib")

# using "plot_surface" ################################################

u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
xs = np.cos(u) * np.sin(v)
ys = np.sin(u) * np.sin(v)
zs = np.cos(v)

trace2 = dict(
    type='plot_surface',
    args=(xs,ys,zs),
    cmap=plt.cm.YlGnBu_r,
)

obj2 = magpy.Collection(position=(-3,0,0))
obj2.style.model3d.add_trace(trace=trace2, backend='matplotlib')

# using "plot_trisurf" ###############################################

u, v = np.mgrid[0:2*np.pi:50j, -.5:.5:10j]
u, v = u.flatten(), v.flatten()

xs = (1 + 0.5 * v * np.cos(u / 2.0)) * np.cos(u)
ys = (1 + 0.5 * v * np.cos(u / 2.0)) * np.sin(u)
zs = 0.5 * v * np.sin(u / 2.0)

tri = mtri.Triangulation(u, v)

trace3 = dict(
    type="plot_trisurf",
    args=(xs,ys,zs),
    triangles=tri.triangles,
    cmap=plt.cm.Spectral,
)

obj3 = magpy.misc.CustomSource(style_model3d_showdefault=False, position=(3,0,0))
obj3.style.model3d.add_trace(trace=trace3, backend="matplotlib")

magpy.show(obj1, obj2, obj3)
```

## Pre-defined plotly models REWORK WITH API

For the plotly graphic backend, several pre-defined 3D models based on `Mesh3d` are provided in the `magpylib.display.plotly` sub-package:

```{code-cell} ipython3
import magpylib as magpy

obj = magpy.misc.CustomSource()

# add prism trace
trace_prism = magpy.display.plotly.make_BasePrism(
    base_vertices=6,
    diameter=2,
    height=1,
    position=(-3,0,0),
)
obj.style.model3d.add_trace(trace_prism, backend='plotly')

# add cone trace
trace_cone = magpy.display.plotly.make_BaseCone(
    base_vertices=30,
    diameter=2,
    height=1,
    position=(3,0,0)
)
obj.style.model3d.add_trace(trace_cone, backend='plotly')

# add cuboid trace
trace_cuboid = magpy.display.plotly.make_BaseCuboid(
    dimension=(2,2,2),
    position=(0,3,0),
)
obj.style.model3d.add_trace(trace_cuboid, backend='plotly')

# add cylinder segment trace
trace_cylinder_segment = magpy.display.plotly.make_BaseCylinderSegment(
    r1=1,
    r2=2,
    h=1,
    phi1=140,
    phi2=220,
    vert=50,
    position=(1,0,-3),
)
obj.style.model3d.add_trace(trace_cylinder_segment, backend='plotly')

# add ellipsoid trace
trace_ellipsoid = magpy.display.plotly.make_BaseEllipsoid(
    dimension=(2,2,1),
    vert=50,
    position=(0,0,3),
)
obj.style.model3d.add_trace(trace_ellipsoid, backend='plotly')

# add arrow trace
trace_arrow = magpy.display.plotly.make_BaseArrow(
    base_vertices=30,
    diameter=0.6,
    height=2,
    position=(0,-3,0),
)
obj.style.model3d.add_trace(trace_arrow, backend='plotly')

obj.show(backend='plotly')
```

## CAD models

A detailed example how to add complex CAD-file is given in {ref}`examples-adding-CAD-model`.

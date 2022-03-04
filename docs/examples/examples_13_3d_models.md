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
## Own 3D models

Each Magpylib object has a default 3D representation which is displayed with `show`. Through the style attributes `style.model3d.add_trace(trace, backend)` method it is possible to add a user defined 3D model to any object. The input `trace` is any dictionary that can be interpreted by the defined graphic `backend`.

1. For Plotly `trace` must be interpretable by a 3D trace constructor, such as `Scatter3d` or `Mesh3d` from the `plotly.graph_objects` module.
2. For Matpotlib it must be compatible with the constructors of `plot` or `plot_trisurf` of `mpl_toolkits.mplot3d.axes3d.Axes3D`.

In the following example with the **Plotly** backend, a `mesh3d` trace will be added to a collection object, which itself has no trace by default, and a `scatter3d` trace will be added to a dipole object:

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

# generate an interpretable plotly trace of type mesh3d
trace1 = dict(
    type='mesh3d',
    opacity=0.5,
    x=(1, 0, -1, 0),
    y=(-.5, 1.2, -.5, 0),
    z=(-.5, -.5, -.5, 1),
    i=(0, 0, 0, 1),
    j=(1, 1, 2, 2),
    k=(2, 3, 3, 3),
)

# add trace to collection object
coll = magpy.Collection()
coll.style.model3d.add_trace(trace=trace1, backend='plotly')

# generate an interpretable plotly trace of type scatter3d
ts = np.linspace(0, 2*np.pi, 30)
trace2 = dict(
    type='scatter3d',
    mode='lines',
    x=np.cos(ts),
    y=np.zeros(30),
    z=np.sin(ts),
)

# add trace to Dipole object
src = magpy.misc.Dipole(moment=(0,0,1), position=(4,0,0))
src.style.model3d.add_trace(trace=trace2, backend='plotly')

magpy.show(coll, src, backend='plotly')
```

User-defined traces move with the object just like the default models do. The default trace can be hidden with the command `obj.model3d.showdefault=False`. It is also possible to have multiple user-defined traces.

```{code-cell} ipython3
coll.rotate_from_angax(np.linspace(0, 270, 30), 'z', anchor=(4,0,0))

trace2 = dict(type='scatter3d', mode='lines', x=np.cos(ts), y=np.sin(ts), z=np.zeros(30))
trace3 = dict(type='scatter3d', mode='lines', x=np.zeros(30), y=np.cos(ts), z=np.sin(ts))
src.style.model3d.add_trace(trace=trace2, backend="plotly")
src.style.model3d.add_trace(trace=trace3, backend="plotly")

magpy.show(coll, src, backend='plotly')
```

For the **Matplotlib** backend it is necessary to provide all input data in dictionary form. In addition the `coordsargs` argument is used to indentify the respective function positional args. The following examples shows how to use `plot` and `plot_trisurf` functionality,

```{code-cell} ipython3
import magpylib as magpy

trace1 = dict(
    type='plot',
    xs=[-1, -1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1, 1,  1, -1],
    ys=[-1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1],
    zs=[-1, -1, -1, -1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1],
    ls='-',
)

src = magpy.misc.Dipole(moment=(0,0,1), position=(4,0,0), style_size=2)
src.style.model3d.add_trace(
    backend="matplotlib",
    trace=trace1,
    coordsargs=dict(x='xs', y='ys', z='zs'),
)

trace2 = dict(
    type="plot_trisurf",
    args=[
        (1, 0, -1, 0),
        (-.5, 1.2, -.5, 0),
        (-.5, -.5, -.5, 1)],
    triangles=[(0,1,2), (0,1,3), (0,2,3), (1,2,3)])

coll = magpy.Collection()
coll.style.model3d.add_trace(
    backend="matplotlib",
    trace=trace2,
    coordsargs={"x": "args[0]", "y": "args[1]", "z": "args[2]"},
)

magpy.show(src, coll)
```

## Pre-defined plotly models

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

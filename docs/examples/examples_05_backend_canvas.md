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

(examples-backends-canvas)=

# Backend and canvas

## Graphic backend

Magpylib supports several common graphic backends.

```{code-cell} ipython3
from magpylib import SUPPORTED_PLOTTING_BACKENDS
SUPPORTED_PLOTTING_BACKENDS
```

The installation default is Matplotlib. To select a graphic backend one can
1. Change the library default with `magpy.defaults.display.backend = 'plotly'`.
2. Set the `backend` kwarg in the `show` function, `show(..., backend='matplotlib')`.

There is a high level of **feature parity**, however, not all graphic features are supported by all backends. In addition, some common Matplotlib syntax (e.g. color `'r'`, linestyle `':'`) is automatically translated to other backends. The following table shows the comparison between backend for most of the features.


|                  Feature                                        | Matplotlib | Plotly | Pyvista | Mayavi |
|:---------------------------------------------------------------:|:----------:|:------:|:-------:|:------:|
| triangular mesh 3d                                              | ✔️         | ✔️    | ✔️      | ✔️    |
| line 3d                                                         | ✔️         | ✔️    | ✔️      | ✔️    |
| line style                                                      | ✔️         | ✔️    | ❌      | ❌    |
| line color                                                      | ✔️         | ✔️    | ✔️      | ✔️    |
| line width                                                      | ✔️         | ✔️    | ✔️      | ❌[^5]|
| marker 3d                                                       | ✔️         | ✔️    | ✔️      | ✔️    |
| marker color                                                    | ✔️         | ✔️    | ✔️      | ✔️    |
| marker size                                                     | ✔️         | ✔️    | ✔️      | ✔️    |
| marker symbol                                                   | ✔️         | ✔️    | ❌      | ✔️[^4]|
| marker numbering                                                | ✔️         | ✔️    | ❌      | ✔️    |
| zoom level                                                      | ✔️         | ✔️    | ❌[^2]  | ✔️    |
| magnetization color gradient                                    | ❌         | ✔️    | ✔️      | ✔️    |
| custom magnetization color gradient                             | ❌         | ✔️    | ✔️[^3]  | ✔️    |
| custom magnetization color gradient <br> for individual objects | ❌         | ✔️    | ✔️[^3]  | ✔️    |
| animation                                                       | ✔️         | ✔️    | ❌      | ✔️    |
| animation time                                                  | ✔️         | ✔️    | ❌      | ✔️    |
| animation fps                                                   | ✔️         | ✔️    | ❌      | ✔️    |
| animation slider                                                | ✔️[^1]     | ✔️    | ❌      | ❌    |
| user canvas                                                     | ✔️         | ✔️    | ✔️      | ✔️    |
| user extra 3d model  - generic [^6]                             | ✔️         | ✔️    | ✔️      | ✔️    |
| user extra 3d model  - backend specific [^7]                    | ✔️         | ✔️    | ❌      | ❌    |


[^1]: when returning animation object and exporting it as jshtml.

[^2]: possible but not implemented at the moment.

[^3]: does not work with ipygany jupyter backend.

[^4]: conversions are done to best match `"2dcross", "2dsquare", "2ddiamond", "2dcircle"`.

[^5]: technically possible but looks too displeasing to implement.

[^6]:  only `"scatter3d"`, and `"mesh3d"`. Gets "translated" to every other backend.

[^7]: custom user defined trace constructors  allowed, which are specific to the backend.


The following example demonstrates the currently supported backends:

```{code-cell} ipython3
import numpy as np
import magpylib as magpy

# define sources and paths
loop = magpy.current.Loop(current=1, diameter=1)
loop.position = np.linspace((0,0,-3), (0,0,3), 40)

cylinder = magpy.magnet.Cylinder(magnetization=(0,-100,0), dimension=(1,2), position=(0,-3,0))
cylinder.rotate_from_angax(np.linspace(0, 300, 40)[1:], 'z', anchor=0)
```

### Matplotlib backend

```{code-cell} ipython3
magpy.show(loop, cylinder, backend='matplotlib')
```

### Plotly backend

```{code-cell} ipython3
magpy.show(loop, cylinder, backend='plotly')
```

+++ {"tags": []}

### Pyvista backend

```{code-cell} ipython3
import pyvista as pv

pv.set_jupyter_backend('panel') # improve rendering in a jupyter notebook

magpy.show(loop, cylinder, backend='pyvista')
```

### Mayavi backend

```{code-cell} ipython3
# Allow rendering in a jupyter notebook - (not necessary in a python script)
from mayavi import mlab
mlab.init_notebook()

fig = magpy.show(loop, cylinder, return_fig=True, backend='mayavi') #`return_fig` not necessary in a python script
fig
```

## Output in custom figure

When calling `show`, a figure is automatically generated and displayed. It is also possible to display the `show` output on a given user-defined canvas with the `canvas` kwarg.

In the following example we show how to combine a 2D field plot with the 3D `show` output in **Matplotlib**:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy

# setup matplotlib figure and subplots
fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(121,)                  # 2D-axis
ax2 = fig.add_subplot(122, projection="3d")  # 3D-axis

# define sources and paths
loop = magpy.current.Loop(current=1, diameter=1)
loop.position = np.linspace((0,0,-3), (0,0,3), 40)

cylinder = magpy.magnet.Cylinder(magnetization=(0,-100,0), dimension=(1,2), position=(0,-3,0))
cylinder.rotate_from_angax(np.linspace(0, 300, 40)[1:], 'z', anchor=0)

# compute field and plot in 2D-axis
B = magpy.getB([loop, cylinder], (0,0,0), sumup=True)
ax1.plot(B)

# display show() output in 3D-axis
magpy.show(loop, cylinder, canvas=ax2)

# generate figure
plt.tight_layout()
plt.show()
```

A similar example with **Plotly**:

```{code-cell} ipython3
import numpy as np
import plotly.graph_objects as go
import magpylib as magpy

# setup plotly figure and subplots
fig = go.Figure().set_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "scene"}]])

# define sources and paths
loop = magpy.current.Loop(current=1, diameter=1)
loop.position = np.linspace((0,0,-3), (0,0,3), 40)

cylinder = magpy.magnet.Cylinder(magnetization=(0,-100,0), dimension=(1,2), position=(0,-3,0))
cylinder.rotate_from_angax(np.linspace(0, 300, 40)[1:], 'z', anchor=0)

# compute field and plot in 2D-axis
B = magpy.getB([loop, cylinder], (0,0,0), sumup=True)
for i,lab in enumerate(['Bx', 'By', 'Bz']):
    fig.add_trace(go.Scatter(x=np.linspace(0,1,40), y=B[:,i], name=lab))

# display show() output in 3D-axis
temp_fig = go.Figure()
magpy.show(loop, cylinder, canvas=temp_fig, backend='plotly')
fig.add_traces(temp_fig.data, rows=1, cols=2)
fig.layout.scene.update(temp_fig.layout.scene)

# generate figure
fig.show()
```

An example with **Pyvista**:

```{code-cell} ipython3
:tags: []

import numpy as np
import magpylib as magpy
import pyvista as pv

pv.set_jupyter_backend('panel') # improve rending in a jupyter notebook

# define sources and paths
loop = magpy.current.Loop(current=1, diameter=5)
loop.position = np.linspace((0,0,-3), (0,0,3), 40)

cylinder = magpy.magnet.Cylinder(magnetization=(0,-100,0), dimension=(1,2), position=(0,-3,0))
cylinder.rotate_from_angax(np.linspace(0, 300, 40)[1:], 'z', anchor=0)

# create a pyvista plotting scene with some graphs
pl = pv.Plotter()
line = np.array([(t*np.cos(15*t), t*np.sin(15*t), t-8) for t in np.linspace(3,5,200)])
pl.add_lines(line, color='black')

# add magpylib.show() output to existing scene
magpy.show(loop, cylinder, backend='pyvista', canvas=pl)

# display scene
pl.camera.position=(50, 10, 10)
#pl.set_background("white")
pl.show()
```

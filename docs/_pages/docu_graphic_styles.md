---
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

# Graphic and Styles

(intro-graphic-output)=

## Graphic output

Once all Magpylib objects and their paths have been created, **`show`** provides a convenient way to graphically display the geometric arrangement using the Matplotlib (default) and Plotly packages. When `show` is called, it generates a new figure which is then automatically displayed.

The desired graphic backend is selected with the `backend` keyword argument. To bring the output to a given, user-defined figure, the `canvas` kwarg is used. This is demonstrated in {ref}`examples-backends-canvas`.

The following example shows the graphical representation of various Magpylib objects and their paths using the default Matplotlib graphic backend.

```{code-cell} ipython3
import numpy as np
import magpylib as magpy
from magpylib.magnet import Cuboid, Cylinder, CylinderSegment, Sphere, Tetrahedron, TriangularMesh
from magpylib.current import Loop, Line
from magpylib.misc import Dipole, Triangle
import pyvista as pv

objects = [
    Cuboid(
        magnetization=(0,-100,0),
        dimension=(1,1,1),
        position=(-6,0,0),
    ),
    Cylinder(
        magnetization=(0,0,100),
        dimension=(1,1),
        position=(-5,0,0),
    ),
    CylinderSegment(
        magnetization=(0,0,100),
        dimension=(.3,1,1,0,140),
        position=(-3,0,0),
    ),
    Sphere(
        magnetization=(0,0,100),
        diameter=1,
        position=(-1,0,0),
    ),
    Tetrahedron(
        magnetization=(0,0,100),
        vertices=((-1,0,0), (1,0,0), (0,-1,0), (0,-1,-1)),
        position=(-4,0,4)
    ),
    Loop(
        current=1,
        diameter=1,
        position=(4,0,0),
    ),
    Line(
        current=1,
        vertices=[(1,0,0), (0,1,0), (-1,0,0), (0,-1,0), (1,0,0)],
        position=(1,0,0),
    ),
    Dipole(
        moment=(0,0,100),
        position=(3,0,0),
    ),
    Triangle(
        magnetization=(0,0,100),
        vertices=((-1,0,0), (1,0,0), (0,1,0)),
        position=(2,0,4),
    ),
    TriangularMesh.from_pyvista(
        magnetization=(0,0,100),
        polydata=pv.Dodecahedron(),
        position=(-1,0,4),
    ),
    magpy.Sensor(
        pixel=[(0,0,z) for z in (-.5,0,.5)],
        position=(0,-3,0),
    ),
]

objects[5].move(np.linspace((0,0,0), (0,0,7), 20))
objects[0].rotate_from_angax(np.linspace(0, 90, 20), 'z', anchor=0)

magpy.show(objects)
```

Notice that, objects and their paths are automatically assigned different colors, the magnetization vector, current directions and dipole objects are indicated by arrows and sensors are shown as tri-colored coordinate cross with pixel as markers.

How objects are represented graphically (color, line thickness, ect.) is defined by the **style**. The default style, which can be seen above, is accessed and manipulated through `magpy.defaults.display.style`. In addition, each object can have an individual style, which takes precedence over the default setting. A local style override is also possible by passing style arguments directly to `show`.

Some practical ways to set styles are shown in the next example:

```{code-cell} ipython3
import magpylib as magpy
from magpylib.magnet import Cuboid

cube1 = Cuboid(magnetization=(0,0,1), dimension=(2,4,4))
cube2 = cube1.copy(position=(3,0,0))
cube3 = cube1.copy(position=(6,0,0))

# change the default
magpy.defaults.display.style.base.color = 'crimson'

# set individual style through properties
cube2.style.color = 'orangered'

# set individual style using update with style dictionary
cube3.style.update({'color': 'gold'})

# set individual style at initialization with underscore magic
cube4 = cube1.copy(position=(9,0,0), style_color='linen')

# show with local style override
magpy.show(cube1, cube2, cube3, cube4, style_magnetization_show=False)
```

The hierarchy that decides about the final graphic object representation, a list of all style parameters and other options for tuning the `show`-output are described in {ref}`examples-graphic-styles` and {ref}`examples-animation`.


(examples-backends-canvas)=

# Graphics - Backend, canvas, return_fig

The graphic backend refers to the plotting library that is used for graphic output. Canvas refers to the frame/window/canvas/axes object the graphic output is forwarded to.

## Graphic backend

Magpylib supports several common graphic backends.

```{code-cell} ipython3
from magpylib import SUPPORTED_PLOTTING_BACKENDS

SUPPORTED_PLOTTING_BACKENDS
```

+++ {"user_expressions": []}

The installation default is set to `'auto'`. In this case the backend is dynamically inferred depending on the current running environment (terminal or notebook), the available installed backend libraries and the set canvas:

| environment      | canvas                                            | inferred backend                        |
|------------------|---------------------------------------------------|-----------------------------------------|
| terminal         | `None`                                            | `matplotlib`                            |
| IPython notebook | `None`                                            | `plotly` if installed else `matplotlib` |
| all              | `matplotlib.axes.Axes`                            | `matplotlib`                            |
| all              | `plotly.graph_objects.Figure` (or `FigureWidget`) | `plotly`                                |
| all              | `pyvista.Plotter`                                 | `pyvista`                               |

To explicitly select a graphic backend one can
1. Change the library default with `magpy.defaults.display.backend = 'plotly'`.
2. Set the `backend` kwarg in the `show` function, `show(..., backend='matplotlib')`.

There is a high level of **feature parity**, however, not all graphic features are supported by all backends. In addition, some common Matplotlib syntax (e.g. color `'r'`, linestyle `':'`) is automatically translated to other backends.

The following example demonstrates the currently supported backends:

```{code-cell} ipython3
import magpylib as magpy
import numpy as np
import pyvista as pv

pv.set_jupyter_backend("panel")  # improve rendering in a jupyter notebook

# define sources and paths
loop = magpy.current.Loop(current=1, diameter=1)
loop.position = np.linspace((0, 0, -3), (0, 0, 3), 40)

cylinder = magpy.magnet.Cylinder(
    magnetization=(0, -100, 0), dimension=(1, 2), position=(0, -3, 0)
)
cylinder.rotate_from_angax(np.linspace(0, 300, 40)[1:], "z", anchor=0)

# show the system using different backends
for backend in magpy.SUPPORTED_PLOTTING_BACKENDS:
    print(f"Plotting backend: {backend!r}")
    magpy.show(loop, cylinder, backend=backend)
```

## Output in custom figure

When calling `show`, a figure is automatically generated and displayed. It is also possible to display the `show` output on a given user-defined canvas with the `canvas` kwarg.

In the following example we show how to combine a 2D field plot with the 3D `show` output in **Matplotlib**:

```{code-cell} ipython3
import magpylib as magpy
import matplotlib.pyplot as plt
import numpy as np

# setup matplotlib figure and subplots
fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(
    121,
)  # 2D-axis
ax2 = fig.add_subplot(122, projection="3d")  # 3D-axis

# define sources and paths
loop = magpy.current.Loop(current=1, diameter=1)
loop.position = np.linspace((0, 0, -3), (0, 0, 3), 40)

cylinder = magpy.magnet.Cylinder(
    magnetization=(0, -100, 0), dimension=(1, 2), position=(0, -3, 0)
)
cylinder.rotate_from_angax(np.linspace(0, 300, 40)[1:], "z", anchor=0)

# compute field and plot in 2D-axis
B = magpy.getB([loop, cylinder], (0, 0, 0), sumup=True)
ax1.plot(B)

# display show() output in 3D-axis
magpy.show(loop, cylinder, canvas=ax2)

# generate figure
plt.tight_layout()
plt.show()
```

A similar example with **Plotly**:

```{code-cell} ipython3
import magpylib as magpy
import numpy as np
import plotly.graph_objects as go

# setup plotly figure and subplots
fig = go.Figure().set_subplots(
    rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "scene"}]]
)

# define sources and paths
loop = magpy.current.Loop(current=1, diameter=1)
loop.position = np.linspace((0, 0, -3), (0, 0, 3), 40)

cylinder = magpy.magnet.Cylinder(
    magnetization=(0, -100, 0), dimension=(1, 2), position=(0, -3, 0)
)
cylinder.rotate_from_angax(np.linspace(0, 300, 40)[1:], "z", anchor=0)

# compute field and plot in 2D-axis
B = magpy.getB([loop, cylinder], (0, 0, 0), sumup=True)
for i, lab in enumerate(["Bx", "By", "Bz"]):
    fig.add_trace(go.Scatter(x=np.linspace(0, 1, 40), y=B[:, i], name=lab))

# display show() output in 3D-axis
temp_fig = go.Figure()
magpy.show(loop, cylinder, canvas=temp_fig, backend="plotly")
fig.add_traces(temp_fig.data, rows=1, cols=2)
fig.layout.scene.update(temp_fig.layout.scene)

# generate figure
fig.show()
```

An example with **Pyvista**:

```{code-cell} ipython3
import magpylib as magpy
import numpy as np
import pyvista as pv

pv.set_jupyter_backend("panel")  # improve rending in a jupyter notebook

# define sources and paths
loop = magpy.current.Loop(current=1, diameter=5)
loop.position = np.linspace((0, 0, -3), (0, 0, 3), 40)

cylinder = magpy.magnet.Cylinder(
    magnetization=(0, -100, 0), dimension=(1, 2), position=(0, -3, 0)
)
cylinder.rotate_from_angax(np.linspace(0, 300, 40)[1:], "z", anchor=0)

# create a pyvista plotting scene with some graphs
pl = pv.Plotter()
line = np.array(
    [(t * np.cos(15 * t), t * np.sin(15 * t), t - 8) for t in np.linspace(3, 5, 200)]
)
pl.add_lines(line, color="black")

# add magpylib.show() output to existing scene
magpy.show(loop, cylinder, backend="pyvista", canvas=pl)

# display scene
pl.camera.position = (50, 10, 10)
pl.set_background("black", top="white")
pl.show()
```

## Return figure

Instead of forwarding a figure to an existing canvas, it is also possible to return the figure object for further manipulation using the `return_fig` command. In the following example this is demonstrated for the pyvista backend.

```{code-cell} ipython3
import magpylib as magpy
import numpy as np
import pyvista as pv

pv.set_jupyter_backend("panel")  # improve rending in a jupyter notebook

# define sources and paths
loop = magpy.current.Loop(current=1, diameter=5)
loop.position = np.linspace((0, 0, -3), (0, 0, 3), 40)

cylinder = magpy.magnet.Cylinder(
    magnetization=(0, -100, 0), dimension=(1, 2), position=(0, -3, 0)
)
cylinder.rotate_from_angax(np.linspace(0, 300, 40)[1:], "z", anchor=0)

# return pyvista scene from magpylib.show()
pl = magpy.show(loop, cylinder, backend="pyvista", return_fig=True)

# add line to the pyvista scene
line = np.array(
    [(t * np.cos(15 * t), t * np.sin(15 * t), t - 8) for t in np.linspace(3, 5, 200)]
)
pl.add_lines(line, color="black")

# display scene
pl.camera.position = (50, 10, 10)
pl.set_background("purple", top="lightgreen")
pl.enable_anti_aliasing("ssaa")
pl.show()
```



(examples-graphic-styles)=
# Graphics - Styles

The graphic styles define how Magpylib objects are displayed visually when calling `show`. They can be fine-tuned and individualized in many ways.

There are multiple hierarchy levels that decide about the final graphical representation of the objects:

1. When no input is given, the **default style** will be applied.
2. Collections will override the color property of all children with their own color.
3. Object **individual styles** will take precedence over these values.
4. Setting a **local style** in `show()` will take precedence over all other settings.

## Setting the default style

The default style is stored in `magpylib.defaults.display.style`. Default styles can be set as properties,

```python
magpy.defaults.display.style.magnet.magnetization.show = True
magpy.defaults.display.style.magnet.magnetization.color.middle = 'grey'
magpy.defaults.display.style.magnet.magnetization.color.mode = 'bicolor'
```

by assigning a style dictionary with equivalent keys,

```python
magpy.defaults.display.style.magnet = {
    'magnetization': {'show': True, 'color': {'middle': 'grey', 'mode': 'tricolor'}}
}
```

or by making use of the `update` method:

```python
magpy.defaults.display.style.magnet.magnetization.update(
    'show': True,
    'color': {'middle'='grey', mode='tricolor',}
)
```

All three examples result in the same default style.

Once modified, the library default can always be restored with the `magpylib.style.reset()` method. The following practical example demonstrates how to create and set a user defined magnetization style as default,

```{code-cell} ipython3
import magpylib as magpy
from magpylib.magnet import Cuboid, Cylinder, Sphere

cube = Cuboid(magnetization=(1, 0, 0), dimension=(1, 1, 1))
cylinder = Cylinder(magnetization=(0, 1, 0), dimension=(1, 1), position=(2,0,0))
sphere = Sphere(magnetization=(0, 1, 1), diameter=1, position=(4,0,0))

print('Default magnetization style')
magpy.show(cube, cylinder, sphere, backend="plotly")

user_defined_style = {
    'show': True,
    "size": 0.5,
    'color': {
        'transition': 0,
        'mode': 'tricolor',
        'middle': 'white',
        'north': 'magenta',
        'south': 'turquoise',
    },
    "mode": "arrow+color",
}
magpy.defaults.display.style.magnet.magnetization = user_defined_style

print('Custom magnetization style')
magpy.show(cube, cylinder, sphere, backend="plotly")
```

## Magic underscore notation
<!-- +++ {"tags": [], "jp-MarkdownHeadingCollapsed": true} -->

To facilitate working with deeply nested properties, all style constructors and object style methods support the magic underscore notation. It enables referencing nested properties by joining together multiple property names with underscores. This feature mainly helps reduce the code verbosity and is heavily inspired by the `plotly` implementation (see [plotly underscore notation](https://plotly.com/python/creating-and-updating-figures/#magic-underscore-notation)).

With magic underscore notation, the previous examples can be written as,

```python
import magpylib as magpy
magpy.defaults.display.style.magnet = {
    'magnetization_show': True,
    'magnetization_color_middle': 'grey',
    'magnetization_color_mode': 'tricolor',
}
```

or directly as kwargs in the `update` method as,

```python
import magpylib as magpy
magpy.defaults.display.style.magnet.update(
    magnetization_show=True,
    magnetization_color_middle='grey',
    magnetization_color_mode='tricolor',
)
```

## Setting individual styles

Any Magpylib object can have its own individual style that will take precedence over the default values when `show` is called. When setting individual styles, the object family specifier such as `magnet` or `current` which is required for the defaults settings, but is implicitly defined by the object type, can be omitted.

```{warning}
Users should be aware that specifying individual style attributes massively increases object initializing time (from <50 to 100-500 $\mu$s).
While this may not be noticeable for a small number of objects, it is best to avoid setting styles until it is plotting time.
```

In the following example the individual style of `cube` is set at initialization, the style of `cylinder` is the default one, and the individual style of `sphere` is set using the object style properties.

```{code-cell} ipython3
import magpylib as magpy
from magpylib.magnet import Cuboid, Cylinder, Sphere

magpy.defaults.reset() # reset defaults defined in previous example

cube = Cuboid(
    magnetization=(1, 0, 0),
    dimension=(1, 1, 1),
    style_magnetization_color_mode='tricycle',
)
cylinder = Cylinder(
    magnetization=(0, 1, 0),
    dimension=(1, 1), position=(2,0,0),
)
sphere = Sphere(
    magnetization=(0, 1, 1),
    diameter=1,
    position=(4,0,0),
)

sphere.style.magnetization.color.mode='bicolor'

magpy.show(cube, cylinder, sphere, backend="plotly")
```

## Setting style via collections

When displaying collections, the collection object `color` property will be automatically assigned to all its children and override the default style. An example that demonstrates this is {ref}`examples-union-operation`. In addition, it is possible to modify the individual style properties of all children with the `set_children_styles` method. Non-matching properties are simply ignored.

In the following example we show how the french magnetization style is applied to all children in a collection,

```{code-cell} ipython3
import magpylib as magpy
from magpylib.magnet import Cuboid, Cylinder, Sphere

magpy.defaults.reset() # reset defaults defined in previous example

cube = Cuboid(magnetization=(1, 0, 0), dimension=(1, 1, 1))
cylinder = Cylinder(magnetization=(0, 1, 0), dimension=(1, 1), position=(2,0,0))
sphere = Sphere(magnetization=(0, 1, 1), diameter=1, position=(4,0,0))

coll = cube + cylinder

coll.set_children_styles(magnetization_color_south="blue")

magpy.show(coll, sphere, backend="plotly")
```

## Local style override

Finally it is possible to hand style input to the `show` function directly and locally override the given properties for this specific `show` output. Default or individual style attributes will not be modified. Such inputs must start with the `style` prefix and the object family specifier must be omitted. Naturally underscore magic is supported.

```{code-cell} ipython3
import magpylib as magpy
from magpylib.magnet import Cuboid, Cylinder, Sphere

cube = Cuboid(magnetization=(1, 0, 0), dimension=(1, 1, 1))
cylinder = Cylinder(magnetization=(0, 1, 0), dimension=(1, 1), position=(2,0,0))
sphere = Sphere(magnetization=(0, 1, 1), diameter=1, position=(4,0,0))

# use local style override
magpy.show(cube, cylinder, sphere, backend="plotly", style_magnetization_show=False)
```

(examples-list-of-styles)=

## List of styles

```{code-cell} ipython3
magpy.defaults.display.style.as_dict(flatten=True, separator='.')
```


(examples-3d-models)=

# Graphics - 3D models and CAD

(examples-own-3d-models)=
## Custom 3D models

Each Magpylib object has a default 3D representation that is displayed with `show`. Users can add a custom 3D model to any Magpylib object with help of the `style.model3d.add_trace` method. The new trace is stored in `style.model3d.data`. User-defined traces move with the object just like the default models do. The default trace can be hidden with the command `obj.model3d.showdefault=False`. When using the `'generic'` backend, custom traces are automatically translated into any other backend. If a specific backend is used, it will only show when called with the corresponding backend.

The input `trace` is a dictionary which includes all necessary information for plotting or a `magpylib.graphics.Trace3d` object. A `trace` dictionary has the following keys:

1. `'backend'`: `'generic'`, `'matplotlib'` or `'plotly'`
2. `'constructor'`: name of the plotting constructor from the respective backend, e.g. plotly `'Mesh3d'` or matplotlib `'plot_surface'`
3. `'args'`: default `None`, positional arguments handed to constructor
4. `'kwargs'`: default `None`, keyword arguments handed to constructor
5. `'coordsargs'`: tells magpylib which input corresponds to which coordinate direction, so that geometric representation becomes possible. By default `{'x': 'x', 'y': 'y', 'z': 'z'}` for the `'generic'` backend and Plotly backend,  and `{'x': 'args[0]', 'y': 'args[1]', 'z': 'args[2]'}` for the Matplotlib backend.
6. `'show'`: default `True`, toggle if this trace should be displayed
7. `'scale'`: default 1, object geometric scaling factor
8. `'updatefunc'`: default `None`, updates the trace parameters when `show` is called. Used to generate  dynamic traces.

The following example shows how a **generic** trace is constructed with  `Mesh3d` and `Scatter3d` and is displayed with three different backends:

```{code-cell} ipython3
import numpy as np
import magpylib as magpy
import pyvista as pv

pv.set_jupyter_backend('panel') # improve rendering in a jupyter notebook

# Mesh3d trace #########################

trace_mesh3d = {
    'backend': 'generic',
    'constructor': 'Mesh3d',
    'kwargs': {
        'x': (1, 0, -1, 0),
        'y': (-.5, 1.2, -.5, 0),
        'z': (-.5, -.5, -.5, 1),
        'i': (0, 0, 0, 1),
        'j': (1, 1, 2, 2),
        'k': (2, 3, 3, 3),
        #'opacity': 0.5,
    },
}
coll = magpy.Collection(position=(0,-3,0), style_label="'Mesh3d' trace")
coll.style.model3d.add_trace(trace_mesh3d)

# Scatter3d trace ######################

ts = np.linspace(0, 2*np.pi, 30)
trace_scatter3d = {
    'backend': 'generic',
    'constructor': 'Scatter3d',
    'kwargs': {
        'x': np.cos(ts),
        'y': np.zeros(30),
        'z': np.sin(ts),
        'mode': 'lines',
    }
}
dipole = magpy.misc.Dipole(moment=(0,0,1), style_label="'Scatter3d' trace", style_size=6)
dipole.style.model3d.add_trace(trace_scatter3d)

magpy.show(coll, dipole, backend='matplotlib')
magpy.show(coll, dipole, backend='plotly')
magpy.show(coll, dipole, backend='pyvista')
```

It is possible to have multiple user-defined traces that will be displayed at the same time. In addition, the following code shows how to quickly copy and manipulate trace dictionaries and `Trace3d` objects,

```{code-cell} ipython3
import copy
dipole.style.size=3

# generate new trace from dictionary
trace2 = copy.deepcopy(trace_scatter3d)
trace2['kwargs']['y'] = np.sin(ts)
trace2['kwargs']['z'] = np.zeros(30)

dipole.style.model3d.add_trace(trace2)

# generate new trace from Trace3d object
trace3 = dipole.style.model3d.data[1].copy()
trace3.kwargs['x'] = np.zeros(30)
trace3.kwargs['z'] = np.cos(ts)

dipole.style.model3d.add_trace(trace3)

dipole.show(dipole, backend='matplotlib')
```

**Matplotlib** plotting functions often use positional arguments for $(x,y,z)$ input, that are handed over from `args=(x,y,z)` in `trace`. The following examples show how to construct traces with `plot`, `plot_surface` and `plot_trisurf`:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import magpylib as magpy

# plot trace ###########################

ts = np.linspace(-10,10,100)
xs = np.cos(ts)
ys = np.sin(ts)
zs = ts/20

trace_plot = {
    'backend': 'matplotlib',
    'constructor': 'plot',
    'args': (xs,ys,zs),
    'kwargs': {'ls': '--', 'lw': 2},
}
magnet = magpy.magnet.Cylinder(magnetization=(0,0,1), dimension=(.5,1))
magnet.style.model3d.add_trace(trace_plot)

# plot_surface trace ###################

u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
xs = np.cos(u) * np.sin(v)
ys = np.sin(u) * np.sin(v)
zs = np.cos(v)

trace_surf = {
    'backend': 'matplotlib',
    'constructor': 'plot_surface',
    'args': (xs,ys,zs),
    'kwargs': {'cmap': plt.cm.YlGnBu_r},
}
ball = magpy.Collection(position=(-3,0,0))
ball.style.model3d.add_trace(trace_surf)

# plot_trisurf trace ###################

u, v = np.mgrid[0:2*np.pi:50j, -.5:.5:10j]
u, v = u.flatten(), v.flatten()

xs = (1 + 0.5 * v * np.cos(u / 2.0)) * np.cos(u)
ys = (1 + 0.5 * v * np.cos(u / 2.0)) * np.sin(u)
zs = 0.5 * v * np.sin(u / 2.0)

tri = mtri.Triangulation(u, v)

trace_trisurf = {
    'backend': 'matplotlib',
    'constructor': 'plot_trisurf',
    'args': (xs,ys,zs),
    'kwargs': {
        'triangles': tri.triangles,
        'cmap': plt.cm.coolwarm,
    },
}
mobius = magpy.misc.CustomSource(style_model3d_showdefault=False, position=(3,0,0))
mobius.style.model3d.add_trace(trace_trisurf)

magpy.show(magnet, ball, mobius, zoom=5, backend="matplotlib")
```

## Pre-defined 3D models

Automatic trace generators are provided for several basic 3D models in `magpylib.graphics.model3d`. If no backend is specified, it defaults back to `'generic'`. They can be used as follows,

```{code-cell} ipython3
import magpylib as magpy
from magpylib.graphics import model3d

# prism trace ###################################
trace_prism = model3d.make_Prism(
    base=6,
    diameter=2,
    height=1,
    position=(-3,0,0),
)
obj0 = magpy.Sensor(style_model3d_showdefault=False, style_label='Prism')
obj0.style.model3d.add_trace(trace_prism)

# pyramid trace #################################
trace_pyramid = model3d.make_Pyramid(
    base=30,
    diameter=2,
    height=1,
    position=(3,0,0),
)
obj1 = magpy.Sensor(style_model3d_showdefault=False, style_label='Pyramid')
obj1.style.model3d.add_trace(trace_pyramid)

# cuboid trace ##################################
trace_cuboid = model3d.make_Cuboid(
    dimension=(2,2,2),
    position=(0,3,0),
)
obj2 = magpy.Sensor(style_model3d_showdefault=False, style_label='Cuboid')
obj2.style.model3d.add_trace(trace_cuboid)

# cylinder segment trace ########################
trace_cylinder_segment = model3d.make_CylinderSegment(
    dimension=(1, 2, 1, 140, 220),
    position=(1,0,-3),
)
obj3 = magpy.Sensor(style_model3d_showdefault=False, style_label='Cylinder Segment')
obj3.style.model3d.add_trace(trace_cylinder_segment)

# ellipsoid trace ###############################
trace_ellipsoid = model3d.make_Ellipsoid(
    dimension=(2,2,2),
    position=(0,0,3),
)
obj4 = magpy.Sensor(style_model3d_showdefault=False, style_label='Ellipsoid')
obj4.style.model3d.add_trace(trace_ellipsoid)

# arrow trace ###################################
trace_arrow = model3d.make_Arrow(
    base=30,
    diameter=0.6,
    height=2,
    position=(0,-3,0),
)
obj5 = magpy.Sensor(style_model3d_showdefault=False, style_label='Arrow')
obj5.style.model3d.add_trace(trace_arrow)

magpy.show(obj0, obj1, obj2, obj3, obj4, obj5, backend='plotly')
```

(examples-adding-CAD-model)=

## Adding a CAD model

As shown in {ref}`examples-3d-models`, it is possible to attach custom 3D model representations to any Magpylib object. In the example below we show how a standard CAD model can be transformed into a generic Magpylib graphic trace, and displayed by both `matplotlib` and `plotly` backends.

```{note}
The code below requires installation of the `numpy-stl` package.
```

```{code-cell} ipython3
import os
import tempfile
import requests
import numpy as np
from stl import mesh  # requires installation of numpy-stl
import magpylib as magpy
from matplotlib.colors import to_hex


def bin_color_to_hex(x):
    """ transform binary rgb into hex color"""
    sb = f"{x:015b}"[::-1]
    r = int(sb[:5], base=2)/31
    g = int(sb[5:10], base=2)/31
    b = int(sb[10:15], base=2)/31
    return to_hex((r,g,b))



def trace_from_stl(stl_file):
    """
    Generates a Magpylib 3D model trace dictionary from an *.stl file.
    backend: 'matplotlib' or 'plotly'
    """
    # load stl file
    stl_mesh = mesh.Mesh.from_file(stl_file)

    # extract vertices and triangulation
    p, q, r = stl_mesh.vectors.shape
    vertices, ixr = np.unique(stl_mesh.vectors.reshape(p * q, r), return_inverse=True, axis=0)
    i = np.take(ixr, [3 * k for k in range(p)])
    j = np.take(ixr, [3 * k + 1 for k in range(p)])
    k = np.take(ixr, [3 * k + 2 for k in range(p)])
    x, y, z = vertices.T

    # generate and return a generic trace which can be translated into any backend
    colors = stl_mesh.attr.flatten()
    facecolor = np.array([bin_color_to_hex(c) for c in colors]).T
    trace = {
        'backend': 'generic',
        'constructor': 'mesh3d',
        'kwargs': dict(x=x, y=y, z=z, i=i, j=j, k=k, facecolor=facecolor),
    }
    return trace


# load stl file from online resource
url = "https://raw.githubusercontent.com/magpylib/magpylib-files/main/PG-SSO-3-2.stl"
file = url.split("/")[-1]
with tempfile.TemporaryDirectory() as temp:
    fn = os.path.join(temp, file)
    with open(fn, "wb") as f:
        response = requests.get(url)
        f.write(response.content)

    # create traces for both backends
    trace = trace_from_stl(fn)

# create sensor and add CAD model
sensor = magpy.Sensor(style_label='PG-SSO-3 package')
sensor.style.model3d.add_trace(trace)

# create magnet and sensor path
magnet = magpy.magnet.Cylinder(magnetization=(0,0,100), dimension=(15,20))
sensor.position = np.linspace((-15,0,8), (-15,0,-4), 21)
sensor.rotate_from_angax(np.linspace(0, 180, 21), 'z', anchor=0, start=0)

# display with matplotlib and plotly backends
args = (sensor, magnet)
kwargs = dict(style_path_frames=5)
magpy.show(args, **kwargs,  backend="matplotlib")
magpy.show(args, **kwargs, backend="plotly")
```


(examples-animation)=

# Graphics - Animate paths

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

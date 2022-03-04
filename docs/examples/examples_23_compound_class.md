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

(examples-own-dynamic-3d-model)=

# Own dynamic 3D model

+++

The Magpylib `Collection` object class serves the purpose of grouping multiple sources and/or sensors in a single object. This bares the advantage of manipulating multiple objects with single commands such as `move` and `rotate`. It can for example be used to create a compound object that acts as a unique source.
A `Collection` also have its own position and holds some basic styling properties that can be useful to modify the associated 3D-representation when plotting it with the `show` method.

+++

## Minimal example

+++

In the following example we will create a linear arrangement of cuboid magnets with alternating polarity. When calling the `getB` method on the created `Collection`, the result is already the sum of the field produced by its children.

```{code-cell} ipython3
import magpylib as magpy

# create children
cuboids = []
for index in range(10):
    mag_sign = index % 2 * 2 - 1  # positive if index is od else even
    cuboid = magpy.magnet.Cuboid(
        magnetization=(0, 0, 1000 * mag_sign),
        dimension=(10, 10, 10),
        position=(index * 10, 0, 0),
    )
    cuboids.append(cuboid)

# group children into a `Collection`
coll = magpy.Collection(cuboids[:-1])
print(f"B-Field at position (0,0,0) → {coll.getB((0,0,0)).round(2)}")
coll.show()
```

## Add an extra 3D-model

+++

Lets now add a body emcompassing our linear arrangement. We can do so by adding a new 3D-model with the `style.model3d.add_trace` method. If we intend only to use the plotly backend to display the system, only a `plotly` trace is necessary.

A model3d trace is a dictionary containing the necessary information to draw a plot type, depending on the chosen backend.

The `add_trace` method has the following signature:

```{code-cell} ipython3
help(magpy.Collection().style.model3d.add_trace)
```

In the case of building a `Collection` of many objects, it can become quite computationally expensive to display every single children. To avoid this issue, it is possible to deactivate the default representation of every children using the `set_children_styles` method.

```{code-cell} ipython3
import magpylib as magpy
from magpylib.display.plotly import make_BaseCuboid

# create children
cuboids = []
for index in range(11):
    mag_sign = index % 2 * 2 - 1  # positive if index is od else even
    cuboid = magpy.magnet.Cuboid(
        magnetization=(0, 0, 1000 * mag_sign),
        dimension=(10, 10, 10),
        position=(index * 10, 0, 0),
    )
    cuboids.append(cuboid)

# group children into a `Collection`
coll = magpy.Collection(cuboids[:-1], style_label='Collection with visible children')

# add extra 3D-trace - the make_BaseCuboid function returns a dictionary
plotly_trace = make_BaseCuboid(dimension=(104, 12, 12), position=(45, 0, 0))
plotly_trace["opacity"] = 0.5
coll.style.model3d.add_trace(trace=plotly_trace, backend="plotly")
coll.show(backend="plotly")

# Hide the children 3D-model representation
coll.set_children_styles(model3d_showdefault=False)
coll.style.label = 'Collection with hidden children'
coll.show(backend="plotly")
```

```{note}
The `Collection` position is set to (0,0,0) at creation time. Any added extra 3D-model will be bound to the local coordinate system of to the `Collection` and `rotated`/`moved` together with its parent object.
```

+++

```{warning}
If no 3D-model as been assigned to a `Collection` and all children representation are hidden, you may end up with an empty plot.
````

+++

## Subclassing `Collection`

+++

By subclassing the Magpylib `Collection` we can define special _compound_ objects that have their own new properties and methods. In the following example we build a _magnetic wheel_ object for which the `diameter`, `cube_size` and number of children `cubes` can be updated for each `MagneticWheel` instance directly.

```{code-cell} ipython3
import magpylib as magpy
import numpy as np


# define the new `Collection` subclass
class MagneticWheel(magpy.Collection):
    """creates a basic Collection Compound object with a rotary arrangement of cuboid magnets"""

    def __init__(self, cubes=6, cube_size=10, diameter=36, **style_kwargs):
        super().__init__(**style_kwargs)
        self.update(cubes=cubes, cube_size=cube_size, diameter=diameter)

    def update(self, cubes=None, cube_size=None, diameter=None):
        """updates the magnetic weel object"""
        self.reset_path()
        self.cubes = cubes if cubes is not None else self.cubes
        self.cube_size = cube_size if cube_size is not None else self.cube_size
        self.diameter = diameter if diameter is not None else self.diameter
        create_cube = lambda: magpy.magnet.Cuboid(
            magnetization=(1000, 0, 0),
            dimension=[self.cube_size] * 3,
            position=(self.diameter / 2, 0, 0),
        )
        ref_cube = create_cube().rotate_from_angax(
            np.linspace(0.0, 360.0, self.cubes, endpoint=False),
            "z",
            anchor=(0, 0, 0),
            start=0,
        )
        children = []
        for ind in range(self.cubes):
            s = create_cube()
            s.position = ref_cube.position[ind]
            s.orientation = ref_cube.orientation[ind]
            children.append(s)
        self.children = children
        return self


# create a `MagneticWeel` class instance
wheel = MagneticWheel()
sens = magpy.Sensor(position=(0, 0, 20))
print(f"B-Field at position {sens.position} → {wheel.getB(sens).round(2)}")
magpy.show(wheel, sens)

# update wheel
wheel.update(cube_size=5, diameter=50, cubes=10)
print(
    f"\nB-Field at position {sens.position} for updated wheel → {wheel.getB(sens).round(2)}"
)
magpy.show(wheel, sens)
```

## Use dynamic extra 3D-model update

+++

As shown previously we can aslo add an extra 3D-model to our new `MagneticWheel` class, since it inherits all methods and properties from the parent `Collection` class. However, if we specify a fixed model from a dictionary the trace will not adapt to the parameters we define in our `update` method or via setting attributes such as `diameter`. To solve this issue, the trace constructor of the `style.model3d` property also accepts callables as argument. This allows the trace to be updated created from dynamic parameters, or in this case, class attributes. This features avoids the need to recreate the 3D-model entirely, any time an attribute of the class has been updated. The actual trace building computation cost will only be due if we choose to display the object.
In the following example, both `matplotlib` and `plotly` backends are made compatible with the `MagneticWheel` class.

```{code-cell} ipython3
from functools import partial

import magpylib as magpy
import numpy as np
import plotly.graph_objects as go
from magpylib.display.plotly import make_BaseCylinderSegment


class MagneticWheel(magpy.Collection):
    """creates a basic Collection Compound object with a rotary arrangement of cuboid magnets"""

    def __init__(self, cubes=6, cube_size=10, diameter=36, **style_kwargs):
        super().__init__(**style_kwargs)
        self.update(cubes=cubes, cube_size=cube_size, diameter=diameter)
        self.style.model3d.add_trace(
            backend="plotly",
            trace=partial(self.get_trace3d, backend="plotly"),
            show=True,
        )
        self.style.model3d.add_trace(
            backend="matplotlib",
            trace=partial(self.get_trace3d, backend="matplotlib"),
            show=True,
            coordsargs={"x": "args[0]", "y": "args[1]", "z": "args[2]"},
        )

    def update(self, cubes=None, cube_size=None, diameter=None):
        """updates the magnetic weel object"""
        self.reset_path()
        self.cubes = cubes if cubes is not None else self.cubes
        self.cube_size = cube_size if cube_size is not None else self.cube_size
        self.diameter = diameter if diameter is not None else self.diameter
        create_cube = lambda: magpy.magnet.Cuboid(
            magnetization=(1, 0, 0),
            dimension=[self.cube_size] * 3,
            position=(self.diameter / 2, 0, 0),
        )
        ref_cube = create_cube().rotate_from_angax(
            np.linspace(0.0, 360.0, self.cubes, endpoint=False),
            "z",
            anchor=(0, 0, 0),
            start=0,
        )
        children = []
        for ind in range(self.cubes):
            s = create_cube()
            s.position = ref_cube.position[ind]
            s.orientation = ref_cube.orientation[ind]
            children.append(s)
        self.children = children
        return self

    def get_trace3d(self, backend):
        """creates dynamically the 3D-model considering class properties"""
        trace_plotly = make_BaseCylinderSegment(
            r1=max(self.diameter / 2 - self.cube_size, 0),
            r2=self.diameter / 2 + self.cube_size,
            h=self.cube_size,
            phi1=0,
            phi2=360,
        )
        opacity = 0.5
        color = "blue"
        trace_plotly = {**trace_plotly, **{"opacity": opacity}}
        if backend == "plotly":
            return trace_plotly
        elif backend == "matplotlib":
            x, y, z, i, j, k = [trace_plotly[k] for k in "xyzijk"]
            triangles = np.array([i, j, k]).T
            trace_mpl = dict(type="plot_trisurf", args=(x, y, z), triangles=triangles)
            return {**trace_mpl, **{"alpha": opacity}}

# create a few `MagneticWheel`instances and manipulate them
wheels = []
for ind, cubes in enumerate((3, 6, 12)):
    diameter = cubes * 5
    wheel = MagneticWheel(
        cubes=cubes,
        cube_size=10,
        diameter=diameter,
        style_label=f"Magnetic Wheel {ind+1}",
    )
    wheel.move((diameter, 0, -diameter * 5))
    wheel.rotate_from_angax(90, "x")
    wheel.rotate_from_angax(
        np.linspace(0, 360, 50, endpoint=False), "z", start=0, anchor=0
    )
    wheel.set_children_styles(path_show=False)
    wheels.append(wheel)

# display the resulting systems in our own canvas with matplotlib
magpy.show(wheels, style_path_frames=6)

# animate the resulting systems in our own canvas with plotly
fig = go.Figure()
magpy.show(wheels, canvas=fig, backend="plotly", animation=2)
fig.update_layout(
    title_text="Magnetic Wheels",
    height=800,
)
```

```{note}
The `magpylib.display.plotly` module incorporates a collection of functions which return dictionaries with the necessary information to build basic geometries as 3D-mesh objects. The outputs can also be used by the `matplotlib` libraray via the `plot_trisurf` command with adapting the syntax as shown in the example above.
```

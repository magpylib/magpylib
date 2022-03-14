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

# Advanced compounds

This tutorial brings the *compound philisopy* of collections to the next level by subclassing the `Collection` class and adding a dynamic 3D representation.

## Efficient 3D models

The Matplotlib and Plotly libraries were not designed for complex 3D graphic outputs. As a result, it becomes often inconvenient and slow when attempting to display many 3D objects. One solution to this problem when dealing with large collections is to represent the latter by a single encompassing body, and to deactivate the individual 3D models of all children. This is demonstrated in the following example,

```{code-cell} ipython3
import magpylib as magpy

# create collection
coll = magpy.Collection()
for index in range(10):
    cuboid = magpy.magnet.Cuboid(
        magnetization=(0, 0, 1000 * (index%2-.5)),
        dimension=(10,10,10),
        position=(index*10,0,0),
    )
    coll.add(cuboid)

# add 3D-trace
plotly_trace = magpy.graphics.model3d.make_Cuboid(
    backend='plotly',
    dimension=(104, 12, 12),
    position=(45, 0, 0),
)
plotly_trace["kwargs"]["opacity"] = 0.5
coll.style.model3d.add_trace(plotly_trace)

coll.style.label='Collection with visible children'
coll.show(backend="plotly")

# hide the children deafult 3D representation
coll.set_children_styles(model3d_showdefault=False)
coll.style.label = 'Collection with hidden children'
coll.show(backend="plotly")
```

```{note}
The `Collection` position is set to (0,0,0) at creation time. Any added extra 3D-model will be bound to the local coordinate system of to the `Collection` and `rotated`/`moved` together with its parent object.
```

## Subclassing collections

By subclassing the Magpylib `Collection` we can define special _compound_ objects that have their own new properties and methods. In the following example we build a _magnetic ring_ object which is simply a ring of cuboid magnets. It has the `cubes` property which refers to the number of cuboids in the ring, and can be dynamically updated, while the `MagnetRing` object itself behaves like a native Magpylib source,

```{code-cell} ipython3
import magpylib as magpy

class MagnetRing(magpy.Collection):
    """ A ring of cuboid magnets

    Parameters
    ----------
    cubes: int, default=6
        Number of cubes on ring
    """

    def __init__(self, cubes=6, **style_kwargs):
        super().__init__(**style_kwargs)
        self._update(cubes)

    @property
    def cubes(self):
        """ Number of cubes"""
        return self._cubes

    @cubes.setter
    def cubes(self, inp):
        """ set cubes"""
        self._update(inp)

    def _update(self, cubes):
        """updates the MagnetRing instance"""
        self._cubes = cubes
        ring_radius = cubes/3

        # construct MagnetRing in temporary Collection
        temp_coll = magpy.Collection()
        for i in range(cubes):
            child = magpy.magnet.Cuboid(
                magnetization=(1000,0,0),
                dimension=(1,1,1),
                position=(ring_radius,0,0)
            )
            child.rotate_from_angax(360/cubes*i, 'z', anchor=0)
            temp_coll.add(child)

        # adjust position/orientation and replace children
        temp_coll.position = self.position
        temp_coll.orientation = self.orientation
        self.children = temp_coll.children

        return self

# add a sensor
sensor = magpy.Sensor(position=(0, 0, 0))

# create a MagnetRing class instance
ring = MagnetRing()

# treat the Magnetic ring like a native magpylib source object
ring.position = (0,0,10)
ring.rotate_from_angax(angle=45, axis=(1,-1,0))
print(f"B-field at sensor → {ring.getB(sensor).round(2)}")
magpy.show(ring, sensor)

# modify object custom attribute
ring.cubes=15
print(f"B-field at sensor for modified ring → {ring.getB(sensor).round(2)}")
magpy.show(ring, sensor)
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

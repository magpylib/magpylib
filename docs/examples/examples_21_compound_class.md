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

(examples-own-dynamic-3d-model)=

# Advanced compounds

This tutorial brings the *compound philosophy* of collections to the next level by subclassing the `Collection` class and adding a dynamic 3D representation.

## Efficient 3D models

The Matplotlib and Plotly libraries were not designed for complex 3D graphic outputs. As a result, it becomes often inconvenient and slow when attempting to display many 3D objects. One solution to this problem when dealing with large collections, is to represent the latter by a single encompassing body, and to deactivate the individual 3D models of all children. This is demonstrated in the following example.

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
    backend='matplotlib',
    dimension=(104, 12, 12),
    position=(45, 0, 0),
    alpha=0.5,
)
coll.style.model3d.add_trace(plotly_trace)

coll.style.label='Collection with visible children'
coll.show()

# hide the children default 3D representation
coll.set_children_styles(model3d_showdefault=False)
coll.style.label = 'Collection with hidden children'
coll.show()
```

```{note}
The `Collection` position is set to (0,0,0) at creation time. Any added extra 3D-model will be bound to the local coordinate system of to the `Collection` and `rotated`/`moved` together with its parent object.
```

## Subclassing collections

By subclassing the Magpylib `Collection`, we can define special _compound_ objects that have their own new properties, methods and 3d trace. In the following example we build a _magnetic ring_ object which is simply a ring of cuboid magnets. It has the `cubes` property which refers to the number of cuboids in the ring and can be dynamically updated. The `MagnetRing` object itself behaves like a native Magpylib source.

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
        self.update(cubes)

    @property
    def cubes(self):
        """ Number of cubes"""
        return self._cubes

    @cubes.setter
    def cubes(self, inp):
        """ set cubes"""
        self.update(inp)

    def update(self, cubes):
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

        # add parameter-dependent 3d trace
        self.style.model3d.data = []
        self.style.model3d.add_trace(self.create_trace3d('plotly'))
        self.style.model3d.add_trace(self.create_trace3d('matplotlib'))

        return self

    def create_trace3d(self, backend):
        """ creates a parameter-dependent 3d model"""
        r1 = self.cubes/3 - .6
        r2 = self.cubes/3 + 0.6
        trace = magpy.graphics.model3d.make_CylinderSegment(
            backend=backend,
            dimension=(r1, r2, 1.1, 0, 360)
        )
        if backend=='plotly':
            trace['kwargs']['opacity'] = 0.5
        return trace

# add a sensor
sensor = magpy.Sensor(position=(0, 0, 0))

# create a MagnetRing class instance
ring = MagnetRing()

# treat the Magnetic ring like a native magpylib source object
ring.position = (0,0,10)
ring.rotate_from_angax(angle=45, axis=(1,-1,0))
print(f"B-field at sensor → {ring.getB(sensor).round(2)}")
magpy.show(ring, sensor, backend='plotly')

# modify object custom attribute
ring.cubes=15
print(f"B-field at sensor for modified ring → {ring.getB(sensor).round(2)}")
magpy.show(ring, sensor, backend='plotly')
```

## Postponed trace construction

Custom traces might be computationally costly to construct, and in the above example they are recomputed every time a parameter is changed. This can lead to quite some unwanted overhead, as the construction is only necessary once `show` is called.

To make your classes ready for heavy computation, it is possible to provide a callable as a trace, which will only be constructed when `show` is called. The following modification of the above example demonstrates this. All we do is to remove the trace from the `update` method, and instead provide `create_trace3d` as callable `model3d`.

```{code-cell} ipython3
from functools import partial
import magpylib as magpy
import numpy as np

class MagnetRing(magpy.Collection):
    """ A ring of cuboid magnets

    Parameters
    ----------
    cubes: int, default=6
        Number of cubes on ring
    """

    def __init__(self, cubes=6, **style_kwargs):
        super().__init__(**style_kwargs)
        self.update(cubes)
        self.style.model3d.add_trace(partial(self.create_trace3d, 'plotly'))
        self.style.model3d.add_trace(partial(self.create_trace3d, 'matplotlib'))

    @property
    def cubes(self):
        """ Number of cubes"""
        return self._cubes

    @cubes.setter
    def cubes(self, inp):
        """ set cubes"""
        self.update(inp)

    def update(self, cubes):
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

    def create_trace3d(self, backend):
        """ creates a parameter-dependent 3d model"""
        r1 = self.cubes/3 - .6
        r2 = self.cubes/3 + 0.6
        trace = magpy.graphics.model3d.make_CylinderSegment(
            backend=backend,
            dimension=(r1, r2, 1.1, 0, 360),
            **{('opacity' if backend=='plotly' else 'alpha') :0.5}
        )
        return trace

# create multiple `MagnetRing` instances and animate paths
rings = []
for i,cub in zip([2,7,12,17,22], [20,16,12,8,4]):
    ring = MagnetRing(cubes=cub, style_label=f'MagnetRing (x{cub})')
    ring.rotate_from_angax(angle=np.linspace(0,45,10), axis=(1,-1,0))
    ring.move(np.linspace((0,0,0), (-i,-i,i), i))
    rings.append(ring)

magpy.show(rings, animation=2, backend='plotly', style_path_show=False)
```

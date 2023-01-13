---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(examples-compounds)=

# Compounds - Collection sub-classing

The `Collection` class is a powerful tool for grouping and tracking object assemblies.
However, in many cases it is convenient to have assembly variables themselves (e.g. geometric arrangement) as class properties of new custom classes, which is achieved by sub-classing `Collection`. We refer to such super-classes as **compounds** and show how to seamlessly integrate them into Magpylib.

## Subclassing collections

In the following example we design a compound class `MagnetRing` which represents a ring of cuboid magnets with the parameter `cubes` that should refer to the number of magnets on the ring. The ring will automatically adjust its size when `cubes` is modified. In the spirit of {ref}`examples-collections-efficient` we also add an encompassing 3D model.

```{code-cell} ipython3
import magpylib as magpy

class MagnetRing(magpy.Collection):
    """ A ring of cuboid magnets

    Parameters
    ----------
    cubes: int, default=6
        Number of cubes on ring.
    """

    def __init__(self, cubes=6, **style_kwargs):
        super().__init__(**style_kwargs)             # hand over style args
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

        # construct in temporary Collection for path transfer
        temp_coll = magpy.Collection()
        for i in range(cubes):
            child = magpy.magnet.Cuboid(
                magnetization=(1000,0,0),
                dimension=(1,1,1),
                position=(ring_radius,0,0)
            )
            child.rotate_from_angax(360/cubes*i, 'z', anchor=0)
            temp_coll.add(child)

        # transfer path and children
        temp_coll.position = self.position
        temp_coll.orientation = self.orientation
        self.children = temp_coll.children

        # add parameter-dependent 3d trace
        self.style.model3d.data = []
        self.style.model3d.add_trace(self._custom_trace3d())

        return self

    def _custom_trace3d(self):
        """ creates a parameter-dependent 3d model"""
        r1 = self.cubes/3 - .6
        r2 = self.cubes/3 + 0.6
        trace = magpy.graphics.model3d.make_CylinderSegment(
            dimension=(r1, r2, 1.1, 0, 360),
            vert=150,
            opacity=0.5,
        )
        return trace
```

The new `MagnetRing` objects will seamlessly integrate into Magpylib and make use of the position and orientation interface, field computation and graphic display.

```{code-cell} ipython3
# add a sensor
sensor = magpy.Sensor(position=(0, 0, 0))

# create a MagnetRing object
ring = MagnetRing()

# move ring around
ring.position = (0,0,10)
ring.rotate_from_angax(angle=45, axis=(1,-1,0))

# compute field
print(f"B-field at sensor → {ring.getB(sensor).round(2)}")

# display graphically
magpy.show(ring, sensor, backend='plotly')
```

The ring parameter `cubes` can be modified dynamically:

```{code-cell} ipython3
ring.cubes=15

print(f"B-field at sensor for modified ring → {ring.getB(sensor).round(2)}")
magpy.show(ring, sensor, backend='plotly')
```

## Postponed trace construction

Custom traces can be computationally costly to construct. In the above example, the trace is constructed in `_update`, every time the parameter `cubes` is modified. This can lead to an unwanted computational overhead, especially as the construction is only necessary for graphical representation.

To make our compounds ready for heavy computation, it is possible to provide a callable as a trace, which will only be constructed when `show` is called. The following modification of the above example demonstrates this:

```{code-cell} ipython3
import magpylib as magpy
import numpy as np

class MagnetRingAdv(magpy.Collection):
    """ A ring of cuboid magnets

    Parameters
    ----------
    cubes: int, default=6
        Number of cubes on ring.
    """

    def __init__(self, cubes=6, **style_kwargs):
        super().__init__(**style_kwargs)             # hand over style args
        self._update(cubes)

        # hand trace over as callable
        self.style.model3d.add_trace(self._custom_trace3d)

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

        # construct in temporary Collection for path transfer
        temp_coll = magpy.Collection()
        for i in range(cubes):
            child = magpy.magnet.Cuboid(
                magnetization=(1000,0,0),
                dimension=(1,1,1),
                position=(ring_radius,0,0)
            )
            child.rotate_from_angax(360/cubes*i, 'z', anchor=0)
            temp_coll.add(child)

        # transfer path and children
        temp_coll.position = self.position
        temp_coll.orientation = self.orientation
        self.children = temp_coll.children

        return self

    def _custom_trace3d(self):
        """ creates a parameter-dependent 3d model"""
        r1 = self.cubes/3 - .6
        r2 = self.cubes/3 + 0.6
        trace = magpy.graphics.model3d.make_CylinderSegment(
            dimension=(r1, r2, 1.1, 0, 360),
            vert=150,
            opacity=0.5,
        )
        return trace
```

All we have done is, to remove the trace construction from the `_update` method, and instead provide `_custom_trace3d` as callable in `__init__` with the help of `partial`.

```{code-cell} ipython3
ring0 = MagnetRing()
%time for _ in range(100): ring0.cubes=10

ring1 = MagnetRingAdv()
%time for _ in range(100): ring1.cubes=10
```

This example is not very impressive because the provided trace is not very heavy. Finally, we play around with our new compound:

```{code-cell} ipython3
rings = []
for i,cub in zip([2,7,12,17,22], [20,16,12,8,4]):
    ring = MagnetRingAdv(cubes=cub, style_label=f'MagnetRingAdv (x{cub})')
    ring.rotate_from_angax(angle=np.linspace(0,45,10), axis=(1,-1,0))
    ring.move(np.linspace((0,0,0), (-i,-i,i), i))
    rings.append(ring)

magpy.show(rings, animation=2, backend='plotly', style_path_show=False)
```

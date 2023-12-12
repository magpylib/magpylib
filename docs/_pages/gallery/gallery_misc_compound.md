---
orphan: true
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

(gallery-misc-compound)=

# Compounds

The `Collection` class is a powerful tool for grouping and tracking object assemblies. However, it is often convenient to have assembly parameters themselves, like number of magnets, as variables. This is achieved by sub-classing `Collection`. We refer to such classes as "**Compounds**" and show how to seamlessly integrate them into Magpylib.

## Subclassing collections

In the following example we design a compound class `MagnetRing` which represents a ring of cuboid magnets with the parameter `cubes` that should refer to the number of magnets on the ring. The ring will automatically adjust its size when `cubes` is modified. We also add an encompassing 3D model.

```{code-cell} ipython3
import magpylib as magpy

class MagnetRing(magpy.Collection):
    """ A ring of cuboid magnets

    Parameters
    ----------
    cubes: int, default=6
        Number of cubes on ring.
    """

    def __init__(self, cubes=6, **kwargs):
        super().__init__(**kwargs)             # hand over style args
        self._update(cubes)

    @property
    def cubes(self):
        """Number of cubes"""
        return self._cubes

    @cubes.setter
    def cubes(self, inp):
        """Set cubes"""
        self._update(inp)

    def _update(self, cubes):
        """Update MagnetRing instance"""
        self._cubes = cubes
        ring_radius = cubes/3

        # Store existing path
        pos_temp = self.position
        ori_temp = self.orientation

        # Clean up old object properties
        self.reset_path()
        self.children = []
        self.style.model3d.data.clear()

        # Add children
        for i in range(cubes):
            child = magpy.magnet.Cuboid(
                magnetization=(0,0,1000),
                dimension=(1,1,1),
                position=(ring_radius,0,0)
            )
            child.rotate_from_angax(360/cubes*i, 'z', anchor=0)
            self.add(child)

        # Re-apply path
        self.position = pos_temp
        self.orientation = ori_temp

        # Add parameter-dependent 3d trace
        trace = magpy.graphics.model3d.make_CylinderSegment(
            dimension=(cubes/3-.6, cubes/3+0.6, 1.1, 0, 360),
            vert=150,
            opacity=0.2,
        )
        self.style.model3d.add_trace(trace)

        return self
```

This new `MagnetRing` class seamlessly integrates into Magpylib and makes use of the position and orientation interface, field computation and graphic display.

```{code-cell} ipython3
# Add a sensor
sensor = magpy.Sensor(position=(0, 0, 0))

# Create a MagnetRing object
ring = MagnetRing()

# Move MagnetRing around
ring.rotate_from_angax(angle=45, axis='x')

# Compute field
print(f"B-field at sensor → {ring.getB(sensor).round(2)}")

# Display graphically
magpy.show(ring, sensor, backend='plotly')
```

The `MagnetRing` parameter `cubes` can be modified dynamically:

```{code-cell} ipython3

print(f"B-field at sensor for modified ring → {ring.getB(sensor).round(2)}")

ring.cubes = 10

print(f"B-field at sensor for modified ring → {ring.getB(sensor).round(2)}")

magpy.show(ring, sensor, backend='plotly')
```

## Postponed trace construction

In the above example, the trace is constructed in `_update`, every time the parameter `cubes` is modified. This can lead to an unwanted computational overhead, especially as the construction is only necessary for graphical representation.

To make our compounds ready for heavy computation, while retaining Magpylib graphic possibilities, it is possible to provide a trace which will only be constructed when `show` is called. The following modification of the above example demonstrates this:

```{code-cell} ipython3
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

        # Hand trace over as callable
        self.style.model3d.add_trace(self._custom_trace3d)

    @property
    def cubes(self):
        """Number of cubes"""
        return self._cubes

    @cubes.setter
    def cubes(self, inp):
        """Set cubes"""
        self._update(inp)

    def _update(self, cubes):
        """Update MagnetRing instance"""
        self._cubes = cubes
        ring_radius = cubes/3

        # Store existing path and reset
        pos_temp = self.position
        ori_temp = self.orientation
        self.reset_path()

        # Add children
        for i in range(cubes):
            child = magpy.magnet.Cuboid(
                magnetization=(0,0,1000),
                dimension=(1,1,1),
                position=(ring_radius,0,0)
            )
            child.rotate_from_angax(360/cubes*i, 'z', anchor=0)
            self.add(child)

        # Re-apply path
        self.position = pos_temp
        self.orientation = ori_temp

        return self

    def _custom_trace3d(self):
        """ creates a parameter-dependent 3d model"""
        trace = magpy.graphics.model3d.make_CylinderSegment(
            dimension=(self.cubes/3-.6, self.cubes/3+0.6, 1.1, 0, 360),
            vert=150,
            opacity=0.2,
        )
        return trace
```

We have removed the trace construction from the `_update` method, and instead provided `_custom_trace3d` as a callable.

```{code-cell} ipython3
ring0 = MagnetRing()
%time for _ in range(100): ring0.cubes=10

ring1 = MagnetRingAdv()
%time for _ in range(100): ring1.cubes=10
```

This example is not very impressive because the provided trace is not very heavy.

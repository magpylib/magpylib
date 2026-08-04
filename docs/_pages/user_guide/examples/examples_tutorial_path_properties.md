---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
orphan: true
---

(examples-tutorial-path-properties)=

# Path-Varying Properties

Paths are not restricted to motion. Any attribute listed in an object's `path_properties` can vary along the path, which makes it possible to model a changing excitation, a deforming geometry or a changing size - with the same vectorized computation that already handles movement.

The technical description is in {ref}`docs-position-path-properties`. This page shows what it is good for.

```{code-cell} ipython3
import magpylib as magpy

loop = magpy.current.Circle(current=1, diameter=0.01)
print(loop.path_properties)
```

## A changing excitation

The most direct use is a current that changes over time. Here a coil carries one period of a sinusoidal current while staying put. Nothing moves - only the excitation varies.

```{code-cell} ipython3
import numpy as np

t = np.linspace(0, 1, 40)
coil = magpy.current.Circle(diameter=0.02, current=5 * np.sin(2 * np.pi * t))

B = magpy.getB(coil, (0, 0, 0.01))
print(B.shape)
```

The field is computed for all 40 steps in one vectorized call. Plotting the z-component recovers the drive signal:

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 2.5))
ax.plot(t, B[:, 2] * 1e6)
ax.set_xlabel("t (a.u.)")
ax.set_ylabel("Bz (µT)")
ax.grid(color=".9")
plt.tight_layout()
plt.show()
```

```{tip}
This is the vectorized alternative to a Python loop over `current` values. As with positional paths, avoid loops for this purpose - see {ref}`docs-position`.
```

## A changing size

Attributes describing geometry can vary too. A magnet that expands - through heating, or simply as a parameter sweep - is a path over `dimension`:

```{code-cell} ipython3
side = np.linspace(0.010, 0.011, 20)  # 10 mm to 11 mm

cube = magpy.magnet.Cuboid(
    dimension=np.column_stack([side, side, side]),
    polarization=(0, 0, 1),
)
print(cube.dimension.shape)
```

Derived properties follow along. Since the dipole moment depends on the volume, it now carries one value per step:

```{code-cell} ipython3
print(cube.dipole_moment.shape)
print(cube.dipole_moment[0], "->", cube.dipole_moment[-1])
```

Note that a parameter sweep like this is *not* the same as several objects: it is one object observed at 20 different sizes, and the field output has one entry per step rather than per object.

## A deforming geometry

`vertices` is a path property as well, so a conductor can change shape. This polyline is a ring that buckles into a three-fold wave:

```{code-cell} ipython3
steps, pts = 15, 60
phi = np.linspace(0, 2 * np.pi, pts)

vertices = np.array(
    [
        np.column_stack(
            [0.01 * np.cos(phi), 0.01 * np.sin(phi), amp * np.sin(3 * phi)]
        )
        for amp in np.linspace(0, 0.004, steps)
    ]
)

line = magpy.current.Polyline(vertices=vertices, current=1)
print(vertices.shape)  # (path, points, 3)
```

```{code-cell} ipython3
magpy.show(line, animation=True, backend="plotly")
```

## Combining with motion

Path properties are independent of each other, so an object can move *and* change at the same time. Attributes of differing length are edge-padded to the longest when the field is computed - see {ref}`examples-tutorial-paths-mismatched-properties`.

```{code-cell} ipython3
coil = magpy.current.Circle(
    diameter=np.linspace(0.01, 0.02, 10),
    current=np.linspace(1, 5, 10),
    position=[(0, 0, z) for z in np.linspace(0, 0.02, 10)],
)

magpy.show(coil, animation=True, backend="plotly")
```

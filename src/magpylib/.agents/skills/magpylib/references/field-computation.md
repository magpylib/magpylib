# Field computation

Read this when output shapes matter, when many parameter sets must be evaluated,
or when a computation is too slow.

## Output shape

`getB`/`getH`/`getJ`/`getM` return `(s, p, o, o1, o2, ..., 3)`:

| Axis        | Meaning                                         |
| ----------- | ----------------------------------------------- |
| `s`         | number of sources                               |
| `p`         | path length (longest path among the objects)    |
| `o`         | number of observers                             |
| `o1, o2, …` | sensor pixel shape, or the observer array shape |
| `3`         | field components (x, y, z)                      |

With `squeeze=True` (the default) every axis of length 1 is dropped. A single
source, no path, one observer therefore yields `(3,)`, while adding a second
source silently changes the rank to `(2, 3)`.

Pass `squeeze=False` in library code, then index explicitly:

```python
B = magpy.getB(sources, observers, squeeze=False)  # always 5+ dims
b_first_source = B[0]
```

## Observers

Any of these work as `observers`:

- an array-like of shape `(..., 3)` — positions in global coordinates
- a `Sensor` — the field is returned in the sensor's own frame
- a `Collection` containing observers
- a flat list mixing the above

Sensors with **different** pixel shapes cannot be broadcast into one array; pass
`pixel_agg="mean"` (or any NumPy aggregator name) to reduce each sensor's pixels
first, which makes the shapes compatible.

## sumup and output

```python
B = magpy.getB([magnet_a, magnet_b], grid, sumup=True)  # superposition
df = magpy.getB(sources, sensors, output="dataframe")  # tidy pandas frame
```

`sumup=True` collapses the source axis by summing — the physically meaningful
superposition of several sources. `output="dataframe"` is convenient for
plotting libraries and for `groupby` over sources/sensors/path indices.

## The functional interface

`magpy.func.*` computes fields for many parameter sets without creating objects.
Use it when the parameters vary, not just the positions — sweeping dimensions,
polarizations, or currents.

```python
import numpy as np
import magpylib as magpy

n = 100_000
B = magpy.func.cuboid_field(
    field="B",
    observers=np.random.rand(n, 3),
    dimensions=np.random.rand(n, 3),
    polarizations=np.tile((0, 0, 1.0), (n, 1)),
)  # (100000, 3)
```

Available: `circle_field`, `polyline_field`, `cuboid_field`, `cylinder_field`,
`cylinder_segment_field`, `sphere_field`, `tetrahedron_field`, `dipole_field`,
`triangle_charge_field`, `triangle_current_field`.

Every function takes `field` (`"B"` or `"H"`), `observers`, its shape and
excitation parameters, and optional `positions`, `orientations`, `squeeze`.
Scalar inputs are tiled to the longest input length. Output is `(i, 3)`.

This replaced the v4-era string dispatch: `getB("Cuboid", ...)` and `getB_dict`
no longer exist.

## Performance

Magpylib vectorises over sources, path steps, observers and pixels in one pass.
The rule is one `getB` call per experiment:

```python
# Slow: one call per position
for z in heights:
    B.append(magpy.getB(cube, (0, 0, z)))

# Fast: one call, all positions
B = magpy.getB(cube, [(0, 0, z) for z in heights])
```

The same applies to moving sources — set a path rather than repositioning in a
loop (see [position-paths.md](position-paths.md)).

If observers sit inside `Tetrahedron` or `TriangularMesh` bodies and their
location is known, `in_out="inside"` / `"outside"` skips the per-observer
containment test.

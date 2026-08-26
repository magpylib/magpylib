# Position, orientation, and paths

Read this before writing any motion — rotation about a point, a swept
trajectory, a parameter ramp — or when a path came out with the wrong length.

## The path

`position` (shape `(3,)` or `(p, 3)`, metres) and `orientation` (a scipy
`Rotation` of length 1 or `p`) together define an object's path. Magpylib keeps
the two the same length. A field computation evaluates every path step at once,
so paths are the vectorised way to model motion.

```python
import numpy as np
import magpylib as magpy
from scipy.spatial.transform import Rotation as R

cube = magpy.magnet.Cuboid(dimension=(0.01, 0.01, 0.01), polarization=(0, 0, 1))
cube.position = np.linspace((0, 0, 0.02), (0, 0, 0.10), 50)
B = magpy.getB(cube, (0, 0, 0))  # (50, 3) — one call, 50 steps
```

`orientation` must be a `scipy.spatial.transform.Rotation`. A tuple of Euler
angles raises `MagpylibBadUserInput`.

## move() and rotate()

```python
obj.move(displacement, start="auto")
obj.rotate(rotation, anchor=None, start="auto")
```

The behaviour depends on whether the input is scalar or vector, and this is the
most common source of surprise:

| Input                          | `start="auto"` means | Effect                                 |
| ------------------------------ | -------------------- | -------------------------------------- |
| scalar (one vector / rotation) | `start=0`            | transforms the **whole existing path** |
| vector (n vectors / rotations) | `start=len(path)`    | **appends** n new steps to the path    |

So `obj.move((0, 0, 0.01))` shifts an existing 50-step path bodily, while
`obj.move(np.linspace((0, 0, 0), (0, 0, 0.01), 10))` grows the path by 10 steps.
Pass an explicit `start=0` to overwrite instead of append.

`anchor` is the pivot for rotation: `anchor=None` (default) spins the object
about its own `position`, an array-like `(3,)` rotates it about that global
point — which is how you build orbits and Halbach arrangements.

```python
# 8 magnets evenly placed on a circle, each rotated about the origin
magnets = []
for i in range(8):
    m = magpy.magnet.Cuboid(dimension=(0.01, 0.01, 0.01), polarization=(0, 0, 1))
    m.position = (0.05, 0, 0)
    m.rotate(R.from_rotvec((0, 0, i * 45), degrees=True), anchor=(0, 0, 0))
    magnets.append(m)
```

Chained calls compose, and `reset_path()` restores `position=(0, 0, 0)` and
`orientation=None`.

## Path-varying attributes

Beyond position and orientation, physical attributes can vary along the path:
`current`, `diameter`, `dimension`, `polarization`, `magnetization`, `moment`,
`vertices`. Ask an object which of its attributes support this with
`obj.path_properties`.

```python
coil = magpy.current.Circle(
    diameter=np.linspace(0.01, 0.05, 10),  # geometry ramps
    current=np.linspace(1, 10, 10),  # excitation ramps
)
```

This models AC or ramped currents, deforming geometry, and thermal expansion
without a Python loop. Attributes are independent — setting one to a path does
not lengthen the others; shorter ones are edge-padded to the longest at
computation time.

## Frames

Shape parameters (`dimension`, `vertices`, sensor `pixel`) are always in the
object's **local** frame. `position`/`orientation` place that frame in the
global one. A `Collection` spans its own frame for its children: moving the
collection moves all children while preserving their relative placement, and
children remain individually addressable afterwards.

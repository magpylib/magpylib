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

Prefer the convenience constructors over assembling a `Rotation` yourself; they
take the same `anchor` and `start`:

| Method                           | Input                        |
| -------------------------------- | ---------------------------- |
| `rotate_from_angax(angle, axis)` | angle (°) about a named axis |
| `rotate_from_rotvec(rotvec)`     | rotation vector              |
| `rotate_from_euler(angles, seq)` | Euler angles                 |
| `rotate_from_quat(quat)`         | quaternion                   |
| `rotate_from_matrix(matrix)`     | rotation matrix              |
| `rotate_from_mrp(mrp)`           | modified Rodrigues params    |

Each accepts vector input as well, so
`sensor.rotate_from_angax(np.linspace(0, 360, 37), "z", start=0)` writes a full
37-step spin over an existing path.

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
`vertices`. Ask an object which of its attributes support this — the answer is
per class:

```python
loop = magpy.current.Circle(current=1, diameter=0.01)
loop.path_properties  # ('position', 'orientation', 'current', 'diameter')
```

A sinusoidal drive is then one object, not forty:

```python
t = np.linspace(0, 1, 40)
coil = magpy.current.Circle(diameter=0.02, current=5 * np.sin(2 * np.pi * t))
B = magpy.getB(coil, (0, 0, 0.01))  # (40, 3) in one vectorised call
```

Geometry works the same way — a cube expanding from 10 mm to 11 mm is a path
over `dimension`, and `vertices` lets a conductor deform:

```python
side = np.linspace(0.010, 0.011, 20)
cube = magpy.magnet.Cuboid(
    dimension=np.column_stack([side, side, side]),
    polarization=(0, 0, 1),
)
cube.dimension.shape  # (20, 3)
cube.dipole_moment.shape  # (20, 3) — derived properties gain the path axis
```

### Storage and reconciliation

`position` and `orientation` are special in exactly one respect: they stay
eagerly synchronised with each other, because geometric consistency is always
required. **Every other path property is independent, and what you set is what
is stored** — an attribute keeps the shape it was given and is never expanded to
match the others.

```python
cube = magpy.magnet.Cuboid(dimension=(0.01, 0.01, 0.01), polarization=(0, 0, 1))
cube.position = [(0, 0, z) for z in np.linspace(0, 0.04, 5)]
np.shape(cube.position)  # (5, 3) — a path
cube.dimension  # [0.01 0.01 0.01] — untouched
```

Lengths are reconciled only when they must be — at field, force, and display
time — on a temporary copy, by two rules:

- **Edge-padding**: when a step beyond an attribute's own length is needed, its
  last entry is repeated. The object is _static_ beyond its path.
- **End-slicing**: when a path is automatically shortened, the **end** is kept.

So mismatched lengths inside one object are legal and quiet:

```python
loop = magpy.current.Circle(current=[1, 2, 3], diameter=0.01)
loop.position = [(0, 0, 0), (0, 0, 0.01)]  # only 2 steps

magpy.getB(loop, (0, 0, 0.02)).shape  # (3, 3) — runs over 3 steps
len(loop.position)  # still 2 — padding never rewrote the attribute
```

The third step reuses the final position while the current keeps changing. This
is a feature for combining a long excitation with a short motion, and a trap
when the mismatch was unintentional: nothing raises, so a wrong path length
shows up only as an unexpected leading axis. The same edge-padding applies
_between_ objects — when sources of different path lengths are combined, the
shorter ones are treated as static beyond their end.

## Frames

Shape parameters (`dimension`, `vertices`, sensor `pixel`) are always in the
object's **local** frame. `position`/`orientation` place that frame in the
global one. A `Collection` spans its own frame for its children: moving the
collection moves all children while preserving their relative placement, and
children remain individually addressable afterwards.

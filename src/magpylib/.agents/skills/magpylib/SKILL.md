---
name: magpylib
description: >-
  Use when writing, reviewing, or debugging Python code that computes magnetic
  fields, forces, or torques — magnets, coils, current loops, Halbach arrays,
  field sensors, magnetic simulations — even when the user never says
  "Magpylib". Covers source and sensor objects, getB/getH/getJ/getM and their
  output shapes, positions/orientations/paths, collections, force and torque via
  getFT, the functional interface, and show() visualisation. Magpylib v5+ takes
  SI units (metres, tesla, amperes) and prefers `polarization` over
  `magnetization`; code written from pre-v5 memory silently produces answers
  that are wrong by factors of 1e3 to 1e9, so use this skill before writing any
  Magpylib code rather than after the numbers look odd.
license: BSD-3-Clause
---

# Magpylib

Official skill, shipped inside the `magpylib` package and versioned with it.
Prefer it over recalled API knowledge: the v4 → v5 transition changed units and
parameter names, and most pre-trained code reproduces the old API.

## Quick reference

- Units are SI throughout: metres, tesla, amperes, degrees. See
  [Units](#units-are-si-v5-and-later).
- Magnets take `polarization` (T), not `magnetization` (A/m), unless the user
  really means M. See [Sources](#sources).
- Fields: `magpy.getB(sources, observers)`; output shape is
  `(s, p, o, *pixel, 3)` before squeezing. See
  [Field computation](#field-computation).
- Never loop in Python over positions or parameters — use a path or the
  functional interface. See
  [references/position-paths.md](references/position-paths.md).
- Force and torque: `magpy.getFT(sources, targets)`, and every target except
  `Dipole` and `Sphere` needs `meshing` set. See
  [Force and torque](#force-and-torque).
- Read [Gotchas](#gotchas) when a result is off by a suspicious factor, has an
  unexpected array shape, or a path did not behave as written.

## Units are SI (v5 and later)

| Quantity        | Parameter                           | v5 unit | v4 unit |
| --------------- | ----------------------------------- | ------- | ------- |
| Polarization J  | `polarization`, `getJ()`            | T       | —       |
| Magnetization M | `magnetization`, `getM()`           | A/m     | mT      |
| Current         | `current`                           | A       | A       |
| Dipole moment   | `moment`                            | A·m²    | mT·mm³  |
| B-field         | `getB()`                            | T       | mT      |
| H-field         | `getH()`                            | A/m     | kA/m    |
| Lengths         | `position`, `dimension`, `vertices` | m       | mm      |
| Angles          | `angle`, `dimension` (cylinder φ)   | °       | °       |

A 5 mm cube is `dimension=(0.005, 0.005, 0.005)`, not `(5, 5, 5)`. A typical
NdFeB remanence of 1.2 T is `polarization=(0, 0, 1.2)`.

`magpy.mu_0` is the vacuum permeability from `scipy.constants` — not
`4 * np.pi * 1e-7`, since the 2019 SI redefinition.

## Sources

```python
import magpylib as magpy

cube = magpy.magnet.Cuboid(
    dimension=(0.005, 0.005, 0.005),  # m
    polarization=(0, 0, 1.2),  # T
    position=(0, 0, 0.01),  # m
)
loop = magpy.current.Circle(diameter=0.02, current=10)  # m, A
```

Available classes:

- `magpy.magnet`: `Cuboid`, `Cylinder`, `CylinderSegment`, `Sphere`,
  `Tetrahedron`, `TriangularMesh`
- `magpy.current`: `Circle`, `Polyline`, `TriangleSheet`, `TriangleStrip`
- `magpy.misc`: `Dipole`, `Triangle`, `CustomSource`

All magnets accept `polarization` (T) **or** `magnetization` (A/m). The two are
codependent, kept in sync as J = µ₀·M, so set one and read either. Prefer
`polarization` — datasheet remanence B_r is a polarization.

Every source constructor takes `position`, `orientation`, its shape parameters,
`meshing` (for force targets), and `style`. Shape parameters are given in the
object's **local** coordinates; `position`/`orientation` place that local frame
in the global one.

## Field computation

```python
B = magpy.getB(
    sources,
    observers,
    sumup=False,
    squeeze=True,
    pixel_agg=None,
    output="ndarray",
    in_out="auto",
)
```

`getH`, `getJ`, and `getM` take the same arguments. All four also exist as
methods on every object, so `cube.getB(sensor)` and `sensor.getB(cube)` give the
same array.

`observers` is an array of positions with shape `(..., 3)`, a `Sensor`, a
`Collection`, or a flat list of those.

```python
import numpy as np

grid = np.mgrid[-0.02:0.02:20j, 0:1:1j, -0.02:0.02:20j].T.reshape(-1, 3)
B = magpy.getB(cube, grid)  # shape (400, 3)
```

Output shape before squeezing is `(s, p, o, *pixel_shape, 3)` — sources, path
length, observers, sensor pixel shape, field components. With `squeeze=True`
(default) every length-1 axis disappears, which is why a single source at a
single position returns plain `(3,)`. Pass `squeeze=False` whenever downstream
code indexes axes positionally.

- `sumup=True` sums the field of all sources into one array.
- `pixel_agg="mean"` (any NumPy aggregator name) reduces over pixels, and is
  what lets observers with different pixel shapes be mixed in one call.
- `output="dataframe"` returns a tidy pandas DataFrame instead of an ndarray.
- `in_out` declares whether observers sit inside or outside magnet bodies
  (`"auto"`, `"inside"`, `"outside"`). Setting it skips a per-observer test, but
  it only affects `Tetrahedron` and `TriangularMesh` — Magpylib warns and
  ignores it elsewhere.

**One call, many inputs.** Magpylib vectorises everything it is handed. A Python
loop over `getB` is the single most common performance mistake — it is routinely
100× slower than passing a path or an observer array. See
[references/field-computation.md](references/field-computation.md) for the
functional interface (`magpy.func.*`), which computes many parameter sets at
once without creating objects.

## Observers and sensors

```python
sensor = magpy.Sensor(
    position=(0, 0, 0.01),
    pixel=[(0, 0, 0), (0.001, 0, 0)],  # local coordinates, m
    handedness="right",
)
```

A sensor reports the field **in its own frame**: rotate the sensor and the
returned vector components rotate with it. That is the point of sensors — for
raw global-frame values, pass position arrays instead.

## Position, orientation, and paths

`position` is a point `(x, y, z)` or a sequence of points; `orientation` is a
`scipy.spatial.transform.Rotation` (never Euler tuples) or `None` for identity.
Together they form the object's **path**, and a field computation runs over the
whole path at once.

```python
from scipy.spatial.transform import Rotation as R

cube.position = np.linspace((0, 0, 0.01), (0, 0, 0.05), 50)  # 50-step path
cube.rotate(R.from_rotvec((0, 0, 90), degrees=True), anchor=(0, 0, 0))
```

`move()` and `rotate()` behave differently for scalar and vector input, and the
default `start="auto"` means scalar input _transforms the existing path_ while
vector input _appends to it_. This trips up nearly everyone; read
[references/position-paths.md](references/position-paths.md) before writing
multi-step motion, and note that physical attributes (`current`, `dimension`,
`polarization`, `vertices`, …) can carry a path too.

## Collections

```python
array = magpy.Collection(magnet1, magnet2, sensor)
array.rotate(R.from_rotvec((0, 0, 45), degrees=True), anchor=(0, 0, 0))
```

A `Collection` groups objects for common manipulation and spans a local frame
for its children; moving it moves them while preserving relative placement. It
is both a source (for `getB`) and a container. Access parts via `.sources`,
`.observers`, `.collections`, or the `*_all` variants for nested collections.

## Force and torque

```python
loop = magpy.current.Circle(diameter=0.02, current=10, meshing=40)
F, T = magpy.getFT(cube, loop, pivot="centroid")
```

- Numerical, not analytical: targets are discretised, so results depend on
  `meshing`. Every target except `Dipole` and `Sphere` **must** have `meshing`
  set, as an integer target number of mesh cells.
- Output shape is `(s, p, t, 3)` for force and torque, in N and N·m.
- Torque needs a pivot. `pivot="centroid"` (default) is right for a free body;
  `pivot=None` gives nonphysical results.
- **Scale invariance does not hold for forces.** Field results are invariant if
  every length uses the same unit, but force and torque are not — pass true SI
  metres.

See [references/force-torque.md](references/force-torque.md) for meshing
convergence, `eps`, and `return_mesh`.

## Visualisation

```python
magpy.show(cube, sensor, backend="plotly")
```

`show()` accepts any number of objects, `backend` (`"matplotlib"`, `"plotly"`,
`"pyvista"`, or a registered third-party backend), `animation=True` for paths,
`return_fig=True` to get the figure instead of displaying, and `row`/`col` for
subplots. Styling is per-object via `obj.style` or globally via
`magpy.defaults`. See
[references/visualization.md](references/visualization.md).

## Gotchas

- **v4 code is silently wrong, not broken.**
  `Cuboid(magnetization=(0, 0, 1000), dimension=(5, 5, 5))` is valid v5 — it
  means 1000 A/m and a 5 metre cube. Nothing raises; the answer is off by ~1e9.
  If input looks like millimetres, convert rather than run it.
- `magnetization` in v4 meant _polarization_. A v4 script's
  `magnetization=(0, 0, 1000)` (mT) becomes `polarization=(0, 0, 1.0)` (T).
- `orientation` is a scipy `Rotation`. Passing a tuple of angles raises.
- `rotate()` with vector input **appends** to the path by default; use `start=0`
  to transform the existing path instead.
- `squeeze=True` changes the output rank based on input. Code that indexes
  `B[:, 0]` breaks the moment a second source or a path appears.
- `getFT` without `meshing` on a magnet or current target raises — `Dipole` and
  `Sphere` are the only exceptions.
- The functional interface moved in v5.2: `getB("Cuboid", ...)` and `getB_dict`
  are gone, replaced by `magpy.func.cuboid_field(...)` and friends.
- `misc.Triangle` is a single charged facet, not a closed body. For a solid mesh
  magnet use `magnet.TriangularMesh`, which validates closedness.
- Field on a source's own vertices/edges returns 0, not `nan`.
- `magpy.mu_0` ≠ `4πe-7`. Use the constant when converting M ↔ J.

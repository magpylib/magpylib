# Force and torque

Read this when computing `getFT` — holding force, magnetic bearings, actuator
torque, floating-magnet equilibria — or when force results look mesh-dependent.

```python
F, T = magpy.getFT(
    sources,
    targets,
    pivot="centroid",
    eps=1e-5,
    squeeze=True,
    meshreport=False,
    return_mesh=False,
)
```

`F` is in newtons, `T` in newton-metres, both shaped `(s, p, t, 3)` — sources,
path, targets, components — before squeezing. They arrive stacked: `getFT`
returns one `(2, s, p, t, 3)` array, and the two-name assignment above unpacks
its leading axis. Bind a single name instead and that axis is still there.

## Meshing is mandatory

Unlike field computation, force is numerical: each target is discretised and the
force integral summed over cells. Every target **except `Dipole` and `Sphere`**
must have `meshing` set, or `getFT` raises `ValueError`.

```python
loop = magpy.current.Circle(diameter=0.02, current=10, meshing=40)
cube = magpy.magnet.Cuboid(
    dimension=(0.01, 0.01, 0.01), polarization=(0, 0, 1.2), meshing=1000
)
```

`meshing` is an integer _target_ cell count; the mesher aims for uniform,
aspect-ratio-1 cells and will not hit the number exactly. Pass `meshreport=True`
to print the realised cell count per target, and `return_mesh=True` to get the
mesh dictionaries (`"pts"`, `"moments"` for magnets, `"cvecs"` for currents)
instead of F and T.

**Always run a convergence check** — double `meshing` until the result stops
moving:

```python
for n in (100, 400, 1600, 6400):
    cube.meshing = n
    F, _ = magpy.getFT(source, cube)
    print(n, F)
```

## Scale invariance does NOT hold

Field computation is scale invariant: a 1 mm magnet at 1 mm distance gives the
same field as a 1 m magnet at 1 m, so any consistent length unit works. **Force
and torque break this.** Inputs must be true SI metres or the newtons are
meaningless.

## Pivot

Torque is defined about a pivot, and the force contributes
`T_F = F × (r_pivot − r_position)`:

- `pivot="centroid"` (default) — the target's barycenter, correct for a freely
  floating body.
- array-like `(3,)` — the same pivot point for every target, e.g. a shaft axis.
- one pivot per target — an array with one row per target.
- `pivot=None` — no pivot; results are nonphysical, use only when you know why.

## eps

`eps` is the finite-difference step used for the field gradient on magnet
targets. The default `1e-5` suits metre-scale problems; a good value is roughly
`1e-6 ×` the characteristic source size. Too large smears the gradient, too
small loses precision to floating-point cancellation. It is unused for current
targets.

## Sanity checks

- Newton's third law: `getFT(a, b)` and `getFT(b, a)` should give opposite
  forces. A mismatch usually means one target is under-meshed.
- A target must be a magnet, a current, or a `Dipole`; `CustomSource` cannot be
  a target.
- Forces between touching or overlapping bodies are unreliable — the analytical
  field diverges at surfaces.

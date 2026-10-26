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
    meshing=None,
)
```

`getFT` returns a tuple `(F, T)`: `F` in newtons, `T` in newton-metres, each
shaped `(s, p, t, 3)` — sources, path, targets, components — before squeezing. A
Collection target counts as one body.

## Meshing is mandatory

Unlike field computation, force is numerical: each target is cut into cells and
the force summed over them. `meshing` is a keyword of `getFT`, required for
every target **except `Dipole` and `Sphere`**, which are exact; without it
`getFT` raises `ValueError`. Since v6 it is no longer a target attribute, and
`Cuboid(..., meshing=100)` raises `TypeError`.

```python
loop = magpy.current.Circle(diameter=0.02, current=10)
ring = magpy.magnet.Cylinder(
    dimension=(0.01, 0.005), polarization=(0, 0, 1.2), position=(0, 0, 0.03)
)
F, T = magpy.getFT(cube, [loop, ring], meshing={loop: 40, ring: 1000})
```

- An integer is a _target_ cell count (segments for currents) for every target,
  including each magnet in a Collection. The mesher aims for near-cubic cells
  and will not hit the number exactly.
- A dict gives one value per target; a Collection key covers its members.
- A `Cuboid` also takes a triple `(n1, n2, n3)` of divisions along its sides, a
  `Cylinder` or `CylinderSegment` a triple `(nphi, nr, nh)`.

**Always run a convergence check** — double `meshing` until the result stops
moving:

```python
for n in (100, 400, 1600, 6400):
    F, _ = magpy.getFT(source, cube, meshing=n)
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

- `pivot="centroid"` (default) — the target's centroid, correct for a freely
  floating body. For a Collection it is the volume-weighted centroid of its
  magnets (the mean of its members' centroids for a coil), so the torque is that
  on one rigid body.
- array-like `(3,)` — the same pivot point for every target, e.g. a shaft axis.
- `(t, 3)` or `(t, p, 3)` — one pivot per target, or per target and path step.
- `pivot=None` — no pivot; results are nonphysical, use only when you know why.

## eps

`eps` is the finite-difference step used for the field gradient on magnet
targets. A good value is `1e-5 ×` the characteristic system size (magnet size or
source-target distance), so the default `1e-5` suits metre-scale problems and
centimetre magnets want about `1e-7`. Too large smears the gradient, too small
loses precision to floating-point cancellation. It is unused for current
targets.

## Sanity checks

- Newton's third law: `getFT(a, b)` and `getFT(b, a)` should give opposite
  forces. A mismatch usually means one target is under-meshed.
- A target must be a magnet, a current, or a `Dipole`; `CustomSource` cannot be
  a target.
- Forces between touching or overlapping bodies are unreliable — the analytical
  field diverges at surfaces.

(guide-migration-v6)=
# Migrating to version 6

Version 6 is a breaking release. This page lists every change that can break a
version 5 script, each with its fix. Anything not listed here works as before.
The complete list of changes, including what was added and fixed, is in the
[changelog](changelog).

## Force and torque

### `meshing` is an argument of `getFT()`

How finely a target is cut describes one computation, not the magnet, so it
moved from the target objects to the call
([#1017](https://github.com/magpylib/magpylib/issues/1017)). An integer applies
to every target, including each magnet inside a Collection. A dict keyed by
target gives each its own value, a Collection key covers its members, and a
`Cuboid` value may be a triple of divisions along its sides. `Dipole` and
`Sphere` targets are exact and need none.

```python
# version 5
cube = magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1), meshing=100)
F, T = magpy.getFT(src, cube)

# version 6
cube = magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))
F, T = magpy.getFT(src, cube, meshing=100)
F, T = magpy.getFT(
    src, [ring, coil, cube], meshing={ring: 100, coil: 40, cube: (10, 10, 10)}
)
```

`meshing` is keyword-only. The old constructor keyword raises `TypeError`
naming the call, and so does `getFT()` for a target given no value. The old
attribute is gone. Saved scenes no longer carry a
numerical setting, and a Collection no longer needs a value on every member.

### Torque on a Collection target is about the Collection's centroid

`getFT` returns one force and torque per Collection target, as for one rigid
body. With `pivot="centroid"`, version 5 took each magnet's torque about its own
centroid, so the sum was not the torque on the body
([#1018](https://github.com/magpylib/magpylib/issues/1018)). Version 6 uses the
Collection's centroid: volume-weighted over its magnets, nested Collections
counted once, or the mean of its members' centroids for a Collection without
volume, such as a coil of current loops. Forces and single-magnet targets are
unchanged; torques on Collection targets change.

Array pivots now count the targets as given, a Collection as one, like the
output does: shape `(t, 3)` or `(t, p, 3)` for `t` entries in `targets`.

```python
# version 5: one pivot per magnet inside the Collection
F, T = magpy.getFT(src, body, pivot=[body.centroid, body.centroid])

# version 6: one pivot per target
F, T = magpy.getFT(src, body, pivot=[body.centroid])  # or simply pivot=body.centroid
```

A single target on a path may also pass its own centroid path, shape `(p, 3)`.

### Cylinder and tetrahedron forces change at the same `meshing`

Cylinders and cylinder segments get about `meshing` cells instead of up to twice
as many, each force point sits at its cell's true centroid, and a `Tetrahedron`
is cut into the closest cube number of equal tetrahedra, so that more cells no
longer make its force worse
([#1023](https://github.com/magpylib/magpylib/pull/1023)). `Cuboid` results are
unchanged. Expect small differences on cylinders and tetrahedra, and re-check a
convergence study if you have one.

Invalid meshing values such as `True`, `(-1, 2, 2)` or `(0, 0, 0)` now raise
`ValueError` instead of giving wrong forces, and NumPy integers are accepted.

## Display

### A 2D panel shows outputs in one unit

A 2D panel no longer mixes outputs in T (`B`, `J`) with outputs in A/m (`H`,
`M`) ([#1010](https://github.com/magpylib/magpylib/pull/1010)). Give them
separate panels.

```python
# version 5
magpy.show(sensor, output=["Bx", "Hx"])

# version 6
with magpy.show_context(sensor) as sc:
    sc.show(output="Bx", col=1)
    sc.show(output="Hx", col=2)
```

### `style.model3d.data` is a tuple

Custom 3D models are stored in a tuple, so `append` is gone
([#987](https://github.com/magpylib/magpylib/pull/987)).

```python
# version 5
obj.style.model3d.data.append(trace)

# version 6
obj.style.model3d.add_trace(trace)
obj.style.model3d.data = [*obj.style.model3d.data, trace]  # equivalent
```

### Invalid style and defaults values raise `ValueError`

They raised `AssertionError` before
([#967](https://github.com/magpylib/magpylib/pull/967)). Number-valued style
properties also reject booleans now, so `style_magnetization_show=1` still works
while a size given as `True` does not. Update any `except AssertionError`.

### Misspelled keyword arguments raise

A misspelled keyword is no longer ignored. `getB()`, `getH()`, `getJ()` and
`getM()` raise `TypeError`, and so does `show()` for its own arguments, such as
`animation_tme=`. `show()` warns about options the chosen backend does not
declare, and a `fig_*` or `show_*` argument the plotting library rejects raises
that library's error, with a note naming the argument
([#973](https://github.com/magpylib/magpylib/pull/973),
[#1005](https://github.com/magpylib/magpylib/pull/1005),
[#1014](https://github.com/magpylib/magpylib/pull/1014)).

### The legend past `legend_maxitems`

With more objects than `legend_maxitems`, the legend lists the first entries and
"… and k more" instead of disappearing
([#1007](https://github.com/magpylib/magpylib/pull/1007)). Nothing to change
unless you relied on the legend vanishing.

## Removed

- `getB("Cuboid", ...)` and `getH("Cuboid", ...)`, deprecated in 5.2. Use the
  functional interface, for example `magpy.func.cuboid_field(...)`
  ([#1005](https://github.com/magpylib/magpylib/pull/1005)).
- The deprecated `magnetization.size` key of a style's `as_dict()`
  ([#967](https://github.com/magpylib/magpylib/pull/967)).

## Deprecated

- `barycenter`, in favour of `centroid`
  ([#916](https://github.com/magpylib/magpylib/pull/916)). It warns in version 6
  and will be removed in version 7.

## Checklist

1. Search your code for `meshing=` in constructors and `.meshing =`
   assignments, and move each value to the `getFT()` call.
2. If you pass array pivots for Collection targets, give one per target.
3. If you compare forces on cylinders or tetrahedra against stored numbers,
   refresh the numbers.
4. Replace `except AssertionError` around style settings with `except
   ValueError`.
5. Replace `getB("Cuboid", ...)` with `magpy.func.cuboid_field(...)`, and
   `obj.barycenter` with `obj.centroid`.

# Visualisation

Read this when producing figures, animating paths, building subplots, or styling
objects.

## show()

```python
magpy.show(
    *objects,
    backend,
    canvas,
    canvas_update,
    animation,
    zoom,
    markers,
    return_fig,
    row,
    col,
    output,
    sumup,
    pixel_agg,
    style,
    **kwargs,
)
```

`show()` takes any number of sources, sensors, and collections. It is also a
method on every object (`cube.show()`), and `magpy.show(cube, sensor)` draws
them in one scene. Every keyword defaults to the matching entry in
`magpy.defaults` rather than to a literal in the signature, so changing a
default globally changes what an unqualified `show()` does.

- `return_fig=True` returns the backend figure instead of displaying it — use
  this in scripts, tests, and docs builds rather than relying on a display.
- `canvas` draws into an existing figure/axes/plotter, which is how Magpylib
  scenes get embedded in a larger dashboard.
- `zoom` widens (positive) or tightens the automatic bounds.
- `markers` adds plain position markers for reference points.

## Backends

Built in: `"matplotlib"`, `"plotly"`, `"pyvista"` (see
`magpy.SUPPORTED_PLOTTING_BACKENDS`). The default is
`magpy.defaults.display.backend`, itself `"auto"`, which picks by context.

```python
magpy.defaults.display.backend = "plotly"  # session-wide
magpy.show(cube, backend="pyvista")  # one call
```

All three animate and do subplots; they differ in what else they can draw:

| Capability                              | matplotlib | plotly | pyvista |
| --------------------------------------- | ---------- | ------ | ------- |
| `supports_animation`                    | yes        | yes    | yes     |
| `supports_subplots`                     | yes        | yes    | yes     |
| `supports_colorgradient`                | **no**     | yes    | yes     |
| `supports_animation_output` (gif / mp4) | no         | no     | **yes** |
| `supports_native_traces` (`model3d`)    | yes        | yes    | **no**  |

Rough guidance: matplotlib for static figures in papers and docs, plotly for
interactive notebook work, pyvista for large meshes, saved animations, and
high-quality 3D. Matplotlib's 3D compositing has no real depth sorting, so
overlapping bodies can render misleadingly — prefer plotly or pyvista for dense
scenes.

A backend handed something it never declared support for warns and falls back
rather than producing silently wrong output — custom `style.model3d` traces sent
to pyvista, for instance.

Third-party backends register at runtime with
`magpy.register_backend(name, show_func, **capabilities)`, or ship in a package
under the `magpylib.backends` entry-point group, after which the name is
accepted anywhere a built-in one is.

## Animation

```python
cube.position = np.linspace((0, 0, 0.02), (0, 0, 0.1), 50)
magpy.show(cube, animation=True)
```

`animation=True` plays the object path; `animation=<seconds>` sets the duration.
All three built-in backends animate, but only pyvista can write the animation to
a file (gif or mp4). Check `magpy.defaults.display.animation` for frame-rate and
looping options.

## Subplots

```python
with magpy.show_context(cube, sensor, backend="plotly") as sc:
    sc.show(col=1)  # 3D scene
    sc.show(col=2, output="Bx")  # field plot of the same objects
```

`show_context()` collects several `show()` calls into one figure, with `row` and
`col` placing each panel. `output` selects what a panel draws: `"model3d"` (the
default) or field components such as `"Bx"`, `"Hz"`, or `"B"` for magnitude,
plotted over the path.

## Styling

Every object carries a `style` tree, and any style leaf can be set at
construction with a `style_`-prefixed keyword:

```python
cube = magpy.magnet.Cuboid(
    dimension=(0.01, 0.01, 0.01),
    polarization=(0, 0, 1),
    style_label="rotor magnet",
    style_color="crimson",
    style_magnetization_show=True,
)
cube.style.opacity = 0.5  # or afterwards, by attribute
```

Global defaults live under `magpy.defaults.display.style`, and anything left
unset on an object defers to them. `magpy.defaults.reset()` restores factory
settings — useful between test cases, since defaults are process-wide state.

## Custom 3D models

`obj.style.model3d` carries extra geometry drawn in place of, or alongside, the
default body — a dashed outline on a `CustomSource`, a coil former, a housing.
Build traces with the helpers in `magpy.graphics.model3d`: `make_Cuboid`,
`make_Prism`, `make_Pyramid`, `make_Ellipsoid`, `make_CylinderSegment`,
`make_Tetrahedron`, `make_TriangularMesh`, `make_Arrow`; or pass a
backend-native trace via `magpy.graphics.Trace3d`. Set
`style.model3d.showdefault = False` to replace the default representation rather
than add to it. Pyvista does not consume these (see the capability table above).

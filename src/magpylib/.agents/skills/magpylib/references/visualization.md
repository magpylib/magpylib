# Visualisation

Read this when producing figures, animating paths, building subplots, or styling
objects.

## show()

```python
magpy.show(
    *objects,
    backend=None,
    canvas=None,
    animation=False,
    zoom=0,
    markers=None,
    return_fig=False,
    row=None,
    col=None,
    style=None,
)
```

`show()` takes any number of sources, sensors, and collections. It is also a
method on every object (`cube.show()`), and `magpy.show(cube, sensor)` draws
them in one scene.

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

Rough guidance: matplotlib for static figures in papers and docs, plotly for
interactive notebook work, pyvista for large meshes and high-quality 3D. Note
that matplotlib's 3D compositing has no real depth sorting, so overlapping
bodies can render misleadingly — prefer plotly or pyvista for dense scenes.

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
Not every backend supports animation — matplotlib does not. Check
`magpy.defaults.display.animation` for frame-rate and looping options.

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

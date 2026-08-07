# Prototype three.js backend

A deliberately minimal third-party display backend, written against the public
`magpylib.graphics.backend` contract to find out where that contract is thin.
Not shipped, not tested, not a dependency of anything.

Scope: `mesh3d` traces only, no animation, no subplots.

```bash
python sandbox/threejs/demo.py   # writes cuboid.html / two_cuboids.html
```

`show()` returns a self-contained HTML page pulling three.js from unpkg.

## What worked without friction

- **Registration is one call.**
  `magpy.register_backend("threejs", show_func, ...)` with no changes anywhere
  in Magpylib, and `show(backend="threejs")` works immediately.
- **`mesh3d` maps onto `BufferGeometry` directly.** `x`/`y`/`z` interleave into
  `position`, `i`/`j`/`k` into the index buffer. No reshaping beyond that.
- **`supports_colorgradient = True` pays off immediately.** three.js does vertex
  colours natively, so declaring it means Magpylib hands over the gradient
  unsliced: `intensity` (per vertex) plus `colorscale`, sampled once in Python
  into a colour attribute. One mesh per object instead of one per colour band.
- **`merge_traces = False` gives addressable objects.** Every trace carries
  `object_id`, so each becomes one `THREE.Mesh` the host page can look up.
- **`Panel.ranges` frames the camera** without needing to walk the geometry.
- **Every capability warning fired correctly** — animation fell back, subplots
  fell back, undeclared trace types and unaccepted options both warned. Nothing
  had to be discovered by reading Magpylib's source.

## Limitations found

### 1. Under default settings, geometry cannot be spliced

The one that matters for an interactive host. Measured through this backend's
own payload, changing only what _else_ is in the scene:

| setting               | object's own vertices, alone → with a distant object | spliceable |
| --------------------- | ---------------------------------------------------- | ---------- |
| `units_length='auto'` | `0.5` → `0.0005`                                     | no         |
| `units_length='m'`    | `0.5` → `0.5`                                        | yes        |
| `sizemode='scaled'`   | `2.25` → `502.1` (a sensor)                          | no         |
| `sizemode='absolute'` | `3` → `3`                                            | yes        |

So a magnet's own vertex coordinates change by 1000x because an unrelated object
moved the scene across an SI-prefix boundary. Both switches must be pinned;
either alone still leaves geometry scene-dependent. Re-rendering one object and
splicing it in by `object_id` is only valid once both are.

### 2. `mesh3d`-only covers more than expected, but not currents

| objects                                                | trace types |
| ------------------------------------------------------ | ----------- |
| Cuboid, Cylinder, CylinderSegment, Sphere, Tetrahedron | `mesh3d`    |
| Dipole, Sensor, Triangle                               | `mesh3d`    |
| Circle, Polyline (currents)                            | `scatter3d` |

A mesh-only backend therefore covers the entire magnet catalogue plus sensors
and dipoles. Currents are the only gap, and `handles_traces` warns about them
rather than dropping them silently. Worth knowing when scoping a real backend:
`scatter3d` is the second and last thing to implement, and its `mode` is a
combination (`"markers+text+lines"` occurs), not an enum.

### 3. `return_fig` is advisory, and nothing enforces it

`magpy.show()` passes the backend's return value back to the caller **whether or
not `return_fig` is set**. A backend that ignores `scene.return_fig` still
"works", but leaks its figure object into the caller's REPL on every call. This
prototype did exactly that until it was fixed. Nothing warns, and it is easy to
miss because the built-in backends all handle it.

### 4. `canvas` has no meaning for a browser backend

`Scene.canvas` is a Python object — a Matplotlib `Figure`, a Plotly `Figure`, a
PyVista `Plotter`. A three.js render target is a DOM element that does not exist
in the Python process. So:

- `show(canvas=...)` cannot be supported; this backend raises.
- `infer_backend(canvas)` can never resolve to a JS backend, so `backend='auto'`
  will never pick one.

There is no capability flag for "cannot accept a canvas", so the only way to
report it is to raise from inside `show`.

### 5. No legend contract

Traces carry `name`, `legendgroup` and `showlegend`, and `Scene.legend_maxitems`
is handed over, but building a legend is entirely the backend's problem — this
prototype hand-rolls an HTML overlay. `api.py` tells a backend author not to
_change_ what `legendgroup` means, without saying what it means: which traces
are expected to collapse into one entry, and how `legend_maxitems` is meant to
be applied.

### 6. Axis labels have no primitive

`Panel.labels` gives `{'x': 'x (m)', ...}`, which the three built-in backends
render for free. three.js has no text in `AxesHelper`; labels need sprites or an
HTML overlay. Minor, but it is real work the contract implies and does not help
with.

### 7. The up-axis is not stated anywhere

Magpylib scenes are z-up, so `camera.up.set(0, 0, 1)` is required or every view
is wrong. That is not in the backend documentation; it was inferred from the
built-in backends. One sentence in the "easy to miss" list would cover it.

## Not hit, but expected later

Screen-space sizing for sensors and dipoles — three.js can keep them at constant
pixel size while zooming, which `sizemode` approximates statically. Irrelevant
for cuboids, so this prototype does not exercise it.

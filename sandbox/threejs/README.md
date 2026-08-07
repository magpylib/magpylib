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

### 3. `return_fig` was advisory and unenforced — since fixed

`magpy.show()` used to return the backend's value to the caller **whether or not
`return_fig` was set**, contradicting its own documented return value. All three
built-in backends return `None` there of their own accord, so the promise held
by convention rather than by construction, and nothing in the suite covered it.
This prototype was the first thing to break it: every `show()` call leaked an
HTML string into the caller.

Fixed on the parent branch — the dispatcher now discards the figure unless
`return_fig` is set, and five tests that had been resting on the leak were
corrected with it. The half that stays with the backend is the _other_ decision
`scene.return_fig` drives: whether to display. This backend still reads it to
choose between opening a browser tab and staying quiet.

### 4. `canvas` has no meaning for a browser backend

`Scene.canvas` is a Python object — a Matplotlib `Figure`, a Plotly `Figure`, a
PyVista `Plotter`. A three.js render target is a DOM element that does not exist
in the Python process. So:

- `show(canvas=...)` cannot be supported; this backend raises.
- `infer_backend(canvas)` can never resolve to a JS backend, so `backend='auto'`
  will never pick one.

There is no capability flag for "cannot accept a canvas", so the only way to
report it is to raise from inside `show`. The parent branch documents that
rather than adding a fifth flag for one hypothetical backend; worth revisiting
if a second such backend ever appears.

### 5. No legend contract — since documented

Traces carry `name`, `legendgroup` and `showlegend`, and `Scene.legend_maxitems`
is handed over, but building a legend is entirely the backend's problem — this
prototype hand-rolls an HTML overlay. `api.py` told a backend author not to
_change_ what `legendgroup` means without saying what it means.

The parent branch now states it: traces sharing a `legendgroup` are one entry,
which is how a `Collection`'s children collapse into a single row.
`legend_maxitems` turned out to be worse than undocumented — it hides the legend
entirely past that count rather than truncating, plotly ignores it outright, and
negative values reserve legend space without drawing one. That is
[#975](https://github.com/magpylib/magpylib/issues/975).

### 6. Axis labels have no primitive

`Panel.labels` gives `{'x': 'x (m)', ...}`, which the three built-in backends
render for free. three.js has no text in `AxesHelper`; labels need sprites or an
HTML overlay. Minor, but it is real work the contract implies and does not help
with. Noted in the parent's "easy to miss" list; nothing more to do about it.

### 7. The up-axis was not stated anywhere — since documented

Magpylib scenes are z-up, so `camera.up.set(0, 0, 1)` is required or every view
is silently wrong. It had to be inferred from the built-in backends; the parent
branch now says so outright.

## Not hit, but expected later

Screen-space sizing for sensors and dipoles — three.js can keep them at constant
pixel size while zooming, which `sizemode` approximates statically. Irrelevant
for cuboids, so this prototype does not exercise it.

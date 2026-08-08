# Prototype three.js backend

A deliberately minimal third-party display backend, written against the public
`magpylib.graphics.backend` contract to find out where that contract is thin.
Not shipped, not tested, not a dependency of anything.

Scope: `mesh3d` and `scatter3d`, which between them cover every Magpylib object.
No animation, no subplots.

The backend is the two modules at the top level; everything in `examples/` is an
ordinary Magpylib script that happens to select it.

```bash
python sandbox/threejs/examples/demo.py         # one page per object kind
python sandbox/threejs/examples/compare.py      # side by side against Plotly
python sandbox/threejs/examples/interactive.py  # the editor: pick, drag, export
python sandbox/threejs/examples/animation.py    # paths, played from the model
```

`show()` returns a self-contained HTML page pulling three.js from unpkg.

## The one thing to internalise first

**The payload is a rendering, not a model.** Ask the object what it _is_; ask
Magpylib how it should _look_. `object_id` is the join between the two, which is
exactly what its docstring says: it is valid for "an interactive viewer holding
the same objects".

This is stated up front because the same mistake was made three separate times
while writing this prototype, each time by trying to recover the model from the
picture:

| tried to derive          | from                    | when the host could just |
| ------------------------ | ----------------------- | ------------------------ |
| an object's origin       | its bounding-box centre | read `obj.position`      |
| an absolute position     | accumulated drag deltas | read `obj.position`      |
| the magnetization vector | the `intensity` array   | read `obj.polarization`  |

Each derivation was wrong, or right only by accident: the bounding-box centre is
off by `0.678` for a Sensor (finding 10), and the intensity formula assumed
convexity. And each was unnecessary — a host that owns the objects has the
answer already.

The same applies in the other direction. To render an edit, do not reimplement
Magpylib's meshing or colouring in the frontend: set the property on the object
and ask Magpylib for that one object again. It costs **0.27–0.37 ms**, about 2%
of a 60fps frame, and cannot drift from Magpylib's own conventions. Only the
things Magpylib is not involved in — a transform, or a scale-covariant resize
preview — belong in the browser.

## What worked without friction

- **Registration is one call.**
  `magpy.register_backend("threejs", show_func, ...)` with no changes anywhere
  in Magpylib, and `show(backend="threejs")` works immediately.
- **`mesh3d` maps onto `BufferGeometry` directly.** `x`/`y`/`z` interleave into
  `position`, `i`/`j`/`k` into the index buffer. No reshaping beyond that.
- **`supports_colorgradient = True` pays off immediately.** Declaring it means
  Magpylib hands over the gradient unsliced — `intensity` per vertex plus a
  `colorscale` — so one mesh per object replaces one per colour band. Consuming
  it correctly is less obvious than it looks; see finding 9.
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
and dipoles. Currents were the only gap, and `handles_traces` warned about them
rather than dropping them silently.

`scatter3d` has since been added, and it was the second and last thing needed —
the prototype now draws every object type Magpylib has. Three `mode` values
occur in practice, and `mode` is a combination rather than an enum, so it has to
be split into tokens:

| mode            | produced by                         |
| --------------- | ----------------------------------- |
| `lines`         | currents                            |
| `markers+lines` | an object's path                    |
| `markers`       | the `markers=` argument to `show()` |

Sensor pixels are _not_ markers, contrary to the obvious guess — a Sensor is
entirely `mesh3d`.

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

### 8. Three line and marker properties have no WebGL primitive

Adding `scatter3d` surfaced a gap on three.js's side rather than Magpylib's. The
payload asks for things plain WebGL cannot draw:

| Magpylib sends  | seen in practice | three.js                                           |
| --------------- | ---------------- | -------------------------------------------------- |
| `line_width`    | `1.0`, `2.0`     | ignored — `LineBasicMaterial` is always 1 px       |
| `line_dash`     | `solid`          | no primitive; `LineDashedMaterial` is uniform-only |
| `marker_symbol` | `o`, `x`         | no primitive; `PointsMaterial` draws squares       |

None of these is a contract problem — Magpylib expresses them perfectly well,
and Pyvista carries ❌ for exactly _line style_ and _marker symbol_ in the
backend feature table, so a three.js column would look much the same.

Line width and marker size are since fixed, and the fix matters beyond them.
`Line2` + `LineMaterial` render lines as camera-facing quads, and
`PointsMaterial` takes `sizeAttenuation: false`; in both cases the size is then
a **pixel count that stays constant under zoom**. That is exactly the
screen-space mechanism sensors and dipoles would want, so it is no longer
hypothetical — the same primitive is already drawing the current loops here.
`line_dash` and `marker_symbol` remain unimplemented.

**Sizes in the payload are nominal, and each backend calibrates them.** Even
with pixel-space lines the widths were still half Plotly's, because
`backend_plotly` carries its own `SIZE_FACTORS_TO_PLOTLY` — `line_width × 2.2`,
`marker_size × 0.7` — and nothing in the contract says what a width of `2`
should look like. A new backend has to calibrate empirically against an existing
one. This prototype borrows Plotly's factors, which transfer directly because
both measure in pixels.

### 9. `mesh3d` carries three colour mechanisms, and two of them are traps

Not a Magpylib defect — the dialect expresses all three perfectly well — but the
obvious reading of each is wrong, and both mistakes render without erroring.
Every one of these was caught by _looking at the output_, not by validation.

| mechanism                | used by | naive handling         | what it needs                    |
| ------------------------ | ------- | ---------------------- | -------------------------------- |
| `color`                  | Dipole  | flat material colour   | correct as read                  |
| `intensity`+`colorscale` | magnets | sample per vertex      | per-fragment lookup texture      |
| `facecolor`              | Sensor  | ignored, falls to flat | de-indexed, per-triangle colours |

**The colorscale is piecewise, so per-vertex sampling loses it.** Magpylib's
default tricolor scale holds green to `0.16`, grey from `0.26` to `0.74`, then
red. A Cuboid has eight vertices whose intensities are all exactly `0` or `1`,
so nothing lands on the grey plateau: sampling the scale in Python and letting
the GPU blend corner colours produces a straight green-to-red ramp with **no
grey band at all**. What has to be interpolated across the face is the
_intensity_, with the colour looked up per fragment — a 256-entry RGBA texture
indexed by an intensity-valued UV. Plotly's shader does the same, which is why
the reference figures look right.

**`facecolor` is a third path, and a trace using it has `color = None`.** A
Sensor is a single `mesh3d` with 216 per-triangle colours: the object's own
colour for the arrow bodies, `red`/`green`/`blue` for the axis heads, `black`
for the pixels. Falling back to `color` yields a uniform blob that looks
plausible enough to miss. Rendering it means giving up the index buffer — one
vertex per triangle corner — which is the same thing
`subdivide_mesh_by_facecolor` does for Matplotlib. Note the values mix CSS names
with hex, so `THREE.Color` parses them rather than a Python helper.

A third bug hid behind these two: `THREE.RGBFormat`, which three.js **removed in
r137**. `node --check` accepts it — it is valid syntax that is merely
`undefined` at runtime. The payload validated, the JS parsed, and the scene
still rendered wrong in three separate ways. That is the case for the
golden-file figure regressions in
[#972](https://github.com/magpylib/magpylib/issues/972).

### 10. The payload carries no object transform — resolve `object_id` instead

Position and orientation are baked into the vertex arrays, and no trace, `Frame`
or `Scene` field carries them. A backend therefore has no anchor to attach a
gizmo to: every mesh arrives with an identity matrix, so a gizmo attached to one
appears at the world origin rather than on the object.

The bounding-box centre is the obvious fallback and is **wrong** for anything
whose origin is not its centroid. Measured on a Sensor at `(0, -3, 0)`, no
heuristic recovers it:

| estimate                     | centre                 | error |
| ---------------------------- | ---------------------- | ----- |
| bbox, all faces              | `0.312, -2.574, 0.425` | 0.678 |
| bbox, excluding black pixels | `0.350, -2.574, 0.425` | 0.696 |
| bbox, RGB arrow heads only   | `0.425, -2.574, 0.425` | 0.737 |

Excluding the pixels makes it _worse_: a Sensor's cross emanates from the origin
along +x, +y, +z, so any bbox estimate is off by roughly half an arrow length by
construction. Translation is unaffected — a delta is a delta — but rotation
orbits the wrong point.

**No API change is needed.** `object_id` is documented as valid for "an
interactive viewer holding the same objects", and a host that owns them keeps a
`{id(obj): obj}` map and reads `position` and `orientation` off the object
directly. That takes the Sensor error from `0.678` to exactly `0`. The anchor
must be scaled by the same `units_length` factor as the geometry.

### 11. Three classes of edit, with very different costs

What an interactive host actually needs to know. Measured:

| class                                           | needs magpylib?               |
| ----------------------------------------------- | ----------------------------- |
| **transform** — position, orientation           | no; one matrix, 0 round-trips |
| **shape** — dimension, diameter                 | yes, but 0.28–0.37 ms/object  |
| **style** — colour, opacity, magnetization mode | depends, see below            |

Regenerating one object costs **0.28–0.37 ms** against a 16.7 ms frame budget,
so a round-trip _per object_ is affordable during a drag. Regenerating the
_scene_ is not: 11.9 ms at 100 objects, 60.4 ms at 500. So a host must re-render
the edited object and splice it by `object_id` — which is only valid once
finding 1 is satisfied.

Better still for the primitives: a dimension change is _exactly_ a scale, so
`mesh.scale` previews it with no magpylib call at all and Python is told only
the final value. `interactive.py` does this for `Cuboid` (free), `Sphere`
(uniform) and `Cylinder` (x/y tied to the diameter). It breaks down for
`CylinderSegment`, whose angles do not scale.

Style splits, and not where you would guess: `magnetization.show` is appearance
only, `magnetization.mode` regenerates geometry — same subtree, opposite cost.
Appearance-only means the vertex buffers are untouched, so the host updates a
material and the LUT texture with no re-upload. Nothing in the API says which is
which, which is [#976](https://github.com/magpylib/magpylib/issues/976).

### 12. Magnetization amplitude renders as nothing

`polarization=(0, 0, 1)` and `(0, 0, 5)` produce **byte-identical** payloads:
same geometry, same `intensity`, same `colorscale`. The colour scheme encodes
polarity, not strength, so magnitude is not visualised at all in colour mode.

Nothing to redraw is convenient, but it also means an editor gets no feedback
from the viewport when the user changes amplitude. That has to be supplied by
the host — a readout, or `magnetization.mode='arrow'`, where the arrow length is
geometry and therefore does change.

Direction is different and cheap: it leaves geometry and colorscale untouched
and changes only the per-vertex `intensity`, which in this backend _is_ the UV
already uploaded for the colorscale lookup. Rewriting one attribute is enough;
positions are not re-uploaded. Re-deriving that array in the browser is tempting
and wrong — see the framing above; ask Magpylib, it costs 0.3 ms.

### 13. A backend cannot reach the objects; only a host can

`object_id` is documented as valid for "an interactive viewer holding the same
objects", and finding 10 leans on exactly that. But it draws a line that is easy
to miss: a **host** holds the objects, because it calls `show` and owns them. A
**backend** does not. `show(scene)` hands over a `Scene` and nothing else -- no
objects, and no transform anywhere in the payload.

That is fine for magpylib-studio, which is the host. It is not fine for a
backend shipped in a package, which is the case the entry-point group exists to
support: an editor needs an object's origin to anchor a gizmo, and its
polarization to aim one, and has no supported way to get either.

This prototype resolves the token back through
`ctypes.cast(oid, ctypes.py_object).value`. It returns the original object --
verified `is` the one passed to `show` -- and is safe _at that moment_ because
magpylib is holding every object in order to draw it. It is also
CPython-specific and exactly the kind of thing a public API should make
unnecessary.

The two honest fixes are for the host to do the work (what studio will do), or
for magpylib to pass the objects alongside the scene. Adding the transform to
the payload, which was the first idea, would cover the gizmo case and not the
polarization one.

## Not hit, but expected later

Screen-space sizing for sensors and dipoles — three.js can keep them at constant
pixel size while zooming, which `sizemode` approximates statically. The
mechanism is no longer speculative: `LineMaterial` with `worldUnits: false` and
`PointsMaterial` with `sizeAttenuation: false` already size this prototype's
lines and markers in pixels (finding 8). Applying it to autosized meshes needs a
reference size, which is the open question.

Also unimplemented: `line_dash` and `marker_symbol` (finding 8), axis labels
(finding 6), animation and subplots.

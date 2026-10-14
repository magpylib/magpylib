---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
orphan: true
---

(guide-graphics-custom-backend)=

# Writing a custom display backend

The three built-in backends are not privileged: Matplotlib, Plotly and
Pyvista all go through the same public contract a third party uses. This
page is for someone writing another one; see [](guide-graphics) for using
the built-in backends.

```{code-cell} ipython3
import magpylib as magpy
from magpylib.graphics.backend import DisplayBackend


class CounterBackend(DisplayBackend):
    name = "counter"
    description = "Reports what it was handed instead of drawing it"
    api_version = 1
    supports_animation = False

    def show(self, scene):
        traces = sum(len(frame.traces) for frame in scene.frames)
        return f"{len(scene.frames)} frame(s), {traces} trace(s)"


src = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
print(magpy.show(src, backend="counter", return_fig=True))
```

Declaring the class registers it, and its `description`, with a `url` if you set one, is shown where Magpylib lists the backends, such as the error for an unknown backend name. A backend shipped in a package should instead advertise itself in the `magpylib.backends` entry-point group, so that installing it is enough:

```toml
[project.entry-points."magpylib.backends"]
counter = "my_package:CounterBackend"
```

Entry points are resolved lazily, the first time a backend name is looked up after `import magpylib` has returned — so magpylib is fully imported before your module runs, and `from magpylib.graphics.backend import DisplayBackend` at module scope is safe. An entry point may name the class or the module that defines it; if loading it registers no backend, magpylib says so rather than leaving the name quietly missing. For a backend defined in a script or notebook, `magpy.register_backend(name, show_func, ...)` is the imperative equivalent.

## What `show` receives

A `Scene` carries everything needed to draw:

| attribute                              | meaning                                                                                                     |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `panels`                               | the subplot grid; each `Panel` has `row`, `col`, `kind` (`"scene3d"` or `"chart2d"`), `ranges`, `labels` and `objects` |
| `frames`                               | the timeline; one `Frame` for a static scene, several when animating                                        |
| `animation`                            | `fps`, `time`, `slider`, `output`, `frame_duration`, `path_indices`, ...                                    |
| `canvas`, `canvas_update`              | a user-supplied figure to draw into, and whether it may be restyled                                         |
| `return_fig`                           | whether to display the figure, see below                                                                    |
| `legend_maxitems`                      | most entries a legend lists; past it, list the first ones and sum up the rest in one entry. 0: no legend    |
| `options`                              | arguments Magpylib does not interpret, forwarded verbatim; `fig_*` and `show_*` among them, see below       |

Each `Frame` has `label`, `traces` and `native_traces`. Every frame carries the same traces in the same order, so a trace can be followed through an animation by its position.

## The trace dialect

Traces are plain dictionaries, deliberately: a new key or a whole new trace type then needs no API change. Keys use the magic underscore convention (`marker_line_color` means `marker.line.color`), so `magpylib.graphics.backend.magic_to_dict` turns them into nested form if that suits the backend better.

Every trace has `type`, `row`, `col`, `name`, `legendgroup` and `showlegend`. Beyond that:

- **`type="mesh3d"`** — `x`, `y`, `z` vertex arrays of floats and `i`, `j`, `k` face indices as integer arrays, plus `opacity`, an optional `flatshading=True` asking for flat faces rather than normals smoothed across edges (a TriangularMesh sets it), and one of three _mutually exclusive_ colourings:
  - a flat `color`;
  - `intensity` per vertex with a `colorscale`, used for magnetization. The scale is **piecewise** — the default tricolor one holds green to `0.16`, grey from `0.26` to `0.74`, then red — so it must be applied per fragment, by interpolating the intensity. Sampling it per vertex instead loses every plateau: a Cuboid's eight vertices are all at intensity `0` or `1`, so none of them lands on the grey band and the result is a flat green-to-red ramp;
  - `facecolor`, one colour per triangle, used where a single object needs several — a Sensor's arrow bodies, its red/green/blue axis heads and its black pixels arrive as one trace. Values mix CSS names with hex. **A trace using `facecolor` has `color = None`**, so a backend that reads only `color` renders it as a uniform blob without erroring.
- **`type="scatter3d"`** — `x`, `y`, `z`, arrays of floats, and a `mode` combining `"lines"`, `"markers"` and `"text"`, with `line_*` and `marker_*` styling.
- **`type="scatter"`** — a 2D curve in a `chart2d` panel, with `x` and `y`. These carry no `opacity`; their `hovertemplate`, `legendgrouptitle_text` and `text` are Plotly-flavoured hints a backend may ignore.

Values follow Magpylib's own vocabularies rather than any one library's: marker symbols come from `magpylib.graphics.backend.ALLOWED_SYMBOLS` and dash styles from `ALLOWED_LINESTYLES`, both Matplotlib-derived. Every built-in backend translates them — see `SYMBOLS_TO_PLOTLY` and `SYMBOLS_TO_PYVISTA`.

Traces also carry Magpylib **metadata** that is not a drawing property: `object_id` identifies the object a trace came from, for picking and per-object manipulation. It is `None` on a trace merged across several objects, and process-local — never persist or transmit it. On a 2D curve it names the object the curve is named after: its sensor, or its source when sources are not summed up. More metadata keys may follow in a later release, so strip them with `drawing_properties` rather than by name.

`object_id` is also how a caller recovers an object's **transform**, which the payload does not contain: position and orientation are baked into the vertex arrays, and no trace carries them separately. A host that holds the same objects — the case `object_id` exists for — resolves the token against them and reads `position` and `orientation` from the object itself. This matters for anything that manipulates an object rather than only drawing it: a gizmo anchored on a mesh's bounding-box centre rotates about the wrong point for any object whose origin is not its centroid, a Sensor being the obvious case.

`row` and `col` place a trace in the subplot grid; no plotting library takes them as trace arguments either. `drawing_properties` strips both the metadata and the placement (`TRACE_META_KEYS`, `PLACEMENT_KEYS`), so read the placement first:

```python
from magpylib.graphics.backend import drawing_properties

row, col = trace["row"], trace["col"]
fig.add_trace(go.Mesh3d(**drawing_properties(trace)), row=row, col=col)
```

Declaring `merge_traces = False` stops Magpylib merging traces _across_ objects, so that every object keeps its own geometry and identity — what picking and drag gizmos need. It affects only objects that would otherwise merge, which in practice means `Collection` children under default styling; separately styled objects are never merged, because each gets its own colour. Merges _within_ one object are always applied.

A few things are easy to miss:

- **`frame.native_traces` must be consumed.** When a user attaches a model through `style.model3d.data` naming your backend, it is routed there rather than into `frame.traces`, already positioned and oriented. Ignoring the list silently drops the user's models. Declare `supports_native_traces = False` if you do not handle them, and Magpylib will warn on your behalf.
- **2D traces.** With `output="Bx"` rather than `"model3d"`, a `chart2d` panel holds `scatter` curves, with the path index as `x` and the field as `y`. The panel's `labels` title its axes, e.g. `{"x": "path index", "y": "Bx, Bz (T)"}`; the outputs of one panel always share a unit. In an animation every frame repeats each curve and only `marker_size` changes, enlarging the point at that frame's path index, so a backend that draws a curve once can place its own cursor from `scene.animation.path_indices`. A pure-3D backend has no answer for these: leave `"scatter"` out of `handles_traces` and Magpylib warns.
- **Declare `handles_traces`** if you draw only some trace types. Magpylib then warns about a type you never handle instead of producing a quietly incomplete figure — which is what makes new trace types safe to add.
- **Declare `accepts_options`**, the arguments your backend takes beyond Magpylib's own, and a misspelled one is reported instead of dropped. A trailing `*` accepts a namespace. `fig_*` and `show_*` are two conventional ones: the built-in backends forward them to their library's figure constructor and show call, read through `scene.prefixed_options("fig")`. A backend with no such library leaves them undeclared, so that a user's `fig_width=400` is reported as the typo it is there. Names Magpylib interprets itself cannot be claimed, and declaring one is reported: `show()`'s own arguments, and any name of `magpy.defaults.display` (`style`, `animation`, `units`, `legend`, `colorsequence`, `autosizefactor`, `backend`), bare or as a `<name>_` prefix. A `<your name>_` prefix is stripped before your backend sees an option, so declare the bare name. The library's own error is what checks a forwarded `fig_*` or `show_*`: it keeps its type and gains a note naming the `show()` argument.
- **Geometry is in world coordinates, and under default settings all of it depends on the whole scene.** Vertex arrays already have the object's position and orientation applied. Two further scalings are derived from the extent of the _whole_ scene, so re-rendering one object on its own and splicing it into an existing figure by `object_id` yields wrong geometry:
  - **Autosize** scales sensors and dipoles, which have no physical size of their own, by the scene extent. It varies continuously, so moving any object rescales every autosized object.
  - **`units_length="auto"`** picks an SI prefix from the scene extent and rescales _every_ trace by the corresponding factor. It is quantized — nothing changes until the extent crosses a decade boundary, at which point the entire scene jumps by 1000.

  A backend that re-renders whole scenes need not care. One that splices or updates incrementally does, and both scalings can be pinned — per call, or globally through `magpy.defaults.display`:

  ```python
  magpy.defaults.display.style.sensor.sizemode = "absolute"  # sizes become physical
  magpy.defaults.display.style.dipole.sizemode = "absolute"
  magpy.defaults.display.units.length = "m"  # or show(..., units_length="m")
  ```

  Both are needed; either alone still leaves the geometry scene-dependent.

- **`scene.return_fig` decides whether to _display_, not what to return.** Magpylib discards whatever `show` returns unless it is set, so the question it answers is whether to call your library's blocking show/display — returning the figure either way is fine.
- **`scene.canvas` is a Python object** — a Matplotlib `Figure`, a Plotly `Figure`, a PyVista `Plotter`. A backend that renders outside the Python process, in a browser for instance, has nothing to bind it to and should raise. For the same reason `backend="auto"`, which infers the backend from the canvas type, will never select one.
- **The objects are there when the traces are not enough.** `Panel.objects` holds what was passed to `show` for that cell, collections whole. It is how a backend recovers what no trace carries: the `Collection` hierarchy — `legendgroup` names only the _outermost_ collection, so three levels of nesting arrive looking like one — and the object behind an `object_id`, by matching `id(obj)`. Lists are unpacked, so an object passed on its own and inside a collection passed with it appears both ways. They are live objects: a backend that keeps them keeps them alive.
- **Scenes are z-up, and the legend is yours to draw.** Camera code needs the z-up convention stated explicitly. `Panel.labels` carries the axis titles and every trace carries `name`, `legendgroup` and `showlegend`, but nothing renders them for you: traces sharing a `legendgroup` belong to one legend entry — that is how a `Collection`'s children collapse into a single row; the group is a stable name such as `"Cuboid#0"`, the outermost object's class and position, so the same scene gives the same payload on every run — and `Scene.legend_maxitems` is the most entries a flat legend should list — the built-in backends list that many and sum up the rest in one "… and k more" entry. The cap is never applied to the traces themselves, so a backend whose legend scales another way, such as a collapsible tree, can ignore it.

Backends drawing sensor pixels their own way, as instances in a scene graph for example, can size and colour them as the built-ins do: `sensor_pixel_fields(objects)` gives the field each sensor's pixels read, `scale_pixel_norms(norms, scaling, minimum=...)` turns field norms into the factors `style.pixel.field.sizescaling` and `colorscaling` ask for, and `colors_from_colorscale(values, colorscale)` maps values to hex colours. All three are in `magpylib.graphics.backend`.

Capability flags (`supports_animation`, `supports_subplots`, `supports_colorgradient`, `supports_animation_output`) all default to `False`, so a capability added in a later Magpylib release never changes an existing backend's behaviour. `show()` warns and falls back rather than handing over something undeclared.

Set `api_version = 1`, the version of this contract you wrote against. Left unset, a backend counts as written against version 1 and is warned once the contract moves on. The version goes up only when a backend written against the previous one could misread what it receives: a field, a trace key or a trace type removed, renamed or given another meaning. Additions never bump it.

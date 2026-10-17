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

:::{versionadded} 6.0
Third-party display backends
:::

The three built-in backends are not privileged: Matplotlib, Plotly and Pyvista all go through the same public contract a third party uses. This page is for someone writing another one; see [](guide-graphics) for using the built-in backends. Everything it names is importable from `magpylib.graphics.backend`.

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

For a complete backend in real use, read the three.js backend of [Magpylib Studio](https://github.com/magpylib/magpylib-studio), in [`magpylib_studio/threejs.py`](https://github.com/magpylib/magpylib-studio/blob/main/magpylib_studio/threejs.py). It turns the `Scene` into buffers for a browser scene graph that is built once and then edited object by object. Several examples below come from it.

## Registering a backend

A subclass that sets its own `name` registers itself at the class statement, instantiated without arguments: configure a backend through class attributes or the options a user passes, not through `__init__`. A subclass without a `name` registers nothing, so it can hold declarations that several backends share. The `description`, with the `url` if you set one, is shown where Magpylib lists the backends, such as the error for an unknown backend name.

A backend shipped in a package should instead advertise itself in the `magpylib.backends` entry-point group (`ENTRY_POINT_GROUP`), so that installing it is enough:

```toml
[project.entry-points."magpylib.backends"]
counter = "my_package:CounterBackend"
```

Entry points are resolved lazily, the first time a backend name is looked up after `import magpylib` has returned. So magpylib is fully imported before your module runs, and `from magpylib.graphics.backend import DisplayBackend` at module scope is safe. An entry point may name the class or the module that defines it. The backend registers under its class's `name`, so keep the entry-point key equal to it: Magpylib warns when they differ. A class without a `name` registers under the entry-point key. If loading fails, or registers no backend, Magpylib warns and carries on without it.

For a backend defined in a script or notebook, `magpy.register_backend(name, show_func, **declarations)` is the imperative equivalent. `show_func(scene)` plays the part of `show`. The keywords are the class declarations: `api_version`, `description`, `url`, `supports_animation`, `supports_subplots`, `supports_colorgradient`, `supports_animation_output`, `supports_native_traces`, `merge_traces`, `handles_traces` and `accepts_options`. Any other name raises `TypeError`. It returns the generated class.

The declaration is checked where it is written, at the class statement or in `register_backend`:

- `name`, `description` and `url` are strings, the flags are `True` or `False`, `api_version` is an integer, and `handles_traces` and `accepts_options` are sets of strings or `None`. Anything else raises `TypeError`.
- A misspelled flag on a class, such as `supports_animations = True`, warns; `register_backend` rejects it, as above.
- Taking a name that a backend from another module holds replaces that backend, with a warning. Re-running the same notebook cell stays quiet.
- Some names are reserved, see [](guide-graphics-custom-backend-reserved).

## What `show` receives

`show(self, scene)` receives one `Scene` and returns the figure. All of `Scene`, `Panel`, `Frame` and `AnimationSettings` are frozen dataclasses.

| `Scene` field     | type                  | meaning                                                                                                                                             |
| ----------------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `panels`          | tuple of `Panel`      | the subplot grid, one panel per cell                                                                                                                |
| `frames`          | tuple of `Frame`      | the timeline: one frame for a static scene, one per step when animating                                                                             |
| `title`           | `str` or `None`       | `show(title=...)`; without it, the `style.label` of a lone object, `"No objects to be displayed"` for an empty `show()`, else `None`                 |
| `animation`       | `AnimationSettings`   | timing, see [](guide-graphics-custom-backend-animation)                                                                                             |
| `canvas`          | any object, or `None` | the user's `canvas=`, passed on unchecked                                                                                                           |
| `canvas_update`   | `bool`                | whether you may restyle the canvas: `True` without a canvas, `False` with one unless the user passed `canvas_update=True`                          |
| `return_fig`      | `bool`                | whether the caller wants the figure back rather than displayed, see [](guide-graphics-custom-backend-details)                                       |
| `legend_maxitems` | `int`                 | the most entries a legend should list, see [](guide-graphics-custom-backend-legend)                                                                 |
| `options`         | `dict`                | the `show()` arguments Magpylib does not interpret, see [](guide-graphics-custom-backend-options)                                                   |

`Scene` also has `is_animation`, `n_rows`, `n_cols`, `has_subplots`, `panel(row, col)`, `panel_kind(row, col)` and `prefixed_options(prefix)`.

| `Panel` field | type                          | meaning                                                                                                                                                              |
| ------------- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `row`, `col`  | `int`                         | the cell, 1-based                                                                                                                                                    |
| `kind`        | `"scene3d"` or `"chart2d"`    | a 3D scene, or a 2D field plot over the path                                                                                                                         |
| `ranges`      | float array `(3, 2)`, or `None` | `(min, max)` of x, y and z, in the display unit. The three spans are equal, and `zoom` is applied (`zoom=1` doubles them). `None` for a `chart2d` panel              |
| `labels`      | `dict[str, str]`              | axis titles: `{"x": "x (mm)", "y": "y (mm)", "z": "z (mm)"}` for `scene3d`, `{"x": "path index", "y": "Bx, Bz (T)"}` for `chart2d`                                   |
| `objects`     | tuple                         | the objects passed to `show` for this cell, see [](guide-graphics-custom-backend-details)                                                                            |

| `Frame` field   | type            | meaning                                                                                        |
| --------------- | --------------- | ---------------------------------------------------------------------------------------------- |
| `label`         | `str`           | display text: the 1-based path index of the frame (`"1"`, `"2"`, ...), `"0"` for a static scene |
| `traces`        | tuple of `dict` | what to draw, in the trace dialect below                                                       |
| `native_traces` | tuple of `dict` | user models in your own format, see [](guide-graphics-custom-backend-native)                   |

Every frame carries the same traces in the same order, so a trace can be followed through an animation by its position.

## The trace dialect

Traces are plain dictionaries, deliberately: a new key or a whole new trace type then needs no API change. Keys use the magic underscore convention (`marker_line_color` means `marker.line.color`), so `magic_to_dict` turns them into nested form if that suits the backend better.

Every trace has these keys:

| key           | type            | meaning                                                                          |
| ------------- | --------------- | -------------------------------------------------------------------------------- |
| `type`        | `str`           | `"mesh3d"`, `"scatter3d"` or `"scatter"`                                         |
| `row`, `col`  | `int`           | the panel the trace belongs to                                                   |
| `name`        | `str`           | the legend text                                                                  |
| `legendgroup` | `str`           | the legend entry (3D) or legend group (2D) of the trace                          |
| `showlegend`  | `bool`          | whether this trace makes a legend entry                                          |
| `object_id`   | `int` or `None` | the object the trace comes from, see [](guide-graphics-custom-backend-metadata)  |

A 3D trace also has `opacity`, from 0 to 1.

**`type="mesh3d"`**, a triangle mesh:

| key           | type                                 | meaning                                                                                       |
| ------------- | ------------------------------------ | --------------------------------------------------------------------------------------------- |
| `x`, `y`, `z` | float arrays `(n,)`                  | vertex coordinates                                                                            |
| `i`, `j`, `k` | integer arrays `(m,)`                | the three vertex indices of each triangle                                                     |
| `color`       | `str` or `None`                      | the object's base colour: its legend swatch, and the fallback. `None` when `facecolor` is set |
| `intensity`   | float array `(n,)`, optional         | one value per vertex, already in [0, 1], to colour with `colorscale`                          |
| `colorscale`  | tuple of `(stop, colour)`, optional  | comes with `intensity`; stops from 0 to 1                                                     |
| `facecolor`   | array of `str`, `(m,)`, optional     | one colour per triangle                                                                       |
| `flatshading` | `bool`, optional                     | `True` asks for flat faces rather than normals smoothed across edges (a TriangularMesh)       |

The colourings are not exclusive. `intensity` with `colorscale` wins, then `facecolor`, then `color`: a magnet's mesh carries both `intensity` and its base `color`.

- **The colorscale is piecewise.** The default tricolor scale holds green to `0.16`, grey from `0.26` to `0.74`, then red. Apply it per fragment: interpolate the intensity across each face and look the colour up there. Sampling it per vertex loses every plateau: a Cuboid polarized along z has its eight vertices at intensity `0` or `1`, none on the grey band, and the result is a flat green-to-red ramp. Studio, like Plotly's shader, samples the scale into a lookup texture and hands the intensity over as a texture coordinate:

  ```python
  lut = colors_from_colorscale(
      np.linspace(0, 1, 256), trace["colorscale"], cmin=0, cmax=1
  )
  uv = np.stack([trace["intensity"], np.full(len(trace["x"]), 0.5)], axis=1)
  ```

- **`facecolor` is for one object of several colours.** A Sensor's axes, arrow heads and pixels arrive as one trace. So does a magnet sliced into colour bands for a backend without colour gradients, see [](guide-graphics-custom-backend-capabilities). Such a trace has `color = None`, so a backend that reads only `color` draws it wrong without an error.

**`type="scatter3d"`**, points and lines:

| key                                            | type                  | meaning                                                                                   |
| ---------------------------------------------- | --------------------- | ----------------------------------------------------------------------------------------- |
| `x`, `y`, `z`                                  | float arrays          | the points. NaN separates line segments: lift the pen there                               |
| `mode`                                         | `str`                 | `"lines"`, `"markers"` and `"text"` joined by `+` in any order, such as `"markers+lines"` |
| `line_color`, `line_width`, `line_dash`        | `str`, number, dash   | with `lines`                                                                              |
| `marker_color`, `marker_size`, `marker_symbol` | `str`, number, symbol | with `markers`                                                                            |
| `text`                                         | list, optional        | with `text`: one label per point (path numbering)                                         |

On a sensor's pixel trace, drawn when its pixels are 2D symbols or `arrow`s, `line_color`, `marker_color` and `marker_size` are arrays with one value per point.

**`type="scatter"`**, a 2D curve in a `chart2d` panel, see [](guide-graphics-custom-backend-2d):

| key                                 | type            | meaning                                                                   |
| ----------------------------------- | --------------- | ------------------------------------------------------------------------- |
| `x`                                 | integer array   | the path index                                                            |
| `y`                                 | float array     | the field value, in the unit of the panel's y label                       |
| `mode`                              | `str`           | `"lines+markers"`                                                         |
| `line_color`, `line_dash`           | `str`, dash     | as in 3D                                                                  |
| `marker_size`                       | integer array   | one size per point: the point at the frame's path index is drawn larger   |
| `marker_color`, `marker_line_color` | `str` or `None` | `None` in some layouts: then use `line_color`                             |
| `marker_symbol`                     | symbol          | as in 3D                                                                  |
| `legendgrouptitle_text`             | `str`           | the title of the curve's legend group                                     |
| `hovertemplate`, `text`             | `str`           | Plotly hover hints, which a backend may ignore                            |

A 2D curve has no `opacity` and no `line_width`.

(guide-graphics-custom-backend-values)=

### Value types

- **Coordinates** are float64 numpy arrays, in world coordinates and already in the display unit. Face indices are int64 arrays.
- **Numbers** may be Python ints, such as `opacity=1` or `line_width=2`: convert with `float()` where your library needs floats.
- **Colours** are CSS names or `"#rrggbb"` hex: a user's `"r"` arrives as `"red"`. Per-face and per-point colours are a numpy string array from Magpylib, and a plain list from a user's model.
- **Symbols and dashes** follow Magpylib's own vocabularies, both Matplotlib-derived: `marker_symbol` is one of `ALLOWED_SYMBOLS` (`"o"`, `"D"`, `"+"`, ...) and `line_dash` one of `"solid"`, `"dashed"`, `"dotted"`, `"dashdot"`, `"loosely dotted"` and `"loosely dashdotted"`, whichever of the spellings in `ALLOWED_LINESTYLES` the user chose. Map them to your library. The exception is a user's `generic` [model](examples-own-3d-models), whose values pass through unchecked and may be Plotly names such as `"circle"`.
- **Sizes are nominal.** `line_width=2` and `marker_size=3` have no unit, and every backend scales them its own way. Plotly multiplies line widths by 2.2 and marker sizes by 0.7, in pixels; Studio uses the same factors.
- **NaN is not JSON.** A backend that ships coordinates to a browser must encode it, as `null` for instance, and restore it there.
- **Optional keys** appear only when they apply: `text`, `flatshading`, `intensity`, `colorscale`, `facecolor`. Read them with `trace.get`.

(guide-graphics-custom-backend-metadata)=

### Metadata and placement

Traces also carry Magpylib **metadata** that is not a drawing property. `object_id` identifies the object a trace came from, for picking and per-object manipulation. It is `None` on a trace merged across several objects and on the `show(markers=...)` trace. On a 2D curve it names the object the curve is named after: its sensor, or its source when sources are not summed up. It is an `id()`: process-local and different on every run, so never persist or transmit it. More metadata keys may follow in a later release (`TRACE_META_KEYS`), so strip them with `drawing_properties` rather than by name.

Studio keeps one scene-graph node per object. It maps each `object_id` to a key of its own, through the objects it holds, and hangs every trace with that key on the node, so that a click selects an object rather than one of its traces.

`object_id` is also how a caller recovers an object's **transform**, which the payload does not contain: position and orientation are baked into the vertex arrays, and no trace carries them separately. A host that holds the same objects, the case `object_id` exists for, resolves the token against them and reads `position` and `orientation` from the object itself. This matters for anything that manipulates an object rather than only drawing it: a gizmo anchored on a mesh's bounding-box centre rotates about the wrong point for any object whose origin is not its centroid, a Sensor being the obvious case.

`row` and `col` place a trace in the subplot grid; no plotting library takes them as trace arguments either. `drawing_properties` strips both the metadata and the placement (`TRACE_META_KEYS`, `PLACEMENT_KEYS`), so read the placement first:

```python
import plotly.graph_objects as go
from magpylib.graphics.backend import drawing_properties

row, col = trace["row"], trace["col"]
fig.add_trace(go.Mesh3d(**drawing_properties(trace)), row=row, col=col)
```

Here `fig` must come from `plotly.subplots.make_subplots`: a plain `go.Figure()` rejects `row` and `col`. The dialect looks like Plotly's but is not Plotly's: the same line fails for a `scatter3d` trace whose `marker_symbol` is `"o"`, until the symbol is translated.

### Which traces each object gives

| object                                                                         | traces                                                                                                                                           |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| magnet: `Cuboid`, `Cylinder`, `CylinderSegment`, `Sphere`, `Tetrahedron`, `TriangularMesh` | one `mesh3d`, plus a `scatter3d` arrow when the magnetization is shown as an arrow; the default mode, `"auto"`, picks colour with `supports_colorgradient` and the arrow without |
| `Triangle`                                                                     | a `mesh3d`, plus a grey `mesh3d` arrow                                                                                                           |
| `Circle`, `Polyline`                                                           | two `scatter3d`: the line and its current arrows                                                                                                |
| `TriangleSheet`, `TriangleStrip`                                               | a `mesh3d` for the sheet, plus a grey `mesh3d` for its current arrows                                                                            |
| `Dipole`                                                                       | one `mesh3d`, an arrow                                                                                                                           |
| `Sensor`                                                                       | one `mesh3d` with `facecolor`: axes, arrows and pixels. Pixels drawn as 2D symbols or `arrow`s come as a `scatter3d` instead                    |
| `CustomSource`                                                                 | one `scatter3d` marker                                                                                                                           |
| any object with a path                                                         | one more `scatter3d` for the path                                                                                                                |
| `show(markers=...)`                                                            | one `scatter3d`, with `object_id = None`                                                                                                         |
| a `generic` model in `style.model3d.data`                                      | the model's trace, positioned and oriented with its object                                                                                      |

By default Magpylib merges the traces of objects that share a legend group and a style into one trace: in practice, the children of a `Collection` drawn in the collection's colour. A merged trace has `object_id = None`. Declaring `merge_traces = False` keeps every object's own traces and `object_id`, which picking and drag gizmos need; Studio declares it. The children's traces still carry the collection's `name` and `legendgroup`. Merges _within_ one object, such as all the current arrows of a `Polyline`, always apply.

An object passed to `show` both on its own and inside a collection is drawn once, under its own entry. The collection's trace leaves it out, but its name still counts it.

(guide-graphics-custom-backend-capabilities)=

## Capabilities

A backend declares what it can draw. Every flag defaults to `False`, so a capability added in a later Magpylib release never changes an existing backend's behaviour. `backend.supports` lists them by short name.

| flag                        | when `False`                                                                                                                                                                                                                       |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `supports_animation`        | `show(animation=...)` warns and draws a static scene                                                                                                                                                                               |
| `supports_subplots`         | a grid warns and is collapsed onto one cell. A grid mixing 3D and 2D panels cannot be collapsed: it is handed over unchanged, with a warning                                                                                        |
| `supports_colorgradient`    | no warning, but the payload changes. Magnetization mode `"auto"` gives a flat-coloured mesh plus a `scatter3d` arrow. A mode with `"color"` slices each magnet into one band per colour, as `facecolor`, with `color = None` |
| `supports_animation_output` | `animation_output=...` warns, and `scene.animation.output` is `None`                                                                                                                                                               |
| `supports_native_traces`    | a user model attached for this backend warns and is not drawn; `Frame.native_traces` stays empty                                                                                                                                   |

Two more declarations shape the payload:

- **`merge_traces`** (default `True`) is a preference, not a capability: see above.
- **`handles_traces`** lists the trace types you draw, such as `{"mesh3d", "scatter3d"}`. Magpylib then warns about a type you never handle, instead of producing a quietly incomplete figure, which is what makes new trace types safe to add. `None`, the default, assumes you draw everything.

(guide-graphics-custom-backend-options)=

## Options

`scene.options` holds the `show()` arguments Magpylib does not interpret, such as Plotly's `renderer`. Their values arrive unchanged. Their names are normalised:

- A user may prefix an option with your backend's name: `counter_alpha=1` and `counter={"alpha": 1}` arrive as `alpha`. With several spellings, the prefixed one wins over the dict, and the dict over the bare name.
- `fig={"width": 400}` arrives as `fig_width`, and `show={...}` likewise. An explicit `fig_width=` wins over the dict.

Declare **`accepts_options`**, the names your backend takes: a name, a namespace such as `"fig_*"`, or `"*"` for anything. Magpylib then warns about any other name, naming the user's spelling, so that a typo does not pass unnoticed. The warned-about name still arrives in `scene.options`. `None`, the default, accepts anything without a word.

`fig_*` and `show_*` are two conventional namespaces. The built-in backends declare both and forward them to their library: `fig_*` to its figure (Matplotlib's `plt.figure`, Pyvista's `Plotter`, Plotly's `Figure.update`, which Plotly applies to a given canvas too) and `show_*` to its show call. `scene.prefixed_options("fig")` reads them with the prefix removed. The library's own error checks a forwarded value: it keeps its type and gains a note naming the `show()` argument. A backend with no such library leaves both undeclared, so that a user's `fig_width=400` is reported as the typo it is there.

Names Magpylib interprets itself never reach a backend, see [](guide-graphics-custom-backend-reserved). Declaring one warns at the first `show()`, and so does declaring a name with your own prefix: the prefix is stripped before you see an option, so declare the bare name.

(guide-graphics-custom-backend-native)=

## Native traces

A user can attach a model in your backend's own format, with `obj.style.model3d.add_trace(backend="<your name>", constructor=..., args=..., kwargs=...)`; see [](examples-own-3d-models). If you declare `supports_native_traces = True`, Magpylib positions and orients the model with its object and hands it over in `Frame.native_traces`, not in `Frame.traces`. You must then draw the list: Magpylib cannot tell whether you did. Without the flag, Magpylib warns and drops the model. A `generic` model is translated into ordinary traces instead, for every backend.

Each item is a dict:

| key            | type    | meaning                                                                                                                                    |
| -------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `constructor`  | `str`   | the user's constructor name, unchecked                                                                                                     |
| `args`         | list    | positional arguments, coordinates already transformed                                                                                      |
| `kwargs`       | dict    | keyword arguments, coordinates already transformed                                                                                         |
| `coordsargs`   | dict    | where the coordinates are: `{"x": "x", "y": "y", "z": "z"}` names keys of `kwargs`, `{"x": "args[0]", "y": "args[1]", "z": "args[2]"}` positions in `args` |
| `kwargs_extra` | dict    | `row`, `col`, `name`, `legendgroup`, `showlegend`, `color` and `opacity`, from the object                                                  |
| `object_id`    | `int`   | the object, as on traces                                                                                                                   |

`coordsargs` is the user's, or by default `args[0]` to `args[2]` when the model has positional arguments and its keywords do not name all of `x`, `y` and `z`, and the keywords otherwise. Placement, legend and colour are in `kwargs_extra`, not in `kwargs`: combine them as your constructor expects.

(guide-graphics-custom-backend-2d)=

## 2D charts

With `output="Bx"` rather than `"model3d"`, a `chart2d` panel holds `scatter` curves, with the path index as `x` and the field as `y`. The panel's `labels` title its axes; the outputs of one panel always share a unit. A pure-3D backend has no answer for these: leave `"scatter"` out of `handles_traces` and Magpylib warns.

A 2D legend lists the curves in titled groups: one group per output and panel, split per sensor when sources are not summed up (`sumup=False`). A curve's `legendgroup` is its group's key, stable from run to run; compare it, but do not parse it, as its format is not part of the contract. The group's title is `legendgrouptitle_text`, such as `"Bx (2 sources)"`, which all its curves carry. Each curve keeps its own entry, with its own `name` and `showlegend`.

In an animation every frame repeats each curve and only `marker_size` changes, enlarging the point at that frame's path index. A backend that draws a curve once can place its own cursor from `scene.animation.path_indices`.

(guide-graphics-custom-backend-animation)=

## Animation

When animating, `scene.frames` holds one frame per selected path step, and `scene.animation` says how to play them:

| `AnimationSettings` field | type             | meaning                                                                                                                                      |
| ------------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `frame_duration`          | `int` or `None`  | how long to show each frame, in milliseconds: the duration over the number of frames. Play at this rate. `None` for a static scene           |
| `path_indices`            | tuple of `int`   | the 0-based path index of each frame; empty for a static scene                                                                               |
| `fps`                     | `int`            | the frame rate the user asked for (`animation_fps`), not the rate to play at                                                                 |
| `time`                    | `float`          | the duration the user asked for, in seconds (`animation_time`)                                                                               |
| `slider`                  | `bool`           | whether to show a slider                                                                                                                     |
| `repeat`                  | `bool`           | whether to loop, `show(repeat=...)`                                                                                                          |
| `output`                  | `str` or `None`  | `"gif"`, `"mp4"`, or a file path ending in one of them; always `None` unless you declare `supports_animation_output`                        |

Magpylib picks the frames from the path length and the user's animation settings, then sets `frame_duration`. The two rates can differ a lot: twelve path steps over 5 seconds give `frame_duration = 416`, about 2.4 frames per second, while `fps` still says 20.

`Frame.label` is display text, the 1-based path index (`"1"`, `"2"`, ...), not an index into anything. A static scene has one frame, labelled `"0"`, which shows the last path step.

## Geometry and units

**Coordinates arrive in the display unit, already scaled**, and so does `Panel.ranges`. The unit itself appears only as text in `Panel.labels`, such as `"x (mm)"`, or `"x (µm)"` with the micro sign (U+00B5). There is no numeric factor.

**Under default settings all geometry depends on the whole scene.** Vertex arrays already have the object's position and orientation applied, and three things are derived from the rest of the scene, so re-rendering one object on its own and splicing it into an existing figure by `object_id` yields wrong geometry:

- **Autosize** scales sensors and dipoles, which have no physical size of their own, by the scene extent. It varies continuously, so moving any object rescales every autosized object.
- **`units_length="auto"`** picks an SI prefix from the largest absolute value in the ranges, zoom included, and rescales _every_ trace by the corresponding factor. It is quantized: nothing changes until that value crosses a threshold (0.1 m and 1000 m around metres), and then the entire scene jumps by a factor of 1000.
- **A sensor showing its pixel field** (`style.pixel.field.source` set) reads every source in the scene: its pixels change size and colour when any source moves, whatever is pinned.

A backend that re-renders whole scenes need not care. One that splices or updates incrementally does, and the first two can be pinned, per call or globally through `magpy.defaults.display`. Studio pins them for as long as it draws, and puts the user's defaults back afterwards:

```python
display = magpy.defaults.display
display.units.length = "m"  # or show(..., units_length="m")
display.style.sensor.sizemode = "absolute"  # sizes become lengths in metres
display.style.dipole.sizemode = "absolute"
display.style.sensor.size = display.style.dipole.size = 0.005  # metres, not 1
```

Pin both: either alone leaves the geometry scene-dependent.

(guide-graphics-custom-backend-details)=

## The objects, the canvas and the figure

- **`scene.return_fig` decides whether to _display_, not what to return.** Magpylib discards whatever `show` returns unless it is set, so the question it answers is whether to call your library's blocking show or display. Returning the figure either way is fine.
- **`scene.canvas` is a Python object**, such as a Matplotlib `Figure`, a Plotly `Figure` or a PyVista `Plotter`, and Magpylib does not check it for you. A backend that renders outside the Python process, in a browser for instance, has nothing to bind it to and should raise. For the same reason `backend="auto"`, which infers the backend from the canvas type, will never select one.
- **The objects are there when the traces are not enough.** `Panel.objects` holds what was passed to `show` for that cell, collections whole. It is how a backend recovers what no trace carries: the `Collection` hierarchy (`legendgroup` names only the _outermost_ collection, so three levels of nesting arrive looking like one), and the object behind an `object_id`, by matching `id(obj)`. Lists are unpacked, so an object passed on its own and inside a collection passed with it appears both ways. They are live objects: a backend that keeps them keeps them alive.
- **An object's style is not resolved.** A style value the user never set reads `None` on the object; Magpylib fills it from `magpy.defaults.display.style` (`.sensor`, `.dipole`, ...) when it draws, and a per-call `show(style_...)` override is not visible from the scene at all. The traces carry the resolved values.
- **Scenes are z-up.** Camera code needs the convention stated explicitly.

(guide-graphics-custom-backend-legend)=

### The legend

`Panel.labels` carries the axis titles and every trace carries `name`, `legendgroup` and `showlegend`, but nothing renders them for you.

In 3D, the traces sharing a `legendgroup` are one legend entry, which the trace with `showlegend=True` makes. That is how a `Collection`'s children collapse into one row: the group names only the outermost object. Its key, such as `"Cuboid#0"`, is the same for the same scene on every run, which `object_id` is not; compare it, but do not parse it. For 2D curves, see [](guide-graphics-custom-backend-2d).

`Scene.legend_maxitems` is the most entries a legend should list. The built-in backends list that many, then one "… and k more" row; `0` hides their 3D legends, while Matplotlib still draws its 2D legends. The cap is never applied to the traces themselves, so a backend whose legend scales another way, such as a collapsible tree, can ignore it.

## Helpers for sensor pixels

A backend drawing sensor pixels its own way, as instances in a scene graph for example, can size and colour them as the built-ins do:

- `sensor_pixel_fields(objects)` returns `{sensor: field}`, the field each sensor showing its pixel field reads. `field` has the shape `(path_length, n_pixels, 3)`, in the sensor's own frame, summed over the sources among `objects`, with the components its `style.pixel.field.source` does not name set to 0. Sensors with nothing to read are left out.
- `pixel_sizes(sensor, field=None, *, autosize=None, path_index=-1)` returns the size each pixel is drawn at: half the distance between the nearest two, or a fifth of the sensor where they coincide, times `style.pixel.size`, and scaled by `field`, from `sensor_pixel_fields`, as `style.pixel.field.sizescaling` and `sizemin` say. `autosize` is the size a sensor with `style.sizemode="scaled"` takes from the scene's extent, which only a lone pixel at the sensor's origin is sized against.
- `scale_pixel_norms(norms, scaling, *, minimum=None)` turns field norms into factors in [`minimum` or 0, 1], scaled over the whole array, so pass `(path_length, n_pixels)` norms to scale over the path as the built-ins do. `scaling` is a value of `style.pixel.field.sizescaling` or `colorscaling`: `"uniform"`, `"linear"`, `"log"`, `"log^2"` to `"log^9"`. `minimum`, as `style.pixel.field.sizemin`, lies in [0, 1]. NaN marks a pixel that reads nothing.
- `colors_from_colorscale(values, colorscale, *, cmin=None, cmax=None, nan_color="#b2beb5")` returns one colour per value of the 1-D `values`, as a list: `"#rrggbb"`, or `nan_color` for NaN. `colorscale` is a Plotly scale name such as `"Viridis"`, or `[(stop, colour), ...]` with any CSS name or hex colour, such as a `mesh3d` trace's own `colorscale`. `cmin` and `cmax` default to the range of `values`; for factors from `scale_pixel_norms`, pass `cmin=0, cmax=1`, as the built-ins do.

Studio draws such pixels as instances of one shape. For one sensor at path index `index`, `-1` for a static scene:

```python
import numpy as np

from magpylib.graphics.backend import (
    colors_from_colorscale,
    pixel_sizes,
    scale_pixel_norms,
    sensor_pixel_fields,
)

for sensor, field in sensor_pixel_fields(panel.objects).items():
    sizes = pixel_sizes(sensor, field, path_index=index)
    norms = np.linalg.norm(field, axis=-1)  # (path_length, n_pixels)
    colors = colors_from_colorscale(
        scale_pixel_norms(norms, "linear")[index], "Viridis", cmin=0, cmax=1
    )
```

Read `"linear"` and `"Viridis"` from the sensor's `style.pixel.field` (`colorscaling`, `colormap`), falling back to the defaults where unset; `pixel_sizes` reads its own. A pixel that reads nothing keeps its size, and its colour is `nan_color`. With `colorscaling="uniform"` the pixels keep `style.pixel.color`, black when unset. There is no opt-out yet: Magpylib still draws the pixels into the sensor's own `mesh3d`. Studio sets `style.pixel.size = 0` on such sensors while it captures the scene, and restores it afterwards.

## Warnings a backend can trigger

- **At the class statement:** a misspelled `supports_*` flag; taking a name another module's backend holds.
- **On discovery:** an entry point that fails to load or registers no backend; an entry-point key that differs from the class's `name`.
- **At the first `show()`:** declared options that Magpylib reserves or that carry your own prefix; a backend name equal to a `show()` argument.
- **At every `show()`:** an `api_version` other than the current one; options you do not declare; trace types outside `handles_traces`; a capability the call needs and you do not declare, except colour gradients.

(guide-graphics-custom-backend-reserved)=

## Reserved names

- **Backend names.** `"auto"`, `"fig"` and `"show"` raise `ValueError`. Avoid also the names of `show()`'s own arguments (`animation`, `backend`, `canvas`, `canvas_update`, `col`, `in_out`, `legend_maxitems`, `markers`, `output`, `pixel_agg`, `repeat`, `return_fig`, `row`, `style`, `sumup`, `title`, `units_length`, `zoom`) and of `magpy.defaults.display` (`animation`, `autosizefactor`, `backend`, `colorsequence`, `legend`, `style`, `units`): neither `<name>={...}` nor `<name>_<option>` could reach such a backend, and Magpylib warns at its first `show()`.
- **Option names.** The same names are Magpylib's own, and so is any name made of a `magpy.defaults.display` name, an underscore and more (`style_color`, `units_length`): they never reach a backend.
- **Class attributes.** A subclass may not redefine `backends`, `supports`, `discover`, `unaccepted_options` or `unhandled_trace_types`: `TypeError` at the class statement. Implement `show` and set the declarations; leave the rest alone.

## Versioning

Set `api_version = 1`, the version of this contract you wrote against (`API_VERSION` is the current one). Left unset, a backend counts as written against version 1 and is warned once the contract moves on. A backend declaring another version than the current one is warned at every `show()`.

The version goes up only when a backend written against the previous one could misread what it receives: a field, a trace key or a trace type removed, renamed or given another meaning. Additions never bump it: a new field with a default, a new trace key, a new trace type (`handles_traces` guards those), a new capability (`False` by default), a new metadata key (`drawing_properties` strips those). Write against that rule: read keys with `trace.get`, ignore keys you do not know, and declare `handles_traces`.

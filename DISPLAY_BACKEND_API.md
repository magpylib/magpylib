# Proposed public display-backend API

**Status:** design draft for review, nothing implemented. **Written:**
2026-08-04.

Supersedes the `register_backend(name, show_func, *, four_bools)` shape
currently in PR #970. Written after surveying pandas, xarray, matplotlib,
networkx and scipy (see §6).

---

## 1. What is wrong with the current shape

Measured against `main` at `46236212`:

```python
show_func(data, max_rows=, max_cols=, subplot_specs=, fig_kwargs=,
          show_kwargs=, canvas=, canvas_update=)
```

1. **Seven loose parameters**, two of which (`canvas`, `canvas_update`) arrive
   through `**kwargs` and are undocumented, though every real backend needs
   them.
2. **Four required bools**, soon five with `merge_traces`. Every capability
   added is a breaking change for every existing backend.
3. **Plotly's vocabulary is the contract.** `colorscale`, `showscale`,
   `showlegend`, `legendgroup`, `type: "mesh3d"`, and `subplot_specs` cells
   typed `{"type": "scene"|"xy"}` — that last is literally
   `plotly.subplots.make_subplots`. A three.js or mayavi author has to learn
   plotly to read magpylib's data.
4. **`input_kwargs` is a misnomer.** It holds six `animation_*` keys.
5. **No discovery.** A third party cannot make `pip install magpylib-three`
   sufficient; the user must call a function first.

The geometry itself is fine: `x/y/z` + `i/j/k` is just structure-of-arrays mesh
data. Only the surrounding vocabulary is plotly-shaped.

---

## 2. The backend class

Replaces function + flags, modelled on `xarray.backends.BackendEntrypoint`.

```python
from magpylib.display import DisplayBackend, Scene

class ThreeBackend(DisplayBackend):
    name = "three"
    description = "Interactive three.js renderer"
    url = "https://github.com/magpylib/magpylib-studio"

    # capabilities -- what the backend *can* do. Defaults are conservative,
    # so a new capability never breaks an existing backend.
    supports_animation = False
    supports_subplots = False
    supports_colorgradient = False
    supports_animation_output = False

    # preferences -- what the backend *wants*. Not capabilities: every
    # backend can render unmerged traces, plotly just prefers merged ones.
    merge_traces = True

    def show(self, scene: Scene):
        ...
```

Why a class:

- new capabilities become attributes with defaults instead of new required
  keyword arguments;
- `description`/`url` let tooling list installed backends meaningfully (xarray
  does this);
- capabilities and preferences are visibly separate, answering the
  `merge_traces` placement question structurally rather than by naming.

`register_backend` from #970 is kept as the imperative escape hatch and
implemented by wrapping a function into a generated subclass, so nothing in
#970's public surface breaks.

---

## 3. The payload

One `Scene` replaces the seven loose parameters.

```python
@dataclass(frozen=True)
class Scene:
    panels:    tuple[Panel, ...]      # the subplot grid, >= 1
    frames:    tuple[Frame, ...]      # animation timeline; length 1 if static
    canvas:    Any | None             # user-supplied figure/axes, or None
    canvas_update: bool               # may the backend restyle the canvas
    animation: AnimationSettings
    fig_kwargs:  Mapping[str, Any]    # forwarded verbatim, backend-specific
    show_kwargs: Mapping[str, Any]

    @property
    def n_rows(self) -> int: ...
    @property
    def n_cols(self) -> int: ...
    def panel(self, row: int, col: int) -> Panel: ...
```

`max_rows`/`max_cols`/`subplot_specs` collapse into `panels`, which carries the
grid _and_ each cell's kind and axis metadata:

```python
@dataclass(frozen=True)
class Panel:
    row: int
    col: int
    kind: Literal["scene3d", "chart2d"]   # was {"type": "scene"|"xy"}
    ranges: np.ndarray | None             # (3, 2) min/max per axis; None for chart2d
    labels: Mapping[str, str]             # {"x": "x (m)", ...}; empty for chart2d


@dataclass(frozen=True)
class Frame:
    label: str                            # was `name`, a stringified index
    title: str | None                     # was frame["layout"]["title"]
    traces: tuple[Trace, ...]             # was `data`
    native_traces: tuple[NativeTrace, ...]  # was `extra_backend_traces`


@dataclass(frozen=True)
class AnimationSettings:
    fps: int
    max_fps: int
    max_frames: int
    time: float
    slider: bool
    output: str | None                    # ".mp4" / ".gif" / None
```

`ranges` and `labels` move from `dict[(row, col)]` onto the `Panel` that owns
them. `input_kwargs` disappears into `AnimationSettings`.

---

## 4. Traces

Four primitives plus a passthrough, replacing
`type: "mesh3d"|"scatter3d"|"scatter"`.

```python
@dataclass(frozen=True)
class Trace:                    # shared by all -- and only these are universal
    row: int
    col: int
    label: str | None           # was `name`
    group: str | None           # was `legendgroup`
    show_in_legend: bool        # was `showlegend`
    object_id: int | None       # Step 3; None on cross-object merges


@dataclass(frozen=True)
class Mesh(Trace):                       # was type="mesh3d"
    vertices: np.ndarray                 # (n, 3) float   -- was x, y, z
    faces: np.ndarray                    # (m, 3) int     -- was i, j, k
    color: str | None                    # uniform color
    vertex_values: np.ndarray | None     # (n,)           -- was `intensity`
    face_colors: np.ndarray | None       # (m,)           -- was `facecolor`
    colormap: tuple[tuple[float, str], ...] | None   # was `colorscale`
    show_colorbar: bool                  # was `showscale`
    opacity: float


@dataclass(frozen=True)
class Scatter3D(Trace):         # was type="scatter3d"
    points: np.ndarray          # (n, 3)
    draw: frozenset[str]        # subset of {"lines", "markers", "text"}
    line: LineStyle | None
    marker: MarkerStyle | None
    texts: tuple[str, ...] | None
    opacity: float


@dataclass(frozen=True)
class Scatter2D(Trace):         # was type="scatter": field plots, output="Bx" etc.
    x: np.ndarray
    y: np.ndarray
    draw: frozenset[str]
    line: LineStyle | None
    marker: MarkerStyle | None
    texts: tuple[str, ...] | None
    hover_template: str | None       # was `hovertemplate`
    group_title: str | None          # was `legendgrouptitle_text`
    # note: no `opacity` -- 2D traces never carry one


@dataclass(frozen=True)
class LineStyle:
    color: str | None
    width: float | None
    dash: str | None


@dataclass(frozen=True)
class MarkerStyle:
    color: str | None
    size: float | None
    symbol: str | None
    line_color: str | None      # was `marker_line_color`; 2D only


@dataclass(frozen=True)
class NativeTrace:
    """A model the user supplied in this backend's own format.

    Already positioned and oriented. No `backend` field: the list only ever
    contains traces addressed to the backend currently rendering.
    """
    constructor: str
    args: tuple
    kwargs: Mapping[str, Any]
    coordsargs: Mapping[str, str]    # which kwargs hold the coordinates
    extra: Mapping[str, Any]         # was `kwargs_extra`: opacity, color, group
```

**Corrected 2026-08-04 after measuring the real payload.** An earlier draft
split `scatter3d` into separate `Polyline`, `Points` and `Labels` classes. That
is wrong: `mode` is a _combination_, and magpylib emits `"markers+text+lines"`
for a path with numbering enabled. Splitting would turn one trace into three,
changing trace counts and breaking the 1:1 mapping a backend needs to reason
about identity. `draw` is therefore a set on a single class, mirroring the
existing `mode`, with line/marker styling grouped rather than flattened into
`line_color` / `marker_size` / ....

Renames are mechanical and each removes a plotly-ism:

| now                           | proposed                         | why                                                        |
| ----------------------------- | -------------------------------- | ---------------------------------------------------------- |
| `x`, `y`, `z` / `i`, `j`, `k` | `vertices` (n,3) / `faces` (m,3) | the spelling pyvista, trimesh, open3d and three.js all use |
| `intensity`                   | `vertex_values`                  | says what it is, not what plotly calls it                  |
| `colorscale`                  | `colormap`                       | neutral between matplotlib `cmap` and plotly `colorscale`  |
| `showscale`                   | `show_colorbar`                  | says what it draws                                         |
| `legendgroup`                 | `group`                          | it is doing identity _and_ legend grouping                 |
| `name`                        | `label`                          | `name` collides with the backend's own `name`              |
| `type: "mesh3d"`              | the class                        | dispatch on type, not on a string                          |

**Resolved (measured 2026-08-04): use `vertices`/`faces`, no mitigation
needed.** The concern was that converting the internal `x/y/z/i/j/k` costs an
`np.stack` per trace. Measured on `main` at `f57b020f`:

| case                           | traces | vertices | build   | convert | overhead  |
| ------------------------------ | ------ | -------- | ------- | ------- | --------- |
| 8000 separate cuboids          | 8000   | 64000    | 1.600 s | 0.019 s | **1.2 %** |
| 8000 cuboids in one Collection | 1      | 64000    | 0.943 s | 0.001 s | **0.1 %** |

Worst case is ~2.4 µs per trace, against a build that already costs ~200 µs per
trace. The cached-property idea is dropped: it would trade a measurable
simplification for an unmeasurable saving.

Note also that 8000 _separate_ objects do **not** merge — `group_traces` keys on
`legendgroup` and `color`, which differ per object. Merging only kicks in when
objects share both, e.g. inside a Collection. That is worth knowing before
designing anything around the merge.

---

## 5. Discovery

The standard mechanism, and the thing #970 lacks entirely.

```toml
# in magpylib-three's pyproject.toml
[project.entry-points."magpylib.backends"]
three = "magpylib_three:ThreeBackend"
```

`pip install magpylib-three` then makes `backend="three"` work with no user
code. Resolution order, mirroring pandas:

1. already registered in-process (via `register_backend` or a previous lookup);
2. an entry point in the `magpylib.backends` group, loaded lazily on first use;
3. otherwise `MagpylibBadUserInput`, listing what is available.

Lazily is important: entry points must not be loaded at import time, or
magpylib's import cost becomes hostage to every installed backend.

---

## 6. Prior art

| library          | discovery                                                   | contract                                       |
| ---------------- | ----------------------------------------------------------- | ---------------------------------------------- |
| pandas plotting  | `pandas_plotting_backends` entry point, else import by name | module exposing `plot()`                       |
| xarray           | `xarray.backends` entry point                               | `BackendEntrypoint` subclass                   |
| matplotlib       | `matplotlib.backend` entry point (3.9+), `use()`            | `FigureCanvasBase` ABC                         |
| networkx         | `networkx.backends` entry point                             | dispatchable functions; shipped _experimental_ |
| scipy fft/linalg | `uarray`                                                    | `set_backend()` + protocol                     |

Two conventions are near-universal: **entry points for discovery**, and **a
class or protocol rather than a function plus flags**. This proposal follows
both.

---

## 7. Migration

1. Land #970 as-is, with `register_backend` **marked provisional** so this can
   change it without a breaking release.
2. Introduce `DisplayBackend`, `Scene` and the trace classes; port the three
   built-ins; keep `register_backend` as a wrapper.
3. Add entry-point discovery.
4. Add `merge_traces` + `object_id` (Step 3) — which becomes an attribute and a
   field rather than a fifth bool and a dict key.

Steps 2–4 are separate PRs. Step 1 is what makes the rest non-breaking, and is
worth doing even if the rest is deferred.

---

## 7b. Bugs found while validating this design (2026-08-04)

Both pre-existing on `main` at `f57b020f`, both independent of this redesign.
Neither should be fixed inside it -- a behaviour change hidden in a large
refactor is how you get an ambiguous bisect later.

1. **`show(title=...)` is silently discarded.** `traces_generic.py:1024` does
   `title = label if len(objs[0]["objects"]) == 1 else None` under a comment
   reading "infer title if necessary" -- but with no `if title is None` guard,
   so it unconditionally overwrites whatever the caller passed. One object in
   the first panel substitutes that object's label; more than one yields `None`.

2. **Cross-object merging is rarer than assumed.** `group_traces` keys on
   `color`, which the colorsequence varies per object, so 8000 separate cuboids
   produce 8000 traces, not one. Merging only applies when objects share both
   legendgroup and color, e.g. inside a Collection. The "single draw call"
   argument for preserving the merge in Step 3 is therefore much weaker than the
   plan states -- the common case never merged in the first place.

3. **2D panels carry 3D axis metadata.** A `chart2d` panel is emitted with a
   `(3, 2)` range and `{"x": "x (m)", "y": "y (m)", "z": "z (m)"}` labels.
   Harmless today because no backend reads them for 2D, which is why the adapter
   drops them (§3).

---

## 8. Open questions

1. **`vertices`/`faces` vs keeping `x/y/z/i/j/k`.** Neutral spelling and a copy,
   or plotly spelling and zero cost? Measure first.
2. **Is `Scene` frozen?** Backends currently mutate frames in place — plotly's
   `fr.pop("extra_backend_traces", None)`. Freezing is cleaner but forces those
   backends to build their own structures.
3. **Does `Curve2D` belong here at all,** or should 2D field plots be a separate
   hand-off? A pure-3D backend has no answer for them either way.
4. **How much of this is public?** `Scene` and the trace classes have to be
   importable for type annotations, so `magpylib.display` becomes a public
   module with its own stability commitment.

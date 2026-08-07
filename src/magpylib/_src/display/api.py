"""Public contract between `magpylib.show` and a display backend.

The split here is deliberate, and follows what glTF, Vega and matplotlib all
do: a **typed envelope** around **open payload**.

*Typed* -- `Scene`, `Panel`, `Frame`, `AnimationSettings` -- because the
envelope is small, fixed, controlled entirely by magpylib and changes rarely.
A backend gets one object describing the whole figure rather than a handful of
loose arguments.

*Open* -- traces stay plain dicts in magpylib's documented dialect (plotly-style
magic-underscore keys, matplotlib-derived symbol and dash vocabularies). They
are large, open-ended, and must be able to grow a key without a release; a
frozen dataclass cannot. `magic_to_dict` turns ``marker_line_color`` into
nested form for backends that prefer it.

The dialect looks plotly-flavoured but is magpylib's own: every backend,
plotly included, translates it (`SYMBOLS_TO_PLOTLY`, `SYMBOLS_TO_PYVISTA`).
Renaming it would buy a permanent adapter and three ported backends in
exchange for nothing a backend author needs; what such an author needs is
specification, which the user guide provides.

Four things not to do here:

- **Do not rewrite the trace format into an abstract IR.** The hand-off is
  already a plain dict of numpy arrays that matplotlib, plotly and pyvista all
  consume, so an IR would be a translation layer with no consumer.
- **Do not remove `group_traces` or the cross-object merge.** It is a real
  optimisation for scenes where objects share a legendgroup and colour. It is
  optional per backend via `DisplayBackend.merge_traces`; do not delete it.
  The four merges *within* one object are unconditional by design.
- **Do not change what `legendgroup` means.** Collection-scoped legend grouping
  is correct behaviour. Identity is a separate axis and lives in `object_id`.
- **Do not "fix" the capability-driven geometry.** `supports_colorgradient=False`
  triggers geometric mesh slicing per colour band in `update_magnet_mesh`,
  because matplotlib cannot interpolate vertex colours. It is a seam, it is
  ugly, and it works.
"""

import warnings
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, ClassVar, Literal

import numpy as np

#: Trace keys carrying magpylib metadata rather than drawing properties.
#: A backend must not pass these to its plotting library -- plotly rejects
#: unknown properties outright, and magic-underscore would read ``object_id``
#: as ``object.id``. Use `drawing_properties` to strip them.
TRACE_META_KEYS = frozenset({"object_id"})

#: Entry-point group third-party packages advertise backends under, so that
#: ``pip install`` is enough. Same mechanism as xarray.backends,
#: matplotlib.backend and pandas_plotting_backends.
ENTRY_POINT_GROUP = "magpylib.backends"

#: Version of the payload contract. Bumped when the *envelope* changes shape
#: in a way a backend written against an older magpylib could not handle.
API_VERSION = 1


def drawing_properties(trace):
    """Return `trace` without magpylib's metadata keys.

    Traces mix drawing properties with magpylib metadata such as
    ``object_id``. Plotting libraries reject the latter, so strip them before
    handing a trace over::

        fig.add_trace(go.Mesh3d(**drawing_properties(trace)))
    """
    return {k: v for k, v in trace.items() if k not in TRACE_META_KEYS}


@dataclass(frozen=True)
class Panel:
    """One cell of the subplot grid, owning its own axis metadata."""

    row: int = 1
    col: int = 1
    kind: Literal["scene3d", "chart2d"] = "scene3d"
    #: (3, 2) of (min, max) per axis. None for chart2d -- magpylib emits a 3D
    #: range there too, but it describes nothing a 2D chart can use.
    ranges: np.ndarray | None = None
    #: e.g. {"x": "x (m)", ...}. Empty for chart2d, for the same reason.
    labels: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Frame:
    """One step of the timeline. A static scene has exactly one."""

    label: str = ""
    #: Trace dicts in magpylib's dialect. Left as dicts on purpose -- see the
    #: module docstring.
    traces: tuple[dict[str, Any], ...] = ()
    #: Models the user supplied in *this* backend's own format, already
    #: positioned and oriented. Keys: constructor, args, kwargs, coordsargs,
    #: kwargs_extra. **A backend that ignores these silently drops the user's
    #: models** -- no warning, no error.
    native_traces: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class AnimationSettings:
    """Timing and playback parameters for an animated scene."""

    fps: int = 20
    max_fps: int = 30
    max_frames: int = 200
    time: float = 5
    slider: bool = True
    output: str | None = None
    repeat: bool = False
    #: Milliseconds per frame, derived from fps and downsampling. None when
    #: the scene is static.
    frame_duration: float | None = None
    #: Which path steps the frames correspond to; may be downsampled.
    path_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class Scene:
    """Everything a backend needs in order to draw."""

    panels: tuple[Panel, ...] = ()
    frames: tuple[Frame, ...] = ()
    #: Figure title, or None. A property of the figure, not of a frame.
    title: str | None = None
    animation: AnimationSettings = field(default_factory=AnimationSettings)
    #: User-supplied figure/axes/plotter to draw into, or None.
    canvas: Any = None
    #: Whether the backend may restyle a user-supplied canvas.
    canvas_update: bool = True
    #: Return the figure instead of displaying it.
    return_fig: bool = False
    #: Hide the legend once it would carry more entries than this.
    legend_maxitems: int = 20
    fig_kwargs: dict[str, Any] = field(default_factory=dict)
    show_kwargs: dict[str, Any] = field(default_factory=dict)
    #: Options magpylib does not interpret, forwarded verbatim. This is where
    #: backend-specific arguments live (plotly's `renderer`, pyvista's
    #: `jupyter_backend`, matplotlib's `antialiased`, ...).
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def is_animation(self) -> bool:
        """True when there is more than one frame to draw."""
        return len(self.frames) > 1

    @property
    def n_rows(self) -> int:
        """Number of subplot rows."""
        return max((p.row for p in self.panels), default=1)

    @property
    def n_cols(self) -> int:
        """Number of subplot columns."""
        return max((p.col for p in self.panels), default=1)

    @property
    def has_subplots(self) -> bool:
        """True when the grid is larger than a single cell."""
        return self.n_rows > 1 or self.n_cols > 1

    def panel(self, row: int, col: int) -> Panel | None:
        """Return the panel at 1-based `row`/`col`, or None if the cell is empty."""
        for p in self.panels:
            if p.row == row and p.col == col:
                return p
        return None

    def panel_kind(self, row: int, col: int) -> str:
        """Kind of the panel at 1-based `row`/`col`.

        An empty cell reports ``"scene3d"``: a backend building a grid needs a
        kind for every cell, and a 3D axis is the harmless default.
        """
        panel = self.panel(row, col)
        return "scene3d" if panel is None else panel.kind


class DisplayBackend:
    """Base class for a magpylib display backend.

    Subclass, set `name`, implement `show`. Capability flags declare what the
    backend can do; `show` warns and falls back rather than handing over
    something the backend cannot draw.

    Capabilities default to False so that a capability added in a later
    magpylib release never silently changes an existing backend's behaviour.
    """

    #: Every registered backend, keyed by name. Subclasses with a `name`
    #: register themselves here on definition.
    backends: ClassVar[dict[str, "DisplayBackend"]] = {}

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    url: ClassVar[str] = ""

    #: Payload contract this backend was written against. magpylib warns when
    #: it no longer matches `API_VERSION`.
    api_version: ClassVar[int] = API_VERSION

    # capabilities: what the backend *can* do
    supports_animation: ClassVar[bool] = False
    supports_subplots: ClassVar[bool] = False
    supports_colorgradient: ClassVar[bool] = False
    supports_animation_output: ClassVar[bool] = False
    #: Whether models attached via ``style.model3d.data`` naming this backend
    #: are rendered. False -- like every capability -- so a backend that never
    #: reads `Frame.native_traces` gets a warning rather than silently dropping
    #: the user's models.
    supports_native_traces: ClassVar[bool] = False

    # preference, not a capability: every backend *can* render unmerged
    # traces, some simply prefer fewer and larger ones.
    merge_traces: ClassVar[bool] = True

    #: Names this backend accepts in `Scene.options`, the bag of arguments
    #: magpylib itself does not interpret. ``None`` means "accept anything",
    #: which is also what makes a typo like ``retrun_fig=True`` pass silently.
    #: Declaring the set lets magpylib say so. Modelled on xarray's
    #: `BackendEntrypoint.open_dataset_parameters`.
    accepts_options: ClassVar[frozenset[str] | None] = None

    #: Trace ``type`` values this backend knows how to draw. ``None`` means
    #: "assume everything". Declaring the set lets magpylib warn when a scene
    #: contains a type the backend never handles, instead of silently
    #: producing an incomplete figure -- which is what makes adding a new
    #: trace type safe. Modelled on HoloViews' element/plot registry.
    handles_traces: ClassVar[frozenset[str] | None] = None

    _discovered: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs):
        """Register any subclass that names itself."""
        super().__init_subclass__(**kwargs)
        if cls.name:
            cls.backends[cls.name] = cls()

    @classmethod
    def discover(cls):
        """Load backends advertised by installed packages, once.

        Resolved on the first backend-name lookup and cached, rather than at
        every lookup. In practice that first lookup happens while magpylib is
        imported, when the defaults tree validates its own default backend, so
        the scan does contribute to import time (a few ms, growing with the
        number of installed distributions).
        """
        if cls._discovered:
            return
        # set on the base, not on cls: discovering through a subclass must
        # still mark it done globally, or the base would rediscover later
        DisplayBackend._discovered = True
        for entry in entry_points(group=ENTRY_POINT_GROUP):
            try:
                loaded = entry.load()
            except Exception as err:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                warnings.warn(
                    f"Could not load display backend {entry.name!r} advertised "
                    f"by {entry.value!r}: {type(err).__name__}: {err}",
                    stacklevel=2,
                )
                continue
            # importing the class registers it via __init_subclass__; only
            # instantiate when it did not, e.g. the entry point names it
            # differently. setdefault would construct it unconditionally.
            if isinstance(loaded, type) and issubclass(loaded, DisplayBackend):
                key = loaded.name or entry.name
                if key not in cls.backends:
                    cls.backends[key] = loaded()

    @property
    def supports(self) -> dict[str, bool]:
        """Capability flags keyed by short name, derived from the attributes.

        Introspected rather than hand-listed so a new ``supports_*`` attribute
        needs no second edit here.
        """
        return {
            name[len("supports_") :]: getattr(self, name)
            for name in dir(type(self))
            if name.startswith("supports_")
        }

    def show(self, scene: Scene):
        """Draw `scene`, returning this backend's figure object."""
        raise NotImplementedError

    def unaccepted_options(self, scene: Scene) -> frozenset[str]:
        """Option names in `scene` this backend does not declare."""
        if self.accepts_options is None:
            return frozenset()
        return frozenset(set(scene.options) - self.accepts_options)

    def unhandled_trace_types(self, scene: Scene) -> frozenset[str]:
        """Trace types present in `scene` that this backend does not declare."""
        if self.handles_traces is None:
            return frozenset()
        present = {
            tr.get("type") for fr in scene.frames for tr in fr.traces if tr.get("type")
        }
        return frozenset(present - self.handles_traces)

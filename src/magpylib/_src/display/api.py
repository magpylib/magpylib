"""Public contract between `magpylib.show` and a display backend.

The split here is deliberate, and follows what glTF, Vega and matplotlib all
do: a **typed envelope** around **open payload**.

*Typed* -- `Scene`, `Panel`, `Frame`, `AnimationSettings` -- because the
envelope is small, fixed, controlled entirely by magpylib and changes rarely.
It was also the genuinely broken part: seven loose parameters, axis ranges and
labels in side-tables keyed by ``(row, col)``, `canvas` arriving undocumented
through ``**kwargs``.

*Open* -- traces stay plain dicts in magpylib's documented dialect (plotly-style
magic-underscore keys, matplotlib-derived symbol and dash vocabularies). They
are large, open-ended, and must be able to grow a key without a release; a
frozen dataclass cannot. `magic_to_dict` turns ``marker_line_color`` into
nested form for backends that prefer it.

Renaming that dialect was considered and rejected: every backend already
translates it (`SYMBOLS_TO_PLOTLY`, `SYMBOLS_TO_PYVISTA`), so it is magpylib's
vocabulary rather than plotly's, and renaming would have cost a permanent
adapter and three ported backends while fixing nothing an author complains
about. What was missing is specification, not different names.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np

#: Version of the payload contract. Bumped when the *envelope* changes shape
#: in a way a backend written against an older magpylib could not handle.
API_VERSION = 1


@dataclass(frozen=True)
class Panel:
    """One cell of the subplot grid.

    Replaces the ``ranges``/``labels`` side-tables that were keyed by
    ``(row, col)``: each panel now owns its own axis metadata.
    """

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
    #: Was buried in ``frame["layout"]["title"]``; varies across an animation.
    title: str | None = None
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
    """Animation parameters, replacing the misnamed ``input_kwargs`` bag."""

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
    animation: AnimationSettings = field(default_factory=AnimationSettings)
    #: User-supplied figure/axes/plotter to draw into, or None.
    canvas: Any = None
    #: Whether the backend may restyle a user-supplied canvas.
    canvas_update: bool = True
    #: Return the figure instead of displaying it.
    return_fig: bool = False
    #: Collapse the legend beyond this many entries.
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
        """Return the panel at `row`/`col`, or None if the cell is empty."""
        for p in self.panels:
            if p.row == row and p.col == col:
                return p
        return None


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
    #: are rendered. False means they are skipped with a warning rather than
    #: silently dropped or crashing inside the placement code.
    supports_native_traces: ClassVar[bool] = True

    # preference, not a capability: every backend *can* render unmerged
    # traces, some simply prefer fewer and larger ones.
    merge_traces: ClassVar[bool] = True

    #: Trace ``type`` values this backend knows how to draw. ``None`` means
    #: "assume everything". Declaring the set lets magpylib warn when a scene
    #: contains a type the backend never handles, instead of silently
    #: producing an incomplete figure -- which is what makes adding a new
    #: trace type safe. Modelled on HoloViews' element/plot registry.
    handles_traces: ClassVar[frozenset[str] | None] = None

    def __init_subclass__(cls, **kwargs):
        """Register any subclass that names itself."""
        super().__init_subclass__(**kwargs)
        if cls.name:
            cls.backends[cls.name] = cls()

    @property
    def supports(self) -> dict[str, bool]:
        """Capability flags keyed by short name, for uniform lookup."""
        return {
            "animation": self.supports_animation,
            "subplots": self.supports_subplots,
            "colorgradient": self.supports_colorgradient,
            "animation_output": self.supports_animation_output,
            "native_traces": self.supports_native_traces,
        }

    def show(self, scene: Scene):
        """Draw `scene`, returning this backend's figure object."""
        raise NotImplementedError

    def unhandled_trace_types(self, scene: Scene) -> frozenset[str]:
        """Trace types present in `scene` that this backend does not declare."""
        if self.handles_traces is None:
            return frozenset()
        present = {
            tr.get("type") for fr in scene.frames for tr in fr.traces if tr.get("type")
        }
        return frozenset(present - self.handles_traces)

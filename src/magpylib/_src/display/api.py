"""Public data model handed to display backends.

Backend-neutral description of what `show` wants drawn. The vocabulary here
is deliberately not plotly's: `vertices`/`faces` rather than `x/y/z` +
`i/j/k`, `colormap` rather than `colorscale`, `group` rather than
`legendgroup`, and typed trace classes rather than a `type` string.

Every field was derived from the payload magpylib actually produces, not
from what a plotting library happens to call things -- see
`DISPLAY_BACKEND_API.md`. Notably, only the fields on `Trace` are universal;
`opacity` for instance is absent from 2D traces and therefore lives on the
3D classes.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import numpy as np

# --------------------------------------------------------------------------
# style
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LineStyle:
    """How a line is stroked."""

    color: str | None = None
    width: float | None = None
    dash: str | None = None


@dataclass(frozen=True)
class MarkerStyle:
    """How a point marker is drawn.

    `size` may be a single value or one value per point -- a multi-pixel
    sensor's field curve carries one per pixel. Do not compare `MarkerStyle`
    instances with `==`; an array-valued `size` makes that ambiguous.
    """

    color: str | None = None
    size: float | np.ndarray | None = None
    symbol: str | None = None
    line_color: str | None = None  # outline; 2D traces only


# --------------------------------------------------------------------------
# traces
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Trace:
    """Fields shared by every trace -- and only these are universal."""

    row: int = 1
    col: int = 1
    label: str | None = None
    group: str | None = None
    show_in_legend: bool = True
    object_id: int | None = None


@dataclass(frozen=True)
class Mesh(Trace):
    """A triangular surface mesh.

    `vertices` is (n, 3) and `faces` is (m, 3) of indices into it -- the
    spelling pyvista, trimesh and three.js use. A backend wanting the
    plotly/structure-of-arrays form takes `vertices.T` and `faces.T`.
    """

    vertices: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    faces: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=int))
    color: str | None = None
    opacity: float = 1.0
    vertex_values: np.ndarray | None = None  # per-vertex scalars for `colormap`
    face_colors: np.ndarray | None = None
    colormap: tuple[tuple[float, str], ...] | None = None
    show_colorbar: bool = False


@dataclass(frozen=True)
class Scatter3D(Trace):
    """Points in space, drawn as any combination of lines, markers and text.

    `draw` is a set rather than a choice of class: magpylib emits a single
    trace drawing all three at once for a numbered path.
    """

    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    draw: frozenset[str] = frozenset({"lines"})
    line: LineStyle | None = None
    marker: MarkerStyle | None = None
    texts: tuple[str, ...] | None = None
    opacity: float = 1.0


@dataclass(frozen=True)
class Scatter2D(Trace):
    """A curve in a 2D panel, e.g. a field-vs-path-index plot.

    Carries no `opacity`: magpylib never emits one for 2D traces.
    """

    x: np.ndarray = field(default_factory=lambda: np.empty(0))
    y: np.ndarray = field(default_factory=lambda: np.empty(0))
    draw: frozenset[str] = frozenset({"lines"})
    line: LineStyle | None = None
    marker: MarkerStyle | None = None
    texts: tuple[str, ...] | None = None
    hover_template: str | None = None
    group_title: str | None = None


@dataclass(frozen=True)
class NativeTrace:
    """A model the user supplied in the rendering backend's own format.

    Already positioned and oriented. There is no `backend` field: this list
    only ever holds traces addressed to the backend currently rendering.

    A backend that ignores these **silently drops the user's models**.
    """

    constructor: str = ""
    args: tuple = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    coordsargs: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# scene
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Panel:
    """One cell of the subplot grid."""

    row: int = 1
    col: int = 1
    kind: Literal["scene3d", "chart2d"] = "scene3d"
    # 3D axis extents, (3, 2) of (min, max). None for chart2d: magpylib emits
    # a 3D range there too, but it describes nothing a 2D chart can use.
    ranges: np.ndarray | None = None
    labels: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AnimationSettings:
    """Animation parameters, previously the misnamed `input_kwargs`."""

    fps: int = 20
    max_fps: int = 30
    max_frames: int = 200
    time: float = 5
    slider: bool = True
    output: str | None = None


@dataclass(frozen=True)
class Frame:
    """One step of the timeline. A static scene has exactly one."""

    label: str = ""
    title: str | None = None
    traces: tuple[Trace, ...] = ()
    native_traces: tuple[NativeTrace, ...] = ()


@dataclass(frozen=True)
class Scene:
    """Everything a backend needs in order to draw."""

    panels: tuple[Panel, ...] = ()
    frames: tuple[Frame, ...] = ()
    canvas: Any = None
    canvas_update: bool = True
    animation: AnimationSettings = field(default_factory=AnimationSettings)
    fig_kwargs: dict[str, Any] = field(default_factory=dict)
    show_kwargs: dict[str, Any] = field(default_factory=dict)

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


# --------------------------------------------------------------------------
# backend
# --------------------------------------------------------------------------


class DisplayBackend:
    """Base class for a magpylib display backend.

    Subclass, set `name`, and implement `show`. Capability flags declare what
    the backend can do; `show` warns and falls back rather than handing over
    something it cannot draw.

    Capabilities default to False so that a capability added in a later
    magpylib release never silently changes an existing backend's behaviour.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    url: ClassVar[str] = ""

    # capabilities: what the backend *can* do
    supports_animation: ClassVar[bool] = False
    supports_subplots: ClassVar[bool] = False
    supports_colorgradient: ClassVar[bool] = False
    supports_animation_output: ClassVar[bool] = False

    # preferences: what the backend *wants*. Not a capability -- every backend
    # can render unmerged traces; some just prefer fewer, larger ones.
    merge_traces: ClassVar[bool] = True

    def show(self, scene: Scene):
        """Draw `scene` and return whatever this backend's figure object is."""
        raise NotImplementedError

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
  optimization for scenes where objects share a legendgroup and color. It is
  optional per backend via `DisplayBackend.merge_traces`; do not delete it.
  The four merges *within* one object are unconditional by design.
- **Do not change what `legendgroup` means.** Collection-scoped legend grouping
  is correct behavior. Identity is a separate axis and lives in `object_id`.
- **Do not "fix" the capability-driven geometry.** `supports_colorgradient=False`
  triggers geometric mesh slicing per color band in `update_magnet_mesh`,
  because matplotlib cannot interpolate vertex colors. It is a seam, it is
  ugly, and it works.
"""

import numbers
import sys
import warnings
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, ClassVar, Literal

import numpy as np

from magpylib._src.exceptions import MagpylibBadUserInput

#: Trace keys carrying magpylib metadata rather than drawing properties.
#: A backend must not pass these to its plotting library -- plotly rejects
#: unknown properties outright, and magic-underscore would read ``object_id``
#: as ``object.id``. Use `drawing_properties` to strip them, never a list of
#: your own: the set may grow in a later release.
TRACE_META_KEYS = frozenset({"object_id"})

#: Trace keys placing a trace in the subplot grid. No plotting library takes
#: them as trace arguments either, so `drawing_properties` strips them too:
#: read them first.
PLACEMENT_KEYS = frozenset({"row", "col"})

#: Entry-point group third-party packages advertise backends under, so that
#: ``pip install`` is enough. Same mechanism as xarray.backends,
#: matplotlib.backend and pandas_plotting_backends.
ENTRY_POINT_GROUP = "magpylib.backends"

#: Version of the payload contract. Bumped when a backend written against the
#: previous version could misread what it is handed: a field of `Scene`,
#: `Panel`, `Frame` or `AnimationSettings`, a trace key or a trace type
#: removed, renamed or given another meaning. Not bumped for an addition: a
#: new field with a default, a new trace key or trace type (`handles_traces`
#: guards those), a new capability (False by default), a new metadata key
#: (`drawing_properties` strips those). `tests/test_display_contract.py` pins
#: it: a change to its golden file is a decision about this number.
API_VERSION = 1


def forward_options(call, prefix, options, /, defaults=None, **fixed):
    """Call `call` with `options`, the user's ``<prefix>_*`` arguments, over
    `defaults`, which they may override, and `fixed`, which they may not.

    The built-in backends forward ``fig_*`` to their figure (Matplotlib's
    `plt.figure`, Pyvista's `Plotter`, Plotly's `Figure.update`) and
    ``show_*`` to their show call. magpylib cannot know what those accept,
    so the library's own error is the check. It keeps its type -- plotly a
    KeyError for an unknown key, a ValueError for a bad value, pyvista an
    IndexError for a malformed window size -- and gains a note naming the
    `show()` arguments that were passed, since the library names only its
    own keyword. Nothing here claims they caused it.
    """
    clash = sorted(set(options) & set(fixed))
    if clash:
        what = ", ".join(f"{k}={fixed[k]!r}" for k in clash)
        msg = (
            f"show() got {[f'{prefix}_{k}' for k in clash]}, but this backend "
            f"sets {what} itself."
        )
        raise MagpylibBadUserInput(msg)
    try:
        return call(**{**(defaults or {}), **options, **fixed})
    except Exception as err:
        if options:
            names = sorted(f"{prefix}_{k}" for k in options)
            err.add_note(
                f"show() passed {getattr(call, '__qualname__', call)} the "
                f"argument(s) {names}; check them first."
            )
        raise


def warn_unused_options(names, reason):
    """Say that the `show()` arguments `names` went nowhere, and why."""
    if names:
        warnings.warn(
            f"show() got {sorted(names)}, but {reason}, so they are ignored.",
            stacklevel=user_stacklevel(),
        )


def user_stacklevel():
    """The ``stacklevel`` that points a warning raised by the caller at the
    nearest frame outside magpylib, which is the user's `show()` call."""
    frame = sys._getframe(1)
    level = 1
    while frame is not None and _in_magpylib(frame.f_globals.get("__name__", "")):
        frame = frame.f_back
        level += 1
    return level


def _in_magpylib(module):
    return module == "magpylib" or module.startswith("magpylib.")


def _iterable(value):
    try:
        iter(value)
    except TypeError:
        return False
    return True


def _option_matches(name, patterns):
    """Whether `name` is in `patterns`, where a trailing ``*`` matches a prefix."""
    return any(
        name == p or (p.endswith("*") and name.startswith(p[:-1])) for p in patterns
    )


def _check_option_pattern(owner, attr, pattern):
    """A declared option is a name, a ``<name>_*`` namespace, or ``*`` for
    anything. Checked at declaration so the rule can only be relaxed later."""
    ok = (
        pattern
        and ("*" not in pattern or pattern == "*" or pattern.endswith("_*"))
        and pattern.count("*") <= 1
    )
    if not ok:
        msg = (
            f"{owner}.{attr} entry {pattern!r} must be an option name, a "
            "namespace such as 'fig_*', or '*' for anything"
        )
        raise ValueError(msg)


#: Names no backend can register under: ``auto`` is `show()`'s inference,
#: ``fig`` and ``show`` are the two option namespaces.
RESERVED_BACKEND_NAMES = frozenset({"auto", "fig", "show"})

#: Members of `DisplayBackend` that magpylib itself uses: a subclass
#: redefining one would break the registry or the dispatcher.
_RESERVED_MEMBERS = frozenset(
    {"backends", "supports", "discover", "unaccepted_options", "unhandled_trace_types"}
)


def drawing_properties(trace):
    """Return `trace` without magpylib's metadata and placement keys.

    Traces mix drawing properties with magpylib metadata such as
    ``object_id`` and with the subplot cell, ``row`` and ``col``. Plotting
    libraries reject both, so read the placement, then strip both before
    handing a trace over::

        row, col = trace["row"], trace["col"]
        fig.add_trace(go.Mesh3d(**drawing_properties(trace)), row=row, col=col)

    with `fig` from `plotly.subplots.make_subplots`: a plain ``go.Figure()``
    rejects `row` and `col`.

    Parameters
    ----------
    trace : dict
        A trace from `Frame.traces`.

    Returns
    -------
    dict
        A copy of `trace` without the keys in `TRACE_META_KEYS` and
        `PLACEMENT_KEYS`.
    """
    return {
        k: v
        for k, v in trace.items()
        if k not in TRACE_META_KEYS and k not in PLACEMENT_KEYS
    }


@dataclass(frozen=True)
class Panel:
    """One cell of the subplot grid, owning its own axis metadata."""

    row: int = 1
    col: int = 1
    kind: Literal["scene3d", "chart2d"] = "scene3d"
    #: (3, 2) of (min, max) per axis, in the display unit, equal on the three
    #: axes, `zoom` applied. None for chart2d, which has no 3D range.
    ranges: np.ndarray | None = None
    #: The size, in m like the objects' positions and not in the display unit,
    #: the built-in backends give a sensor or dipole with
    #: ``style.sizemode="scaled"`` in this cell, from its extent; pass it to
    #: `pixel_sizes`. None for chart2d.
    autosize: float | None = None
    #: Axis titles. scene3d: {"x": "x (m)", "y": "y (m)", "z": "z (m)"}.
    #: chart2d: {"x": "path index", "y": "Bx, Bz (T)"} -- the panel's
    #: outputs, which always share one unit, over the path index.
    labels: dict[str, str] = field(default_factory=dict)
    #: The objects drawn in this cell, as they were passed to `show`: lists
    #: unpacked, collections whole, in the order given -- so an object passed
    #: on its own and inside a collection passed with it is here both ways.
    #: For what no trace carries: the `Collection` hierarchy (walk
    #: ``children``; `legendgroup` names only the outermost collection), and
    #: the object an ``object_id`` names (match ``id(obj)``). Live objects: a
    #: backend that keeps them keeps them alive.
    objects: tuple[Any, ...] = ()


@dataclass(frozen=True)
class Frame:
    """One step of the timeline. A static scene has exactly one. Every frame
    carries the same traces in the same order."""

    #: Display text: the 1-based path index ("0" for a static scene), as
    #: plotly's slider shows it. Index with `AnimationSettings.path_indices`.
    label: str = ""
    #: Trace dicts in magpylib's dialect. Left as dicts on purpose -- see the
    #: module docstring.
    traces: tuple[dict[str, Any], ...] = ()
    #: Models the user supplied in *this* backend's own format, already
    #: positioned and oriented. Keys: constructor, args, kwargs, coordsargs,
    #: kwargs_extra, object_id. Handed over only with
    #: `supports_native_traces`; without it magpylib warns and drops the
    #: model. **A backend that declares it and ignores these drops the user's
    #: models silently.**
    native_traces: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class AnimationSettings:
    """Timing and playback parameters for an animated scene."""

    #: The rate asked for. Fewer path steps than ``fps * time`` play slower:
    #: play the frames at `frame_duration`.
    fps: int = 20
    #: Total duration asked for, in seconds.
    time: float = 5
    #: Whether a slider over the path steps is shown.
    slider: bool = True
    #: Where a backend with `supports_animation_output` writes the animation:
    #: a ``.gif`` or ``.mp4`` path, or only the suffix for a temporary file.
    output: str | None = None
    #: Whether the animation starts again when it reaches its end.
    repeat: bool = False
    #: Milliseconds per frame, an int: ``time`` spread over the frames handed
    #: over. None when the scene is static.
    frame_duration: int | None = None
    #: The 0-based path step of each frame; may be downsampled.
    path_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class Scene:
    """Everything a backend needs in order to draw."""

    panels: tuple[Panel, ...] = ()
    frames: tuple[Frame, ...] = ()
    #: Figure title: `show(title=...)`, else a lone object's label, else None
    #: ("No objects to be displayed" for an empty `show()`).
    title: str | None = None
    animation: AnimationSettings = field(default_factory=AnimationSettings)
    #: User-supplied figure/axes/plotter to draw into, or None.
    canvas: Any = None
    #: Whether the backend may restyle a user-supplied canvas.
    canvas_update: bool = True
    #: Return the figure instead of displaying it.
    return_fig: bool = False
    #: Most entries a flat legend lists. Past it the built-in backends list
    #: that many and sum up the rest in one more, "... and k more", entry;
    #: 0 lists none. Matplotlib and pyvista apply it to each 3D panel's
    #: legend, plotly to its one figure-wide legend. A backend
    #: whose legend scales another way, such as a collapsible tree, may
    #: ignore it. Never applied to the traces themselves: the payload stays
    #: whole.
    legend_maxitems: int = 20
    #: Options magpylib does not interpret, values forwarded verbatim, names
    #: normalized (backend prefix removed, ``fig={...}`` flattened); one the
    #: backend did not declare is reported, and still forwarded. This is where
    #: backend-specific arguments live (plotly's `renderer`, pyvista's
    #: `jupyter_backend`, matplotlib's `antialiased`, ...), spelled bare or
    #: with the backend's name as prefix (``plotly_renderer``). Two
    #: conventional namespaces among them: ``fig_*`` for the library's figure
    #: (Plotly: `Figure.update`) and ``show_*`` for a show call, which the built-in
    #: backends forward to their library; `prefixed_options` reads either.
    #: A backend with no such library treats them as any other option.
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

    def prefixed_options(self, prefix: str) -> dict[str, Any]:
        """The options under ``<prefix>_``, with the prefix removed.

        ``show(fig_figsize=(3, 3))`` arrives as ``options["fig_figsize"]`` and
        ``scene.prefixed_options("fig")`` gives ``{"figsize": (3, 3)}``. The
        built-in backends read their figure-constructor and show-call
        arguments this way.
        """
        head = f"{prefix}_"
        return {
            k[len(head) :]: v for k, v in self.options.items() if k.startswith(head)
        }


class DisplayBackend:
    """Base class for a magpylib display backend.

    Subclass, set `name`, implement `show`. Capability flags declare what the
    backend can do; `show` warns and falls back rather than handing over
    something the backend cannot draw. Two exceptions: without
    `supports_colorgradient` the payload changes without a warning (arrows,
    and magnets sliced into `facecolor` bands), and a grid mixing 3D and 2D
    panels reaches a backend without `supports_subplots` unchanged, with a
    warning.

    Capabilities default to False so that a capability added in a later
    magpylib release never silently changes an existing backend's behavior.
    """

    #: Every registered backend, keyed by name. Subclasses with a `name`
    #: register themselves here on definition.
    #:
    #: :meta hide-value:
    backends: ClassVar[dict[str, "DisplayBackend"]] = {}

    name: ClassVar[str] = ""
    #: Shown, with `url`, where magpylib lists the backends -- the error for
    #: an unknown backend name, for one.
    description: ClassVar[str] = ""
    url: ClassVar[str] = ""

    #: Payload contract this backend was written against; pin it. Left unset,
    #: a backend counts as written against version 1, the first, so it is
    #: warned once the contract moves on. Inheriting the current version would
    #: make the warning impossible for exactly the backends that never said.
    api_version: ClassVar[int | None] = None

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
    #: Declaring the set lets magpylib say so. A trailing ``*`` accepts a
    #: namespace: the built-ins declare ``"fig_*"`` and ``"show_*"``, which
    #: they forward to their library. Modelled on xarray's
    #: `BackendEntrypoint.open_dataset_parameters`.
    accepts_options: ClassVar[frozenset[str] | None] = None

    #: Trace ``type`` values this backend knows how to draw. ``None`` means
    #: "assume everything". Declaring the set lets magpylib warn when a scene
    #: contains a type the backend never handles, instead of silently
    #: producing an incomplete figure -- which is what makes adding a new
    #: trace type safe. Modelled on HoloViews' element/plot registry.
    handles_traces: ClassVar[frozenset[str] | None] = None

    _discovered: ClassVar[bool] = False
    #: Whether the dispatcher has checked `accepts_options` against the names
    #: magpylib reserves; done once, at the first `show()`.
    _reserved_checked: bool = False

    #: Cleared at the end of ``magpylib/__init__.py``. Until then no entry
    #: point may be loaded: loading one runs third-party code that imports
    #: magpylib back, and the module it is told to subclass from --
    #: ``magpylib.graphics.backend`` -- cannot be imported while this package
    #: is still executing. Its parent's ``__init__`` pulls the style and
    #: display stack, which comes back round to ``default_settings`` before
    #: that name is bound, so the backend fails to load with a circular-import
    #: error and never registers.
    _importing: ClassVar[bool] = True

    #: What a backend declares besides `name` and `show`: the names
    #: `register_backend` accepts as keywords.
    _declarable: ClassVar[frozenset[str]] = frozenset(
        {
            "api_version",
            "description",
            "url",
            "supports_animation",
            "supports_subplots",
            "supports_colorgradient",
            "supports_animation_output",
            "supports_native_traces",
            "merge_traces",
            "accepts_options",
            "handles_traces",
        }
    )

    def __init_subclass__(cls, **kwargs):
        """Check the declaration and register any subclass that names itself.

        Checked here, at the class statement, so a malformed declaration is
        reported where it is written rather than at the first `show()` call,
        from inside a set difference.
        """
        super().__init_subclass__(**kwargs)
        clash = sorted(_RESERVED_MEMBERS & set(cls.__dict__))
        if clash:
            msg = (
                f"{cls.__name__} redefines {clash}, which DisplayBackend uses "
                "itself; choose other names."
            )
            raise TypeError(msg)
        unknown = sorted(
            n
            for n in cls.__dict__
            if n.startswith("supports_") and n not in DisplayBackend._declarable
        )
        if unknown:
            warnings.warn(
                f"{cls.__name__} declares {unknown}, which magpylib does not know "
                "and ignores; a typo?",
                stacklevel=user_stacklevel(),
            )
        for attr in ("handles_traces", "accepts_options"):
            value = getattr(cls, attr)
            if value is None:
                continue
            # materialised once: a generator would be spent by the check
            names = (
                None if isinstance(value, str) or not _iterable(value) else list(value)
            )
            if names is None or not all(isinstance(v, str) for v in names):
                msg = (
                    f"{cls.__name__}.{attr} must be a set of names or None, "
                    f"not {value!r}"
                )
                raise TypeError(msg)
            if attr == "accepts_options":
                for pattern in names:
                    _check_option_pattern(cls.__name__, attr, pattern)
            setattr(cls, attr, frozenset(names))
        for attr in ("description", "url"):
            if not isinstance(getattr(cls, attr), str):
                msg = f"{cls.__name__}.{attr} must be a string, not {getattr(cls, attr)!r}"
                raise TypeError(msg)
        for attr in (*cls._capability_attrs(), "merge_traces"):
            if not isinstance(getattr(cls, attr), bool):
                msg = (
                    f"{cls.__name__}.{attr} must be True or False, "
                    f"not {getattr(cls, attr)!r}"
                )
                raise TypeError(msg)
        if cls.api_version is not None:
            if isinstance(cls.api_version, bool) or not isinstance(
                cls.api_version, numbers.Integral
            ):
                msg = (
                    f"{cls.__name__}.api_version must be an integer such as "
                    f"{API_VERSION}, or None, not {cls.api_version!r}"
                )
                raise TypeError(msg)
            cls.api_version = int(cls.api_version)
        if not isinstance(cls.name, str):
            msg = f"{cls.__name__}.name must be a string, not {cls.name!r}"
            raise TypeError(msg)
        if cls.name in RESERVED_BACKEND_NAMES:
            msg = f"{cls.name!r} cannot name a display backend; it is reserved."
            raise ValueError(msg)
        # registered only by a class that names itself: subclassing a
        # registered backend without a `name` of its own is a tweak, not a
        # replacement of what it subclasses
        name = cls.__dict__.get("name", "")
        if name:
            old = cls.backends.get(name)
            if old is not None and type(old).__module__ != cls.__module__:
                # a notebook re-running its cell is the same module and stays
                # quiet; a package taking over "matplotlib" is worth a word
                warnings.warn(
                    f"{cls.__module__}.{cls.__qualname__} replaces the {name!r} "
                    f"display backend from {type(old).__module__}.",
                    stacklevel=user_stacklevel(),
                )
            cls.backends[name] = cls()

    @classmethod
    def _capability_attrs(cls):
        """Every ``supports_*`` attribute, inherited or added by a subclass."""
        return [name for name in dir(cls) if name.startswith("supports_")]

    @classmethod
    def discover(cls):
        """Load backends advertised by installed packages, once.

        Resolved on the first backend-name lookup after magpylib has finished
        importing, and cached. A lookup during the import -- the defaults tree
        validating its own default backend is one -- is not the first lookup
        for this purpose: it is answered from the built-ins alone, and the scan
        waits. Otherwise a third-party backend would be imported into a
        half-built magpylib and fail with a circular import.

        Deferring it also keeps the scan (a few ms, growing with the number of
        installed distributions) off the import path entirely, for a program
        that imports magpylib without ever drawing anything.
        """
        if cls._discovered or DisplayBackend._importing:
            return
        # set on the base, not on cls: discovering through a subclass must
        # still mark it done globally, or the base would rediscover later
        DisplayBackend._discovered = True
        for entry in entry_points(group=ENTRY_POINT_GROUP):
            # what this entry finds registered before it runs. Importing the
            # module its value names is itself what registers a backend, via
            # __init_subclass__, so "did this entry give us anything" cannot be
            # answered by looking at `loaded` alone.
            known = set(cls.backends)
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
                if key != entry.name:
                    warnings.warn(
                        f"Display backend {entry.value!r} is advertised as "
                        f"{entry.name!r} but named {key!r}; select it by "
                        f"{key!r}.",
                        stacklevel=2,
                    )
            elif entry.name not in cls.backends and set(cls.backends) == known:
                # A module or an instance is a legitimate thing for an entry
                # point to name -- importing it defines the subclass, which
                # registers itself -- so what is reported is the absence of a
                # backend, not the shape of what was loaded. Both tests apply:
                # the name may already be there because the module was
                # imported earlier, in which case nothing appeared here either.
                # Silence is the failure worth naming: a package whose class
                # comes out None on the magpylib it found looks installed and
                # does nothing.
                what = getattr(loaded, "__name__", None) or repr(loaded)
                warnings.warn(
                    f"Display backend {entry.name!r} advertised by "
                    f"{entry.value!r} registered no backend ({what}); it is "
                    "ignored. An entry point should name a DisplayBackend "
                    "subclass, or a module that defines one.",
                    stacklevel=2,
                )

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
        return frozenset(
            n for n in scene.options if not _option_matches(n, self.accepts_options)
        )

    def unhandled_trace_types(self, scene: Scene) -> frozenset[str]:
        """Trace types present in `scene` that this backend does not declare."""
        if self.handles_traces is None:
            return frozenset()
        present = {
            tr.get("type") for fr in scene.frames for tr in fr.traces if tr.get("type")
        }
        return frozenset(present - self.handles_traces)

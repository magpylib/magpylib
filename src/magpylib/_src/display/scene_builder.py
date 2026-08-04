"""Build a `Scene` from the generic frame structure produced by `get_frames`.

This is the single translation point between magpylib's internal, plotly-
flavoured trace dicts and the neutral public model in `api.py`. Keeping it in
one place means the internal builders never have to change, and the public
contract never has to inherit their vocabulary.
"""

import numpy as np

from magpylib._src.display.api import (
    AnimationSettings,
    Frame,
    LineStyle,
    MarkerStyle,
    Mesh,
    NativeTrace,
    Panel,
    Scatter2D,
    Scatter3D,
    Scene,
)

# how `mode` spells the parts of a scatter trace
_DRAW_PARTS = ("lines", "markers", "text")


def _draw_set(mode):
    """Translate a plotly-style `mode` string into a set of parts."""
    if not mode:
        return frozenset()
    return frozenset(p for p in _DRAW_PARTS if p in mode)


def _any_set(tr, keys):
    """True if any of `keys` is present and not None.

    Deliberately not a dataclass comparison: `marker_size` may be a per-point
    array, and comparing those raises "truth value is ambiguous".
    """
    return any(tr.get(k) is not None for k in keys)


def _line_style(tr):
    """Collect the flattened `line_*` keys, or None if none are set."""
    keys = ("line_color", "line_width", "line_dash")
    if not _any_set(tr, keys):
        return None
    return LineStyle(
        color=tr.get("line_color"),
        width=tr.get("line_width"),
        dash=tr.get("line_dash"),
    )


def _marker_style(tr):
    """Collect the flattened `marker_*` keys, or None if none are set."""
    keys = ("marker_color", "marker_size", "marker_symbol", "marker_line_color")
    if not _any_set(tr, keys):
        return None
    return MarkerStyle(
        color=tr.get("marker_color"),
        size=tr.get("marker_size"),
        symbol=tr.get("marker_symbol"),
        line_color=tr.get("marker_line_color"),
    )


def _texts(tr):
    """Normalize the `text` key to a tuple of strings, or None."""
    text = tr.get("text")
    if text is None:
        return None
    if isinstance(text, str):
        return (text,)
    return tuple(str(t) for t in text)


def _base_fields(tr):
    """The fields every trace carries."""
    return {
        "row": tr.get("row", 1),
        "col": tr.get("col", 1),
        "label": tr.get("name"),
        "group": tr.get("legendgroup"),
        "show_in_legend": bool(tr.get("showlegend", True)),
        "object_id": tr.get("object_id"),
    }


def _colormap(tr):
    """Normalize `colorscale` into a tuple of (stop, color) pairs."""
    cs = tr.get("colorscale")
    if cs is None:
        return None
    return tuple((float(stop), color) for stop, color in cs)


def _to_trace(tr):
    """Convert one internal trace dict into its public counterpart."""
    kind = tr.get("type")
    base = _base_fields(tr)

    if kind == "mesh3d":
        return Mesh(
            **base,
            vertices=np.stack([tr["x"], tr["y"], tr["z"]], axis=1),
            faces=np.stack([tr["i"], tr["j"], tr["k"]], axis=1),
            color=tr.get("color"),
            opacity=tr.get("opacity", 1.0),
            vertex_values=tr.get("intensity"),
            face_colors=tr.get("facecolor"),
            colormap=_colormap(tr),
            show_colorbar=bool(tr.get("showscale", False)),
        )

    if kind == "scatter3d":
        return Scatter3D(
            **base,
            points=np.stack([tr["x"], tr["y"], tr["z"]], axis=1),
            draw=_draw_set(tr.get("mode")),
            line=_line_style(tr),
            marker=_marker_style(tr),
            texts=_texts(tr),
            opacity=tr.get("opacity", 1.0),
        )

    if kind == "scatter":
        return Scatter2D(
            **base,
            x=np.asarray(tr["x"]),
            y=np.asarray(tr["y"]),
            draw=_draw_set(tr.get("mode")),
            line=_line_style(tr),
            marker=_marker_style(tr),
            texts=_texts(tr),
            hover_template=tr.get("hovertemplate"),
            group_title=tr.get("legendgrouptitle_text"),
        )

    msg = f"Cannot convert trace of type {kind!r} into the public scene model."
    raise ValueError(msg)


def _to_native(tr):
    """Convert one `extra_backend_traces` entry."""
    return NativeTrace(
        constructor=tr.get("constructor", ""),
        args=tuple(tr.get("args", ()) or ()),
        kwargs=dict(tr.get("kwargs", {}) or {}),
        coordsargs=dict(tr.get("coordsargs", {}) or {}),
        extra=dict(tr.get("kwargs_extra", {}) or {}),
    )


def _panels(data, subplot_specs):
    """Build the panel grid from `ranges`/`labels` and the spec array.

    `ranges` and `labels` are keyed by (row, col) in the internal payload;
    they move onto the panel that owns them. For 2D panels both are dropped:
    magpylib emits 3D axis metadata there, which describes nothing a chart
    can use.
    """
    ranges = data.get("ranges") or {}
    labels = data.get("labels") or {}
    specs = np.asarray(subplot_specs) if subplot_specs is not None else None

    panels = []
    for row, col in sorted(set(ranges) | set(labels)):
        kind = "scene3d"
        if specs is not None and specs.size:
            try:
                cell = specs[row - 1, col - 1]
            except IndexError:
                cell = None
            if isinstance(cell, dict) and cell.get("type") != "scene":
                kind = "chart2d"
        is_3d = kind == "scene3d"
        rng = ranges.get((row, col))
        panels.append(
            Panel(
                row=row,
                col=col,
                kind=kind,
                ranges=np.asarray(rng) if (is_3d and rng is not None) else None,
                labels=dict(labels.get((row, col), {})) if is_3d else {},
            )
        )
    return tuple(panels)


def _animation(data):
    """Build AnimationSettings from the `input_kwargs` bag."""
    ik = data.get("input_kwargs") or {}
    defaults = AnimationSettings()
    return AnimationSettings(
        fps=ik.get("animation_fps", defaults.fps),
        max_fps=ik.get("animation_maxfps", defaults.max_fps),
        max_frames=ik.get("animation_maxframes", defaults.max_frames),
        time=ik.get("animation_time", defaults.time),
        slider=ik.get("animation_slider", defaults.slider),
        output=ik.get("animation_output", defaults.output),
    )


def scene_from_generic(
    data,
    *,
    subplot_specs=None,
    canvas=None,
    canvas_update=True,
    fig_kwargs=None,
    show_kwargs=None,
):
    """Translate the internal `get_frames` payload into a public `Scene`."""
    frames = tuple(
        Frame(
            label=fr.get("name", ""),
            title=(fr.get("layout") or {}).get("title"),
            traces=tuple(_to_trace(t) for t in fr.get("data", ())),
            native_traces=tuple(
                _to_native(t) for t in fr.get("extra_backend_traces", ())
            ),
        )
        for fr in data.get("frames", ())
    )
    return Scene(
        panels=_panels(data, subplot_specs),
        frames=frames,
        canvas=canvas,
        canvas_update=canvas_update,
        animation=_animation(data),
        fig_kwargs=dict(fig_kwargs or {}),
        show_kwargs=dict(show_kwargs or {}),
    )

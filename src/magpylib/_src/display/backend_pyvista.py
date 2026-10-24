"""pyvista backend"""

# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=too-many-positional-arguments

import contextlib
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    import pyvista as pv
except ImportError as missing_module:  # pragma: no cover
    msg = (
        "Backend 'pyvista' requires installation of PyVista package, "
        "see https://docs.pyvista.org/getting-started/installation.html"
    )
    raise ModuleNotFoundError(msg) from missing_module

from matplotlib.colors import LinearSegmentedColormap
from pyvista.plotting.colors import Color  # pylint: disable=import-error

from magpylib._src.display.api import (
    drawing_properties,
    forward_options,
    warn_unused_options,
)
from magpylib._src.display.traces_utility import (
    get_trace_kw,
    legend_overflow_label,
    split_input_arrays,
)
from magpylib._src.utility import is_array_like, open_animation

# from magpylib._src.utility import format_obj_input

SYMBOLS_TO_PYVISTA = {
    ".": "o",
    "o": "o",
    "+": "+",
    "D": "d",
    "d": "d",
    "s": "s",
    "x": "x",
    "circle": "o",
    "cross": "+",
    "diamond": "d",
    "square": "s",
}

LINESTYLES_TO_PYVISTA = {
    "solid": "-",
    "-": "-",
    "dash": "--",
    "dashed": "--",
    "--": "--",
    "dashdot": "-.",
    "-.": "-.",
    "dotted": ":",
    ".": ":",
    ":": ":",
    "dot": ":",
    (0, (1, 1)): ":",
    "loosely dotted": ":",
    "loosely dashdotted": "-..",
    "longdash": ":",
    "longdashdot": "-..",
}


@lru_cache(maxsize=32)
def colormap_from_colorscale(colorscale, name="plotly_to_mpl", N=256, gamma=1.0):
    """Create Matplotlib colormap from plotly colorscale"""

    cs_rgb = [(v[0], Color(v[1]).float_rgb) for v in colorscale]
    cdict = {
        rgb_col: [
            (
                c[0],
                *[c[1][rgb_ind]] * 2,
            )
            for c in cs_rgb
        ]
        for rgb_ind, rgb_col in enumerate(("red", "green", "blue"))
    }
    return LinearSegmentedColormap(name, cdict, N, gamma)


def mesh3d_to_pyvista(trace):
    """Convert mesh3d trace input to a list of mesh constructor dict"""
    vertices = np.array([trace[k] for k in "xyz"], dtype=float).T
    faces = np.array([trace[k] for k in "ijk"]).T.flatten()
    faces = np.insert(faces, range(0, len(faces), 3), 3)
    colorscale = trace.get("colorscale", None)
    mesh = pv.PolyData(vertices, faces)
    facecolor = trace.get("facecolor", None)
    trace_pv = {
        "type": "mesh",
        "mesh": mesh,
        "color": trace.get("color", None),
        "scalars": trace.get("intensity", None),
        "opacity": trace.get("opacity", None),
    }
    if facecolor is not None:
        # pylint: disable=unsupported-assignment-operation
        # uint8, or numpy stores the rgb triples as int64: vtk keeps that
        # dtype through to the scene export, where it becomes a BigInt64Array
        # that webgl viewers cannot read - the scene then breaks at this actor
        mesh.cell_data["colors"] = np.array(
            [Color(c, default_color=(0, 0, 0)).int_rgb for c in facecolor],
            dtype=np.uint8,
        )
        trace_pv.update(
            {
                "scalars": "colors",
                "rgb": True,
                "preference": "cell",
            }
        )
    if colorscale is not None:
        trace_pv["cmap"] = colormap_from_colorscale(colorscale)
    return trace_pv


def scatter_to_pyvista(trace):
    """Convert scatter trace input to a list of plot or scatter constructor dicts
    Note 3d scatter plot is done with the mesh constructor and does not support symbols
    """
    traces = []

    # get kwargs
    mode = get_trace_kw(trace, "mode", none_replace="markers")
    line_color = get_trace_kw(trace, "line_color")
    line_width = get_trace_kw(trace, "line_width", none_replace=1)
    line_dash = get_trace_kw(trace, "line_dash")
    line_dash = LINESTYLES_TO_PYVISTA.get(line_dash, "-")
    marker_color = get_trace_kw(trace, "marker_color", none_replace=line_color)
    marker_size = get_trace_kw(trace, "marker_size", none_replace=1)
    marker_symbol = get_trace_kw(trace, "marker_symbol", none_replace="o")

    if trace["type"] == "scatter3d":
        points = np.array([trace[k] for k in "xyz"], dtype=float).T
        if len(points) == 0:
            # an empty trace, kept so that every frame has the same traces
            return traces
        if "lines" in mode:
            for (lcol,), inds in split_input_arrays(line_color):
                trace_pv_line = {
                    "type": "mesh",
                    "mesh": pv.lines_from_points(points[inds]),
                    "color": lcol,
                    "line_width": line_width,
                    "opacity": trace.get("opacity", None),
                }
                traces.append(trace_pv_line)
        if "markers" in mode:
            splits = split_input_arrays(marker_color, marker_size, ordered=False)
            for (mcolor, msize), inds in splits:
                if msize != 0:
                    trace_pv_marker = {
                        "type": "mesh",
                        "mesh": pv.PolyData(points[inds]),
                        "color": mcolor,
                        "point_size": msize,
                        "opacity": trace.get("opacity", None),
                    }
                    traces.append(trace_pv_marker)
        if "text" in mode and trace.get("text", False) and len(points) > 0:
            txt = trace["text"]
            txt = [txt] * len(points[0]) if isinstance(txt, str) else txt
            trace_pv_text = {
                "type": "point_labels",
                "points": points,
                "labels": txt,
                "always_visible": True,
            }
            traces.append(trace_pv_text)
    else:
        if "lines" in mode:
            splits = split_input_arrays(line_color, line_width, line_dash)
            for (lcolor, lwidth, ldash), inds in splits:
                trace_pv_line = {
                    "type": "line",
                    "x": trace["x"][inds],
                    "y": trace["y"][inds],
                    "color": lcolor,
                    "width": lwidth,
                    "style": ldash,
                    "label": trace.get("name", ""),
                }
                traces.append(trace_pv_line)
        if "markers" in mode:
            for (msize,), inds in split_input_arrays(marker_size, ordered=False):
                if msize != 0:
                    mcol = marker_color
                    if is_array_like(mcol):
                        mcol = mcol[inds]
                    if is_array_like(marker_symbol):
                        msymb = marker_symbol[inds]
                        msymb = [SYMBOLS_TO_PYVISTA.get(s, "o") for s in msymb]
                    else:
                        msymb = SYMBOLS_TO_PYVISTA.get(marker_symbol, "o")
                    trace_pv_marker = {
                        "type": "scatter",
                        "x": trace["x"][inds],
                        "y": trace["y"][inds],
                        "color": mcol,
                        "size": msize,
                        "style": msymb,
                    }
                    traces.append(trace_pv_marker)
    return traces


def generic_trace_to_pyvista(trace):
    """Transform a generic trace into a pyvista traces"""
    row, col = trace.get("row", 1), trace.get("col", 1)
    trace = drawing_properties(trace)
    traces_pv = []
    if trace["type"] == "mesh3d":
        traces_pv.append(mesh3d_to_pyvista(trace))
    elif trace["type"] in ("scatter", "scatter3d"):
        traces_pv.extend(scatter_to_pyvista(trace))
    else:  # pragma: no cover
        msg2 = (
            "Unsupported trace type: "
            f"{trace['type']!r} cannot be transformed into a PyVista trace."
        )
        raise ValueError(msg2)
    showlegend = trace.get("showlegend", False)
    for tr in traces_pv:
        tr["row"] = row - 1
        tr["col"] = col - 1
        if tr["type"] == "point_labels":
            continue
        if showlegend:
            showlegend = False  # show only first subtrace
            if "label" not in tr:
                tr["label"] = trace.get("name", "")
            leg_title = trace.get("legendgrouptitle_text", None)
            if leg_title is not None:
                tr["label"] += f" ({leg_title})"
        else:
            # a 2D line arrives labelled with the trace's name; a label is a
            # legend entry in a chart, so every other subtrace -- and every
            # subtrace of a trace left out of the legend -- drops it
            tr.pop("label", None)
        if not tr.get("label", ""):
            tr.pop("label", None)
    return traces_pv


def add_capped_legend(canvas, labels, maxitems):
    """Add the active subplot's legend with at most `maxitems` of its
    `labels`, the rest summed up in one entry. 0 adds none."""
    if maxitems == 0 or not labels:
        return
    hidden = len(labels) - maxitems
    if hidden <= 0:
        canvas.add_legend(bcolor=None)
        return
    # a bare label takes its face and color from the labelled actor at the
    # same index, so the entries kept look as they would uncapped
    entries = [
        *labels[:maxitems],
        {"label": legend_overflow_label(hidden), "color": "gray", "face": "none"},
    ]
    canvas.add_legend(labels=entries, bcolor=None)


def display_pyvista(scene):
    """Display objects and paths graphically using the pyvista library."""
    canvas = scene.canvas
    canvas_update = scene.canvas_update
    return_fig = scene.return_fig
    legend_maxitems = scene.legend_maxitems
    repeat = scene.animation.repeat
    # options magpylib does not interpret are the backend's own
    jupyter_backend = scene.options.get("jupyter_backend")
    mp4_quality = scene.options.get("mp4_quality", 5)
    # native_traces are intentionally unused; see supports_native_traces
    frames = [fr.traces for fr in scene.frames]

    fig_kwargs = scene.prefixed_options("fig")
    show_kwargs = scene.prefixed_options("show")
    unused_show = [f"show_{k}" for k in show_kwargs]
    if jupyter_backend is not None:
        unused_show.append("jupyter_backend")

    animation = scene.is_animation
    show_canvas = False
    if canvas is None:
        if not return_fig:
            show_canvas = True  # pragma: no cover
        # the subplot grid is magpylib's to set; off-screen rendering is the
        # user's choice too, e.g. fig_off_screen=True to take a screenshot
        canvas = forward_options(
            pv.Plotter,
            "fig",
            fig_kwargs,
            defaults={"off_screen": animation},
            shape=(scene.n_rows, scene.n_cols),
        )
    else:
        warn_unused_options(
            [f"fig_{k}" for k in fig_kwargs], "the plotter was given as canvas"
        )

    charts = {}
    jupyter_backend = show_kwargs.pop("jupyter_backend", jupyter_backend)
    if jupyter_backend is None:
        jupyter_backend = pv.global_theme.jupyter_backend
    legend_labels = {}
    charts_max_ind = 0

    def draw_frame(frame_ind):
        nonlocal charts_max_ind
        for tr0 in frames[frame_ind]:
            for tr1 in generic_trace_to_pyvista(tr0):
                row = tr1.pop("row", 1)
                col = tr1.pop("col", 1)
                typ = tr1.pop("type")
                if frame_ind == 0:
                    labels_rc = legend_labels.setdefault((row, col), [])
                    if tr1.get("label", ""):
                        labels_rc.append(tr1["label"])
                canvas.subplot(row, col)
                panel = scene.panel(row + 1, col + 1)
                if panel is None or panel.kind == "scene3d":
                    getattr(canvas, f"add_{typ}")(**tr1)
                else:
                    if charts.get((row, col)) is None:
                        charts_max_ind += 1
                        # the chart is magpylib's own, even on a given
                        # plotter, so it is labelled whatever canvas_update says
                        axis_labels = panel.labels if panel is not None else {}
                        charts[(row, col)] = pv.Chart2D(
                            x_label=axis_labels.get("x", ""),
                            y_label=axis_labels.get("y", ""),
                        )
                        canvas.add_chart(charts[(row, col)])
                    getattr(charts[(row, col)], typ)(**tr1)
            # in pyvista there is no way to set the bounds so we add corners with
            # a transparent scatter plot to set the ranges and zoom correctly.
            # 2D panels carry no 3D range -- there are no bounds to set there.
            panel = scene.panel(row + 1, col + 1)
            if panel is not None and panel.ranges is not None:
                pts = np.array(np.meshgrid(*panel.ranges)).T.reshape(-1, 3)
                canvas.add_mesh(pv.PolyData(pts), opacity=0)
            with contextlib.suppress(StopIteration, IndexError):
                canvas.remove_scalar_bar()
                # needs to happen in the loop otherwise they cummulate
                # while the max of 10 is reached and throws a ValueError

        for (row, col), labels_rc in legend_labels.items():
            canvas.subplot(row, col)
            # match other backends plotter properties
            if canvas_update:
                if callable(canvas.show_axes):
                    canvas.show_axes()
                canvas.camera.azimuth = -90
                canvas.set_background("gray", top="white")
            if scene.panel_kind(row + 1, col + 1) == "scene3d":
                add_capped_legend(canvas, labels_rc, legend_maxitems)

    def run_animation(filename, embed=True):
        # embed=True, embeds the animation into the notebook page and is necessary when using
        # temp files
        nonlocal show_canvas, charts_max_ind, charts
        # absolute here, so the writer and the viewer get the same file
        # (imageio expands ~, pyvista prepends pv.FIGURE_PATH to relative names)
        filename = Path(filename).expanduser().absolute()
        suff = filename.suffix
        if suff == ".gif":
            loop = 1 if repeat is False else 0 if repeat is True else int(repeat)
            canvas.open_gif(
                filename, loop=loop, fps=1000 / scene.animation.frame_duration
            )
        elif suff == ".mp4":
            canvas.open_movie(
                filename,
                framerate=1000 / scene.animation.frame_duration,
                quality=mp4_quality,
            )

        for frame_ind, _ in enumerate(frames):
            canvas.clear_actors()
            for ind in range(charts_max_ind):
                canvas.remove_chart(ind)
            charts_max_ind = 0
            charts = {}
            draw_frame(frame_ind)
            canvas.write_frame()
        canvas.close()
        show_canvas = False
        return open_animation(filename, embed=embed)

    # pyvista has no figure-level title: add_title applies to the active
    # subplot, so on a grid it would read as that one panel's title
    if canvas_update and scene.title and not scene.has_subplots:
        canvas.subplot(0, 0)
        canvas.add_title(scene.title)

    if not scene.is_animation:
        draw_frame(0)
    elif animation:
        animation_output = scene.animation.output
        animation_output = "gif" if animation_output is None else animation_output
        if animation_output in ("gif", "mp4"):
            fd, temp = tempfile.mkstemp(
                prefix="magpylib_animation_", suffix=f".{animation_output}"
            )
            os.close(fd)
            viewer_has_file = False
            try:
                viewer_has_file = run_animation(temp, embed=True)
            finally:
                # a notebook embeds the file; a viewer outside reads it after
                # show() returns, so then it stays in the temp dir
                if not viewer_has_file:
                    Path(temp).unlink(missing_ok=True)
        else:
            run_animation(animation_output, embed=True)

    if not show_canvas:
        warn_unused_options(
            unused_show,
            "the plotter is returned or was given as canvas rather than shown",
        )
    if return_fig and not show_canvas:
        return canvas
    if show_canvas:  # pragma: no cover
        forward_options(
            canvas.show,
            "show",
            show_kwargs,
            defaults={"jupyter_backend": jupyter_backend},
        )
    return None

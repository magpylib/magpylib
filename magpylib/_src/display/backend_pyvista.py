"""pyvista backend"""
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
import os
import tempfile
import warnings
from functools import lru_cache

import numpy as np

try:
    import pyvista as pv
except ImportError as missing_module:  # pragma: no cover
    raise ModuleNotFoundError(
        """In order to use the pyvista plotting backend, you need to install pyvista via pip or
        conda, see https://docs.pyvista.org/getting-started/installation.html"""
    ) from missing_module

from pyvista.plotting.colors import Color  # pylint: disable=import-error
from matplotlib.colors import LinearSegmentedColormap
from magpylib._src.utility import open_animation

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

INCOMPATIBLE_JUPYTER_BACKENDS_2D = {"panel", "ipygany", "pythreejs"}


@lru_cache(maxsize=32)
def colormap_from_colorscale(colorscale, name="plotly_to_mpl", N=256, gamma=1.0):
    """Create matplotlib colormap from plotly colorscale"""

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


def generic_trace_to_pyvista(trace, jupyter_backend=None):
    """Transform a generic trace into a pyvista trace"""
    traces_pv = []
    leg_title = trace.get("legendgrouptitle_text", None)
    if trace["type"] == "mesh3d":
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
            mesh.cell_data["colors"] = [
                Color(c, default_color=(0, 0, 0)).int_rgb for c in facecolor
            ]
            trace_pv.update(
                {
                    "scalars": "colors",
                    "rgb": True,
                    "preference": "cell",
                }
            )
        traces_pv.append(trace_pv)
        if colorscale is not None:
            # ipygany does not support custom colorsequences
            if jupyter_backend == "ipygany":
                trace_pv["cmap"] = "PiYG"
            else:
                trace_pv["cmap"] = colormap_from_colorscale(colorscale)
    elif "scatter" in trace["type"]:
        line = trace.get("line", {})
        line_color = line.get("color", trace.get("line_color", None))
        line_width = line.get("width", trace.get("line_width", None))
        line_width = 1 if line_width is None else line_width
        line_style = line.get("dash", trace.get("line_dash"))
        marker = trace.get("marker", {})
        marker_color = marker.get("color", trace.get("marker_color", None))
        marker_size = marker.get("size", trace.get("marker_size", None))
        marker_size = 1 if marker_size is None else marker_size
        marker_symbol = marker.get("symbol", trace.get("marker_symbol", None))
        mode = trace.get("mode", None)
        mode = "markers" if mode is None else mode
        if trace["type"] == "scatter3d":
            points = np.array([trace[k] for k in "xyz"], dtype=float).T
            if "lines" in mode:
                trace_pv_line = {
                    "type": "mesh",
                    "mesh": pv.lines_from_points(points),
                    "color": line_color,
                    "line_width": line_width,
                    "opacity": trace.get("opacity", None),
                }
                traces_pv.append(trace_pv_line)
            if "markers" in mode:
                trace_pv_marker = {
                    "type": "mesh",
                    "mesh": pv.PolyData(points),
                    "color": marker_color,
                    "point_size": marker_size,
                    "opacity": trace.get("opacity", None),
                }
                traces_pv.append(trace_pv_marker)
            if "text" in mode and trace.get("text", False) and len(points) > 0:
                txt = trace["text"]
                txt = [txt] * len(points[0]) if isinstance(txt, str) else txt
                trace_pv_text = {
                    "type": "point_labels",
                    "points": points,
                    "labels": txt,
                    "always_visible": True,
                }
                traces_pv.append(trace_pv_text)
        elif trace["type"] == "scatter":
            if "lines" in mode:
                trace_pv_line = {
                    "type": "line",
                    "x": trace["x"],
                    "y": trace["y"],
                    "color": line_color,
                    "width": line_width,
                    "style": LINESTYLES_TO_PYVISTA.get(line_style, "-"),
                    "label": trace.get("name", ""),
                }
                traces_pv.append(trace_pv_line)
            if "markers" in mode:
                trace_pv_marker = {
                    "type": "scatter",
                    "x": trace["x"],
                    "y": trace["y"],
                    "color": marker_color,
                    "size": marker_size,
                    "style": SYMBOLS_TO_PYVISTA.get(marker_symbol, "o"),
                }
                marker_size = (
                    marker_size
                    if isinstance(marker_size, (list, tuple, np.ndarray))
                    else np.array([marker_size])
                )
                for size in np.unique(marker_size):
                    tr = trace_pv_marker.copy()
                    mask = marker_size == size
                    tr = {
                        **tr,
                        "x": np.array(tr["x"])[mask],
                        "y": np.array(tr["y"][mask]),
                        "size": size,
                    }
                    traces_pv.append(tr)
    else:  # pragma: no cover
        raise ValueError(
            f"Trace type {trace['type']!r} cannot be transformed into pyvista trace"
        )
    showlegend = trace.get("showlegend", False)
    for tr in traces_pv:
        tr["row"] = trace.get("row", 1) - 1
        tr["col"] = trace.get("col", 1) - 1
        if tr["type"] != "point_labels":
            if showlegend:
                showlegend = False  # show only first subtrace
                if "label" not in tr:
                    tr["label"] = trace.get("name", "")
                if leg_title is not None:
                    tr["label"] += f" ({leg_title})"
        if not tr.get("label", ""):
            tr.pop("label", None)
    return traces_pv


def display_pyvista(
    data,
    canvas=None,
    return_fig=False,
    jupyter_backend=None,
    max_rows=None,
    max_cols=None,
    subplot_specs=None,
    repeat=False,
    legend_maxitems=20,
    fig_kwargs=None,
    show_kwargs=None,
    mp4_quality=5,
    **kwargs,  # pylint: disable=unused-argument
):
    """Display objects and paths graphically using the pyvista library."""

    frames = data["frames"]

    fig_kwargs = {} if not fig_kwargs else fig_kwargs
    show_kwargs = {} if not show_kwargs else show_kwargs
    show_kwargs = {**show_kwargs}

    animation = bool(len(frames) > 1)
    max_rows = max_rows if max_rows is not None else 1
    max_cols = max_cols if max_cols is not None else 1
    show_canvas = False
    if canvas is None:
        if not return_fig:
            show_canvas = True  # pragma: no cover
        canvas = pv.Plotter(
            shape=(max_rows, max_cols),
            off_screen=animation,
            **fig_kwargs,
        )

    charts = {}
    jupyter_backend = show_kwargs.pop("jupyter_backend", jupyter_backend)
    if jupyter_backend is None:
        jupyter_backend = pv.global_theme.jupyter_backend
    jupyter_backend_2D_compatible = (
        jupyter_backend not in INCOMPATIBLE_JUPYTER_BACKENDS_2D
    )
    count_with_labels = {}
    charts_max_ind = 0

    def draw_frame(frame_ind):
        nonlocal count_with_labels, charts_max_ind
        frame = frames[frame_ind]
        for tr0 in frame["data"]:
            for tr1 in generic_trace_to_pyvista(tr0, jupyter_backend=jupyter_backend):
                row = tr1.pop("row", 1)
                col = tr1.pop("col", 1)
                typ = tr1.pop("type")
                if frame_ind == 0:
                    if (row, col) not in count_with_labels:
                        count_with_labels[(row, col)] = 0
                    if tr1.get("label", ""):
                        count_with_labels[(row, col)] += 1
                canvas.subplot(row, col)
                if subplot_specs[row, col]["type"] == "scene":
                    getattr(canvas, f"add_{typ}")(**tr1)
                    canvas.show_axes()
                else:
                    if jupyter_backend_2D_compatible:
                        if charts.get((row, col), None) is None:
                            charts_max_ind += 1
                            charts[(row, col)] = pv.Chart2D()
                            canvas.add_chart(charts[(row, col)])
                        getattr(charts[(row, col)], typ)(**tr1)
                    else:
                        warnings.warn(
                            f"The set `jupyter_backend={jupyter_backend}` is incompatible with "
                            "2D plots. Empty plots will be shown instead"
                        )
        for rowcol, count in count_with_labels.items():
            if 0 < count <= legend_maxitems:
                row, col = rowcol
                canvas.subplot(row, col)
                if subplot_specs[row, col]["type"] == "scene":
                    canvas.add_legend(bcolor=None)
        # match other backends plotter properties
        canvas.set_background("gray", top="white")
        canvas.camera.azimuth = -90
        try:
            canvas.remove_scalar_bar()
        except IndexError:
            # try to remove scalar bar, if none, pass
            pass

    def run_animation(filename, embed=True):
        # embed=True, embeds the animation into the notebook page and is necessary when using
        # temp files
        nonlocal show_canvas, charts_max_ind, charts

        suff = os.path.splitext(filename)[-1]
        if suff == ".gif":
            loop = 1 if repeat is False else 0 if repeat is True else int(repeat)
            canvas.open_gif(filename, loop=loop, fps=1000 / data["frame_duration"])
        elif suff == ".mp4":
            canvas.open_movie(
                filename, framerate=1000 / data["frame_duration"], quality=mp4_quality
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
        open_animation(filename, embed=embed)

    if len(frames) == 1:
        draw_frame(0)
    elif animation:
        animation_output = data["input_kwargs"].get("animation_output", None)
        animation_output = "gif" if animation_output is None else animation_output
        if animation_output in ("gif", "mp4"):
            with tempfile.NamedTemporaryFile(suffix=f".{animation_output}") as temp:
                run_animation(temp.name, embed=True)
        else:
            run_animation(animation_output, embed=True)

    if return_fig and not show_canvas:
        return canvas
    if show_canvas:
        canvas.show(jupyter_backend=jupyter_backend, **show_kwargs)  # pragma: no cover
    return None

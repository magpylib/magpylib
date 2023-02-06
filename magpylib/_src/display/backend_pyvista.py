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
from magpylib._src.display.traces_generic import get_frames
from magpylib._src.utility import show_gif, show_video

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
        mode = trace.get("mode", "markers+lines")
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
            if "text" in mode and trace.get("text", False):
                trace_pv_text = {
                    "type": "point_labels",
                    "points": points,
                    "labels": trace["text"],
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
                if not isinstance(marker_size, (list, tuple, np.ndarray)):
                    marker_size = np.array([marker_size])
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
    for ind, tr in enumerate(traces_pv):
        tr["row"] = trace.get("row", 1) - 1
        tr["col"] = trace.get("col", 1) - 1
        if ind == 0 and trace.get("showlegend", False):
            if "label" not in tr:
                tr["label"] = trace.get("name", "")
            if leg_title is not None:
                tr["label"] += f" ({leg_title})"
    return traces_pv


def display_pyvista(
    data,
    canvas=None,
    animation=False,
    return_fig=False,
    jupyter_backend=None,
    max_rows=None,
    max_cols=None,
    subplot_specs=None,
    animation_output="gif",
    repeat=False,
    **kwargs,
):
    """Display objects and paths graphically using the pyvista library."""

    max_rows = max_rows if max_rows is not None else 1
    max_cols = max_cols if max_cols is not None else 1
    show_canvas = False
    if canvas is None:
        if not return_fig:
            show_canvas = True  # pragma: no cover
        canvas = pv.Plotter(shape=(max_rows, max_cols), off_screen=animation)

    frames = data["frames"]

    charts = {}
    if jupyter_backend is None:
        jupyter_backend = pv.global_theme.jupyter_backend
    jupyter_backend_2D_compatible = (
        jupyter_backend not in INCOMPATIBLE_JUPYTER_BACKENDS_2D
    )
    warned2d = False

    def draw_frame(frame):
        nonlocal warned2d
        for tr0 in frame["data"]:
            for tr1 in generic_trace_to_pyvista(tr0, jupyter_backend=jupyter_backend):
                row = tr1.pop("row", 1)
                col = tr1.pop("col", 1)
                typ = tr1.pop("type")
                canvas.subplot(row, col)
                if subplot_specs[row, col]["type"] == "scene":
                    getattr(canvas, f"add_{typ}")(**tr1)
                    canvas.show_axes()
                else:
                    if jupyter_backend_2D_compatible:
                        if charts.get((row, col), None) is None:
                            charts[(row, col)] = pv.Chart2D()
                            canvas.add_chart(charts[(row, col)])
                        getattr(charts[(row, col)], typ)(**tr1)
                    elif not warned2d:
                        warnings.warn(
                            f"The set `{jupyter_backend=}` is incompatible with 2D plots. "
                            "Empty plots will be shown instead"
                        )
                        warned2d = True
        # match other backends plotter properties
        canvas.set_background("gray", top="white")
        canvas.camera.azimuth = -90
        try:
            canvas.remove_scalar_bar()
        except IndexError:
            pass

    def run_animation(filename):
        nonlocal show_canvas

        suff = os.path.splitext(filename)[-1]
        if suff == ".gif":
            canvas.open_gif(
                filename, loop=int(repeat), fps=1000 / data["frame_duration"]
            )
            show_fn = show_gif
        elif suff == ".mp4":
            canvas.open_movie(
                filename, framerate=1000 / data["frame_duration"], quality=5
            )
            show_fn = show_video
        else:
            raise ValueError(
                "Animation filename must end with `'.gif'` or `'mp4'`, "
                f"received {suff!r} instead"
            )

        for frame in frames:
            canvas.clear_actors()
            draw_frame(frame)
            canvas.write_frame()
        canvas.close()
        show_canvas = False
        show_fn(filename)

    if len(frames) == 1:
        draw_frame(frames[0])
    elif animation:
        if animation_output in ("gif", "mp4"):
            with tempfile.TemporaryFile() as temp:
                run_animation(f"{temp.name}_animation.{animation_output}")
        else:
            run_animation(animation_output)

    if return_fig and not show_canvas:
        return canvas
    if show_canvas:
        canvas.show(jupyter_backend=jupyter_backend)  # pragma: no cover
    return None

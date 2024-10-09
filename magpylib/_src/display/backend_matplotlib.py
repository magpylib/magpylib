"""matplotlib backend"""

# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=import-outside-toplevel
# pylint: disable=wrong-import-position
# pylint: disable=too-many-positional-arguments

import os
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.animation import FuncAnimation

from magpylib._src.display.traces_utility import subdivide_mesh_by_facecolor

if os.getenv("MAGPYLIB_MPL_SVG") == "true":  # pragma: no cover
    from matplotlib_inline.backend_inline import set_matplotlib_formats

    set_matplotlib_formats("svg")

SYMBOLS_TO_MATPLOTLIB = {
    "circle": "o",
    "cross": "+",
    "diamond": "d",
    "square": "s",
    "x": "x",
}

LINE_STYLES_TO_MATPLOTLIB = {
    "solid": "-",
    "dash": "--",
    "dashdot": "-.",
    "dot": (0, (1, 1)),
    "longdash": "loosely dotted",
    "longdashdot": "loosely dashdotted",
}

SCATTER_KWARGS_LOOKUPS = {
    "ls": ("line", "dash"),
    "lw": ("line", "width"),
    "color": ("line", "color"),
    "marker": ("marker", "symbol"),
    "mfc": ("marker", "color"),
    "mec": ("marker", "color"),
    "ms": ("marker", "size"),
}


class StripedHandler:
    """
    Handler for creating a striped legend key using given color data.

    Parameters
    ----------
    color_data : dict
        Dictionary containing color names as keys and their respective proportions as values.

    Attributes
    ----------
    colors : list
        List of colors extracted from the color_data dictionary.
    proportions : list
        Normalized list of proportions extracted from the color_data dictionary.
    """

    def __init__(self, color_data):
        total = sum(color_data.values())
        self.colors = list(color_data.keys())
        self.proportions = [value / total for value in color_data.values()]

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        # pylint: disable=unused-argument
        """Create custom legend key"""
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch_width = width
        current_position = x0

        for color, proportion in zip(self.colors, self.proportions):
            handlebox.add_artist(
                patches.Rectangle(
                    [current_position, y0], patch_width * proportion, height, fc=color
                )
            )
            current_position += patch_width * proportion


def generic_trace_to_matplotlib(trace, antialiased=True):
    """Transform a generic trace into a matplotlib trace"""
    traces_mpl = []
    leg_title = trace.get("legendgrouptitle_text", None)
    showlegend = trace.get("showlegend", True)
    if trace["type"] == "mesh3d":
        subtraces = [trace]
        has_facecolor = trace.get("facecolor", None) is not None
        if has_facecolor:
            subtraces = subdivide_mesh_by_facecolor(trace)
        for ind, subtrace in enumerate(subtraces):
            x, y, z = np.array([subtrace[k] for k in "xyz"], dtype=float)
            triangles = np.array([subtrace[k] for k in "ijk"]).T
            tr_mesh = {
                "constructor": "plot_trisurf",
                "args": (x, y, z),
                "kwargs": {
                    "triangles": triangles,
                    "alpha": subtrace.get("opacity", None),
                    "color": subtrace.get("color", None),
                    "linewidth": 0,
                    "antialiased": antialiased,
                },
            }
            if showlegend and has_facecolor:
                tr_mesh["legend_handler"] = StripedHandler(Counter(trace["facecolor"]))
            if ind != 0:  # hide substrace legends except first
                tr_mesh["kwargs"]["label"] = "_nolegend_"
            traces_mpl.append(tr_mesh)
    elif "scatter" in trace["type"]:
        props = {
            k: trace.get(v[0], {}).get(v[1], trace.get("_".join(v), None))
            for k, v in SCATTER_KWARGS_LOOKUPS.items()
        }
        coords_str = "xyz"
        if trace["type"] == "scatter":
            coords_str = "xy"
            # marker size is proportional to area, not radius like generic
            props["ms"] = np.pi * props["ms"] ** 2
        coords = np.array([trace[k] for k in coords_str], dtype=float)
        if isinstance(props["ms"], (list, tuple, np.ndarray)):
            traces_mpl.append(
                {
                    "constructor": "scatter",
                    "args": (*coords,),
                    "kwargs": {
                        "s": props["ms"],
                        "color": props["mec"],
                        "marker": SYMBOLS_TO_MATPLOTLIB.get(
                            props["marker"], props["marker"]
                        ),
                        "label": None,
                    },
                }
            )
            props.pop("ms")
            props.pop("marker")
        if "ls" in props:
            props["ls"] = LINE_STYLES_TO_MATPLOTLIB.get(props["ls"], props["ls"])
        if "marker" in props:
            props["marker"] = SYMBOLS_TO_MATPLOTLIB.get(
                props["marker"], props["marker"]
            )
        mode = trace.get("mode", None)
        mode = "markers" if mode is None else mode
        if "lines" not in mode:
            props["ls"] = ""
        if "markers" in mode:
            if not props.get("marker", None):
                props["marker"] = "o"
        else:
            props["marker"] = None
        if "text" in mode and trace.get("text", False) and len(coords) > 0:
            txt = trace["text"]
            txt = [txt] * len(coords[0]) if isinstance(txt, str) else txt
            for *coords_s, txt in zip(*coords, txt):
                traces_mpl.append(
                    {
                        "constructor": "text",
                        "args": (*coords_s, txt),
                    }
                )
        traces_mpl.append(
            {
                "constructor": "plot",
                "args": coords,
                "kwargs": {
                    **{k: v for k, v in props.items() if v is not None},
                    "alpha": trace.get("opacity", 1),
                },
            }
        )
    else:  # pragma: no cover
        raise ValueError(
            f"Trace type {trace['type']!r} cannot be transformed into matplotlib trace"
        )
    for tr_mesh in traces_mpl:
        tr_mesh["row"] = trace.get("row", 1)
        tr_mesh["col"] = trace.get("col", 1)
        tr_mesh["kwargs"] = tr_mesh.get("kwargs", {})
        if tr_mesh["constructor"] != "text":
            if showlegend:
                if "label" not in tr_mesh["kwargs"]:
                    tr_mesh["kwargs"]["label"] = trace.get("name", "")
                    if leg_title is not None:
                        tr_mesh["kwargs"]["label"] += f" ({leg_title})"
            else:
                tr_mesh["kwargs"]["label"] = "_nolegend"
    return traces_mpl


def extract_axis_from_row_col(fig, row, col):
    "Return axis from row and col values"

    def geom(ax):
        return ax.get_subplotspec().get_topmost_subplotspec().get_geometry()

    # get nrows and ncols of fig for first axis
    rc = geom(fig.axes[0])[:2]
    # get the axis index based on row first
    default_ind = rc[0] * (row - 1) + col - 1
    # get last index of geometry, gives the actual index,
    # since axis can be added in a different order
    inds = [geom(ax)[-1] for ax in fig.axes]
    # retrieve first index that matches
    ind = inds.index(default_ind)
    ax = fig.axes[ind]
    return ax


def process_extra_trace(model):
    "process extra trace attached to some magpylib object"
    trace3d = model.copy()
    kw = trace3d.pop("kwargs_extra")
    trace3d.update({"row": kw["row"], "col": kw["col"]})
    kw = {
        "alpha": kw["opacity"],
        "color": kw["color"],
        "label": kw["name"] if kw["showlegend"] else "_nolegend_",
    }
    trace3d["kwargs"] = {**kw, **trace3d["kwargs"]}
    return trace3d


def display_matplotlib(
    data,
    canvas=None,
    repeat=False,
    return_fig=False,
    canvas_update="auto",
    return_animation=False,
    max_rows=None,
    max_cols=None,
    subplot_specs=None,
    antialiased=True,
    legend_maxitems=20,
    fig_kwargs=None,
    show_kwargs=None,
    **kwargs,  # pylint: disable=unused-argument
):
    """Display objects and paths graphically using the matplotlib library."""
    frames = data["frames"]
    ranges = data["ranges"]
    labels = data["labels"]

    # only update layout if canvas is not provided
    fig_kwargs = {} if not fig_kwargs else fig_kwargs
    show_kwargs = {} if not show_kwargs else show_kwargs
    show_kwargs = {**show_kwargs}

    for fr in frames:
        new_data = []
        for tr in fr["data"]:
            new_data.extend(generic_trace_to_matplotlib(tr, antialiased=antialiased))
        for model in fr["extra_backend_traces"]:
            new_data.append(process_extra_trace(model))
        fr["data"] = new_data

    show_canvas = bool(canvas is None)
    axes = {}
    if canvas_update:
        fig_kwargs["dpi"] = fig_kwargs.get("dpi", 80)
        if fig_kwargs.get("figsize", None) is None:
            figsize = (8, 8)
            ratio = subplot_specs.shape[1] / subplot_specs.shape[0]
            if legend_maxitems != 0:
                ratio *= 1.5  # extend horizontal ratio if legend is present
            fig_kwargs["figsize"] = (figsize[0] * ratio, figsize[1])
    if canvas is None:
        fig = plt.figure(**{"tight_layout": True, **fig_kwargs})
    elif isinstance(canvas, matplotlib.axes.Axes):
        fig = canvas.get_figure()
        if max_rows is not None or max_cols is not None:
            raise ValueError(
                "Provided canvas is an instance of `matplotlib.axes.Axes` and does not support "
                "`rows` or `cols` attributes. Use an instance of `matplotlib.figure.Figure` "
                "instead"
            )
    elif isinstance(canvas, matplotlib.figure.Figure):
        fig = canvas
    else:
        raise TypeError(
            "The `canvas` parameter must be one of `[None, matplotlib.axes.Axes, "
            f"matplotlib.figure.Figure]`. Received type {type(canvas)!r} instead"
        )
    if canvas is not None and canvas_update:
        fig.set_size_inches(*fig_kwargs["figsize"], forward=True)
        fig.set_dpi(fig_kwargs["dpi"])
    if max_rows is None and max_cols is None:
        if isinstance(canvas, matplotlib.axes.Axes):
            axes[(1, 1)] = canvas
        else:
            sp_typ = subplot_specs[0, 0]["type"]
            axes[(1, 1)] = fig.add_subplot(
                111, projection="3d" if sp_typ == "scene" else None
            )
    else:
        max_rows = max_rows if max_rows is not None else 1
        max_cols = max_cols if max_cols is not None else 1
        count = 0
        for row in range(1, max_rows + 1):
            for col in range(1, max_cols + 1):
                subplot_found = True
                count += 1
                row_col_num = (row, col)
                projection = (
                    "3d" if subplot_specs[row - 1, col - 1]["type"] == "scene" else None
                )
                if isinstance(canvas, matplotlib.figure.Figure):
                    try:
                        axes[row_col_num] = extract_axis_from_row_col(fig, row, col)
                    except (ValueError, IndexError):  # IndexError if axis is not found
                        subplot_found = False
                if canvas is None or not subplot_found:
                    axes[row_col_num] = fig.add_subplot(
                        max_rows, max_cols, count, projection=projection
                    )
                if axes[row_col_num].name == "3d":
                    axes[row_col_num].set_box_aspect((1, 1, 1))

    def draw_frame(frame_ind):
        count_with_labels = {}
        handler_map = {}
        for tr in frames[frame_ind]["data"]:
            row_col_num = (tr["row"], tr["col"])
            ax = axes[row_col_num]
            constructor = tr["constructor"]
            args = tr.get("args", ())
            kwargs = tr.get("kwargs", {})
            if frame_ind == 0:
                if row_col_num not in count_with_labels:
                    count_with_labels[row_col_num] = 0
                label = kwargs.get("label", "_")
                if label and not label.startswith("_"):
                    count_with_labels[row_col_num] += 1
            trace = getattr(ax, constructor)(*args, **kwargs)
            if "legend_handler" in tr:
                handler_map[trace] = tr["legend_handler"]
            if constructor == "plot_trisurf":
                # 'Poly3DCollection' object has no attribute '_edgecolors2d'
                for arg in ("face", "edge"):
                    color = getattr(trace, f"_{arg}color3d", None)
                    color = (  # for mpl version <3.3.3
                        getattr(trace, f"_{arg}colors3d", None)
                        if color is None
                        else color
                    )
                    setattr(trace, f"_{arg}colors2d", color)
        for row_col_num, ax in axes.items():
            count = count_with_labels.get(row_col_num, 0)
            if ax.name == "3d":
                if row_col_num in ranges and canvas_update:
                    ax.set(
                        **{f"{k}label": labels[row_col_num][k] for k in "xyz"},
                        **{f"{k}lim": r for k, r in zip("xyz", ranges[row_col_num])},
                    )
                ax.set_box_aspect(aspect=(1, 1, 1))
                if 0 < count <= legend_maxitems:
                    lg_kw = {"bbox_to_anchor": (1.04, 1), "loc": "upper left"}
                    if handler_map:
                        lg_kw["handler_map"] = handler_map
                    try:
                        ax.legend(**lg_kw)
                    except AttributeError:
                        # see https://github.com/matplotlib/matplotlib/pull/25565
                        pass
            else:
                ax.legend(loc="best")

    def animate(ind):  # pragma: no cover
        for ax in axes.values():
            ax.clear()
        draw_frame(ind)
        return list(axes.values())

    anim = None
    if len(frames) == 1:
        draw_frame(0)
    else:
        anim = FuncAnimation(
            fig,
            animate,
            frames=range(len(frames)),
            interval=data["frame_duration"],
            blit=False,
            repeat=repeat,
        )

    out = ()
    if return_fig:
        show_canvas = False
        out += (fig,)
    if return_animation and len(frames) != 1:
        show_canvas = False
        out += (anim,)
    if show_canvas:
        plt.show(**show_kwargs)

    if out:
        return out[0] if len(out) == 1 else out

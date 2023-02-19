"""matplotlib backend"""
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from magpylib._src.display.traces_utility import place_and_orient_model3d
from magpylib._src.display.traces_utility import subdivide_mesh_by_facecolor

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


def generic_trace_to_matplotlib(trace):
    """Transform a generic trace into a matplotlib trace"""
    traces_mpl = []
    leg_title = trace.get("legendgrouptitle_text", None)
    if trace["type"] == "mesh3d":
        subtraces = [trace]
        if trace.get("facecolor", None) is not None:
            subtraces = subdivide_mesh_by_facecolor(trace)
        for ind, subtrace in enumerate(subtraces):
            x, y, z = np.array([subtrace[k] for k in "xyz"], dtype=float)
            triangles = np.array([subtrace[k] for k in "ijk"]).T
            tr = {
                "constructor": "plot_trisurf",
                "args": (x, y, z),
                "kwargs": {
                    "triangles": triangles,
                    "alpha": subtrace.get("opacity", None),
                    "color": subtrace.get("color", None),
                },
            }
            if ind != 0:  # hide substrace legends except first
                tr["kwargs"]["label"] = "_no_legend_"
            traces_mpl.append(tr)
    elif "scatter" in trace["type"]:
        props = {
            k: trace.get(v[0], {}).get(v[1], trace.get("_".join(v), None))
            for k, v in {
                "ls": ("line", "dash"),
                "lw": ("line", "width"),
                "color": ("line", "color"),
                "marker": ("marker", "symbol"),
                "mfc": ("marker", "color"),
                "mec": ("marker", "color"),
                "ms": ("marker", "size"),
            }.items()
        }
        coords_str = "xyz"
        if trace["type"] == "scatter":
            coords_str = "xy"
        coords = np.array([trace[k] for k in coords_str], dtype=float)
        if isinstance(props["ms"], (list, tuple, np.ndarray)):
            uniq = np.unique(props["ms"])
            if uniq.shape[0] == 1:
                props["ms"] = props["ms"][0]
                props["label"] = None
            else:
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
        if mode is not None:
            if "lines" not in mode:
                props["ls"] = ""
            if "markers" not in mode:
                props["marker"] = None
            if "text" in mode and trace.get("text", False):
                for *coords_s, txt in zip(*coords, trace["text"]):
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
    showlegend = trace.get("showlegend", True)
    for tr in traces_mpl:
        tr["row"] = trace.get("row", 1)
        tr["col"] = trace.get("col", 1)
        tr["kwargs"] = tr.get("kwargs", {})
        if showlegend:
            if "label" not in tr["kwargs"]:
                tr["kwargs"]["label"] = trace.get("name", "")
                if leg_title is not None:
                    tr["kwargs"]["label"] += f" ({leg_title})"
        else:
            tr["kwargs"]["label"] = "_no_legend"
    return traces_mpl


def process_extra_trace(model):
    "process extra trace attached to some magpylib object"
    extr = model["model3d"]
    model_kwargs = {"color": model["kwargs"]["color"]}
    model_kwargs.update(extr.kwargs() if callable(extr.kwargs) else extr.kwargs)
    model_args = extr.args() if callable(extr.args) else extr.args
    trace3d = {
        "constructor": extr.constructor,
        "kwargs": model_kwargs,
        "args": model_args,
        "row": model["kwargs"]["row"],
        "col": model["kwargs"]["col"],
    }
    kwargs, args, = place_and_orient_model3d(
        model_kwargs=model_kwargs,
        model_args=model_args,
        orientation=model["orientation"],
        position=model["position"],
        coordsargs=extr.coordsargs,
        scale=extr.scale,
        return_model_args=True,
    )
    trace3d["kwargs"].update(kwargs)
    trace3d["args"] = args
    return trace3d


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


def display_matplotlib(
    data,
    canvas=None,
    repeat=False,
    return_fig=False,
    return_animation=False,
    max_rows=None,
    max_cols=None,
    subplot_specs=None,
    dpi=80,
    figsize=None,
    legend_max_items=10,
    **kwargs,  # pylint: disable=unused-argument
):

    """Display objects and paths graphically using the matplotlib library."""
    frames = data["frames"]
    ranges = data["ranges"]

    for fr in frames:
        new_data = []
        for tr in fr["data"]:
            new_data.extend(generic_trace_to_matplotlib(tr))
        for model in fr["extra_backend_traces"]:
            new_data.append(process_extra_trace(model))
        fr["data"] = new_data

    show_canvas = False
    axes = {}
    if canvas is None:
        show_canvas = True
        if figsize is None:
            figsize = (8, 8)
            ratio = subplot_specs.shape[1] / subplot_specs.shape[0]
            figsize = (figsize[0] * ratio, figsize[1])
        fig = plt.figure(dpi=dpi, figsize=figsize)
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
        raise ValueError(
            "The `canvas` parameter must be one of `[None, matplotlib.axes.Axes, "
            f"matplotlib.figure.Figure]`. Received type {type(canvas)!r} instead"
        )
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
                count += 1
                row_col_num = (row, col)
                if canvas is None:
                    projection = (
                        "3d"
                        if subplot_specs[row - 1, col - 1]["type"] == "scene"
                        else None
                    )
                    axes[row_col_num] = fig.add_subplot(
                        max_rows, max_cols, count, projection=projection
                    )
                elif isinstance(canvas, matplotlib.figure.Figure):
                    axes[row_col_num] = extract_axis_from_row_col(fig, row, col)
                if axes[row_col_num].name == "3d":
                    axes[row_col_num].set_box_aspect((1, 1, 1))

    def draw_frame(ind):
        count_with_labels = 0
        for tr in frames[ind]["data"]:
            row_col_num = (tr["row"], tr["col"])
            ax = axes[row_col_num]
            constructor = tr["constructor"]
            args = tr.get("args", ())
            kwargs = tr.get("kwargs", {})
            label = kwargs.get("label", "_")
            if label and not label.startswith("_"):
                count_with_labels += 1
            trace = getattr(ax, constructor)(*args, **kwargs)
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
        for ax in axes.values():
            if ax.name == "3d":
                ax.set(
                    **{f"{k}label": f"{k} [mm]" for k in "xyz"},
                    **{f"{k}lim": r for k, r in zip("xyz", ranges)},
                )
                if 0 < count_with_labels <= legend_max_items:
                    ax.legend(bbox_to_anchor=(0.5, 1.1), loc="upper left")
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
    plt.tight_layout()
    out = ()
    if return_fig:
        show_canvas = False
        out += (fig,)
    if return_animation and len(frames) != 1:
        show_canvas = False
        out += (anim,)
    if show_canvas:
        plt.show()

    if out:
        return out[0] if len(out) == 1 else out

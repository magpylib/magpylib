"""plotly backend"""

# pylint: disable=C0302
# pylint: disable=too-many-branches
# pylint: disable=too-many-positional-arguments

import inspect
from functools import lru_cache

import numpy as np

try:
    import plotly.graph_objects as go
except ImportError as missing_module:  # pragma: no cover
    raise ModuleNotFoundError(
        """In order to use the plotly plotting backend, you need to install plotly via pip or conda,
        see https://github.com/plotly/plotly.py"""
    ) from missing_module

from magpylib._src.defaults.defaults_utility import linearize_dict
from magpylib._src.display.traces_utility import get_scene_ranges

SYMBOLS_TO_PLOTLY = {
    ".": "circle",
    "o": "circle",
    "+": "cross",
    "D": "diamond",
    "d": "diamond",
    "s": "square",
    "x": "x",
}

LINESTYLES_TO_PLOTLY = {
    "solid": "solid",
    "-": "solid",
    "dashed": "dash",
    "--": "dash",
    "dashdot": "dashdot",
    "-.": "dashdot",
    "dotted": "dot",
    ".": "dot",
    ":": "dot",
    (0, (1, 1)): "dot",
    "loosely dotted": "longdash",
    "loosely dashdotted": "longdashdot",
}

SIZE_FACTORS_TO_PLOTLY = {
    "line_width": 2.2,
    "marker_size": 0.7,
}


@lru_cache(maxsize=None)  # Cache all results
def match_args(ttype: str):
    """
    Return named arguments of a Plotly graph object function.

    Parameters
    ----------
    ttype : str
        Type of Plotly graph object (e.g., 'scatter' for go.Scatter).

    Returns
    -------
    set
        Names of the named arguments of the specified function.
    """
    func = getattr(go, ttype.capitalize())
    sig = inspect.signature(func)
    params = sig.parameters
    named_args = [
        name
        for name, param in params.items()
        if param.default != inspect.Parameter.empty
    ]
    return set(named_args)


def apply_fig_ranges(fig, ranges_rc, labels_rc, apply2d=True):
    """This is a helper function which applies the ranges properties of the provided `fig` object
    according to a provided ranges for each subplot. All three space direction will be equal and
    match the maximum of the ranges needed to display all objects, including their paths.

    Parameters
    ----------
    ranges_rc: dict of arrays of dimension=(3,2)
        min and max graph range
    labels_rc: dict of dicts
        contains a dict with 'x', 'y', 'z' keys and respective labels as strings for each subplot

    apply2d: bool, default = True
        applies fixed range also on 2d traces

    Returns
    -------
    None: NoneType
    """
    for rc, ranges in ranges_rc.items():
        row, col = rc
        labels = labels_rc.get(rc, {k: "" for k in "xyz"})
        kwargs = {
            **{
                f"{k}axis": {
                    "range": ranges[i],
                    "autorange": False,
                    "title": labels[k],
                }
                for i, k in enumerate("xyz")
            },
            "aspectratio": {k: 1 for k in "xyz"},
            "aspectmode": "manual",
            "camera_eye": {"x": 1, "y": -1.5, "z": 1.4},
        }

        # pylint: disable=protected-access
        if fig._grid_ref is not None:
            kwargs.update({"row": row, "col": col})
        fig.update_scenes(**kwargs)
    if apply2d:
        apply_2d_ranges(fig)


def apply_2d_ranges(fig, factor=0.05):
    """Apply Figure ranges of 2d plots"""
    traces = fig.data
    ranges = {}
    for t in traces:
        for k in "xy":
            try:
                ax_str = getattr(t, f"{k}axis")
                ax_suff = ax_str.replace(k, "")
                if ax_suff not in ranges:
                    ranges[ax_suff] = {"x": [], "y": []}
                vals = getattr(t, k)
                ranges[ax_suff][k].append([min(vals), max(vals)])
            except AttributeError:
                pass
    for ax, r in ranges.items():
        for k in "xy":
            m, M = [np.min(r[k]), np.max(r[k])]
            getattr(fig.layout, f"{k}axis{ax}").range = [
                m - (M - m) * factor,
                M + (M - m) * factor,
            ]


def animate_path(
    fig,
    frames,
    path_indices,
    frame_duration,
    animation_slider=False,
    update_layout=True,
    rows=None,
    cols=None,
):
    """This is a helper function which attaches plotly frames to the provided `fig` object
    according to a certain zoom level. All three space direction will be equal and match the
    maximum of the ranges needed to display all objects, including their paths.
    """
    fps = int(1000 / frame_duration)

    play_dict = {
        "args": [
            None,
            {
                "frame": {"duration": frame_duration},
                "transition": {"duration": 0},
                "fromcurrent": True,
            },
        ],
        "label": "Play",
        "method": "animate",
    }
    pause_dict = {
        "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
        "label": "Pause",
        "method": "animate",
    }
    buttons_dict = {
        "buttons": [play_dict, pause_dict],
        "direction": "left",
        "pad": {"r": 10, "t": 20},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top",
    }

    if animation_slider:
        sliders_dict = {
            "active": 0,
            "yanchor": "top",
            "font": {"size": 10},
            "xanchor": "left",
            "currentvalue": {
                "prefix": f"Fps={fps}, Path index: ",
                "visible": True,
                "xanchor": "right",
            },
            "pad": {"b": 10, "t": 10},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [],
        }
        for ind in path_indices:
            slider_step = {
                "args": [
                    [str(ind + 1)],
                    {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                ],
                "label": str(ind + 1),
                "method": "animate",
            }
            sliders_dict["steps"].append(slider_step)

    # update fig
    fig.frames = frames
    frame0 = fig.frames[0]
    title = frame0.layout.title.text
    fig.add_traces(frame0.data, rows=rows, cols=cols)
    if update_layout:
        fig.update_layout(
            height=None,
            title=title,
        )
    sliders = [sliders_dict] if animation_slider else None
    fig.update_layout(
        updatemenus=[*fig.layout.updatemenus, buttons_dict],
        sliders=[*fig.layout.sliders, *sliders],
    )


def generic_trace_to_plotly(trace):
    """Transform a generic trace into a plotly trace"""
    if "scatter" in trace["type"]:
        if trace.get("line_width", None):
            trace["line_width"] *= SIZE_FACTORS_TO_PLOTLY["line_width"]
        dash = trace.get("line_dash", None)
        if dash is not None:
            trace["line_dash"] = LINESTYLES_TO_PLOTLY.get(dash, "solid")
        symb = trace.get("marker_symbol", None)
        if symb is not None:
            trace["marker_symbol"] = SYMBOLS_TO_PLOTLY.get(symb, "circle")
        if "marker_size" in trace:
            trace["marker_size"] = (
                np.array(trace["marker_size"], dtype="float")
                * SIZE_FACTORS_TO_PLOTLY["marker_size"]
            )
    return trace


def process_extra_trace(model):
    "process extra trace attached to some magpylib object"
    ttype = model["constructor"].lower()
    kwargs = {**model["kwargs_extra"], **model["kwargs"]}
    trace3d = {"type": ttype, **kwargs}
    if ttype == "scatter3d":
        for k in ("marker", "line"):
            trace3d[f"{k}_color"] = trace3d.get(f"{k}_color", kwargs["color"])
    elif ttype == "mesh3d":
        trace3d["showscale"] = trace3d.get("showscale", False)
        trace3d["color"] = trace3d.get("color", kwargs["color"])
    # need to match parameters to the constructor
    # the color parameter is added by default  int the `traces_generic` module but is not
    # compatible with some traces (e.g. `go.Surface`)
    allowed_prefs = match_args(ttype)
    trace3d = {
        k: v
        for k, v in trace3d.items()
        if k in {"row", "col", "type"}
        or k in allowed_prefs
        or ("_" in k and k.split("_")[0] in allowed_prefs)
    }
    trace3d.update(linearize_dict(trace3d, separator="_"))
    return trace3d


def display_plotly(
    data,
    canvas=None,
    renderer=None,
    return_fig=False,
    canvas_update="auto",
    max_rows=None,
    max_cols=None,
    subplot_specs=None,
    fig_kwargs=None,
    show_kwargs=None,
    **kwargs,  # pylint: disable=unused-argument
):
    """Display objects and paths graphically using the plotly library."""

    fig_kwargs = {} if not fig_kwargs else fig_kwargs
    show_kwargs = {} if not show_kwargs else show_kwargs
    show_kwargs = {"renderer": renderer, **show_kwargs}

    # only update layout if canvas is not provided
    fig = canvas
    show_fig = False
    extra_data = False
    if fig is None:
        if not return_fig:
            show_fig = True
        fig = go.Figure()

    if not (max_rows is None and max_cols is None):
        # pylint: disable=protected-access
        if fig._grid_ref is None:
            fig = fig.set_subplots(
                rows=max_rows,
                cols=max_cols,
                specs=subplot_specs.tolist(),
            )

    frames = data["frames"]
    for fr in frames:
        new_data = []
        for tr in fr["data"]:
            new_data.append(generic_trace_to_plotly(tr))
        for model in fr["extra_backend_traces"]:
            extra_data = True
            new_data.append(process_extra_trace(model))
        fr["data"] = new_data
        fr.pop("extra_backend_traces", None)
    with fig.batch_update():
        for frame in frames:
            rows_list = []
            cols_list = []
            for tr in frame["data"]:
                row = tr.pop("row", None)
                col = tr.pop("col", None)
                rows_list.append(row)
                cols_list.append(col)
        if max_rows is None and max_cols is None:
            rows_list = cols_list = None
        isanimation = len(frames) != 1
        if not isanimation:
            fig.add_traces(frames[0]["data"], rows=rows_list, cols=cols_list)
        else:
            animation_slider = data["input_kwargs"].get("animation_slider", False)
            animate_path(
                fig,
                frames,
                data["path_indices"],
                data["frame_duration"],
                animation_slider=animation_slider,
                update_layout=canvas_update,
                rows=rows_list,
                cols=cols_list,
            )
        if canvas_update:
            ranges_rc = data["ranges"]
            if extra_data:
                ranges_rc = get_scene_ranges(*frames[0]["data"])
            apply_fig_ranges(
                fig, ranges_rc, labels_rc=data["labels"], apply2d=isanimation
            )
            fig.update_layout(
                legend_itemsizing="constant",
                # legend_groupclick="toggleitem",
            )
        fig.update(fig_kwargs)

    if return_fig and not show_fig:
        return fig
    if show_fig:
        fig.show(**show_kwargs)
    return None

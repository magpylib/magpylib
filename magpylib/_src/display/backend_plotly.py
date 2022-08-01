"""plotly backend"""
# pylint: disable=C0302
# pylint: disable=too-many-branches

try:
    import plotly.graph_objects as go
except ImportError as missing_module:  # pragma: no cover
    raise ModuleNotFoundError(
        """In order to use the plotly plotting backend, you need to install plotly via pip or conda,
        see https://github.com/plotly/plotly.py"""
    ) from missing_module

from magpylib._src.defaults.defaults_classes import default_settings as Config
from magpylib._src.display.traces_generic import get_frames
from magpylib._src.defaults.defaults_utility import linearize_dict
from magpylib._src.display.traces_utility import place_and_orient_model3d
from magpylib._src.display.traces_utility import get_scene_ranges
from magpylib._src.defaults.defaults_utility import SIZE_FACTORS_MATPLOTLIB_TO_PLOTLY
from magpylib._src.style import LINESTYLES_MATPLOTLIB_TO_PLOTLY
from magpylib._src.style import SYMBOLS_MATPLOTLIB_TO_PLOTLY


def apply_fig_ranges(fig, ranges):
    """This is a helper function which applies the ranges properties of the provided `fig` object
    according to a certain zoom level. All three space direction will be equal and match the
    maximum of the ranges needed to display all objects, including their paths.

    Parameters
    ----------
    ranges: array of dimension=(3,2)
        min and max graph range

    zoom: float, default = 1
        When zoom=0 all objects are just inside the 3D-axes.

    Returns
    -------
    None: NoneType
    """
    fig.update_scenes(
        **{
            f"{k}axis": dict(range=ranges[i], autorange=False, title=f"{k} [mm]")
            for i, k in enumerate("xyz")
        },
        aspectratio={k: 1 for k in "xyz"},
        aspectmode="manual",
        camera_eye={"x": 1, "y": -1.5, "z": 1.4},
    )


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

    buttons_dict = {
        "buttons": [
            {
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
            },
            {
                "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                "label": "Pause",
                "method": "animate",
            },
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 20},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top",
    }

    for ind in path_indices:
        if animation_slider:
            slider_step = {
                "args": [
                    [str(ind + 1)],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                    },
                ],
                "label": str(ind + 1),
                "method": "animate",
            }
            sliders_dict["steps"].append(slider_step)

    # update fig
    fig.frames = frames
    frame0 = fig.frames[0]
    fig.add_traces(
        frame0.data,
        rows=rows,
        cols=cols,
    )
    title = frame0.layout.title.text
    if update_layout:
        fig.update_layout(
            height=None,
            title=title,
        )
    fig.update_layout(
        updatemenus=[buttons_dict],
        sliders=[sliders_dict] if animation_slider else None,
    )


def generic_trace_to_plotly(trace):
    """Transform a generic trace into a plotly trace"""
    if trace["type"] == "scatter3d":
        if "line_width" in trace:
            trace["line_width"] *= SIZE_FACTORS_MATPLOTLIB_TO_PLOTLY["line_width"]
        dash = trace.get("line_dash", None)
        if dash is not None:
            trace["line_dash"] = LINESTYLES_MATPLOTLIB_TO_PLOTLY.get(dash, dash)
        symb = trace.get("marker_symbol", None)
        if symb is not None:
            trace["marker_symbol"] = SYMBOLS_MATPLOTLIB_TO_PLOTLY.get(symb, symb)
        if "marker_size" in trace:
            trace["marker_size"] *= SIZE_FACTORS_MATPLOTLIB_TO_PLOTLY["marker_size"]
    return trace


def process_extra_trace(model):
    "process extra trace attached to some magpylib object"
    extr = model["model3d"]
    kwargs = model["kwargs"]
    trace3d = {**kwargs}
    ttype = extr.constructor.lower()
    trace_kwargs = extr.kwargs() if callable(extr.kwargs) else extr.kwargs
    trace3d.update({"type": ttype, **trace_kwargs})
    if ttype == "scatter3d":
        for k in ("marker", "line"):
            trace3d[f"{k}_color"] = trace3d.get(f"{k}_color", kwargs["color"])
            trace3d.pop("color", None)
    elif ttype == "mesh3d":
        trace3d["showscale"] = trace3d.get("showscale", False)
        trace3d["color"] = trace3d.get("color", kwargs["color"])
    trace3d.update(
        linearize_dict(
            place_and_orient_model3d(
                model_kwargs=trace3d,
                orientation=model["orientation"],
                position=model["position"],
                scale=extr.scale,
            ),
            separator="_",
        )
    )
    return trace3d


def extract_layout_kwargs(kwargs):
    """Extract layout kwargs"""
    layout = kwargs.pop("layout", {})
    layout_kwargs = {k[7:]: v for k, v in kwargs.items() if k.startswith("layout")}
    kwargs = {k: v for k, v in kwargs.items() if not k.startswith("layout")}
    layout.update(layout_kwargs)
    return layout, kwargs


def display_plotly(
    *obj_list,
    zoom=1,
    canvas=None,
    renderer=None,
    animation=False,
    colorsequence=None,
    return_fig=False,
    update_layout=True,
    max_rows=None,
    max_cols=None,
    subplot_specs=None,
    **kwargs,
):

    """Display objects and paths graphically using the plotly library."""

    fig = canvas
    show_fig = False
    extra_data = False
    if fig is None:
        if not return_fig:
            show_fig = True
        if max_rows is None and max_cols is None:
            fig = go.Figure()
        else:
            fig = go.Figure().set_subplots(
                rows=max_rows,
                cols=max_cols,
                specs=subplot_specs.tolist(),
            )

    if colorsequence is None:
        colorsequence = Config.display.colorsequence

    layout, kwargs = extract_layout_kwargs(kwargs)
    data = get_frames(
        objs=obj_list,
        colorsequence=colorsequence,
        zoom=zoom,
        animation=animation,
        extra_backend="plotly",
        **kwargs,
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
        if len(frames) == 1:
            fig.add_traces(frames[0]["data"], rows=rows_list, cols=cols_list)
        else:
            animation_slider = data.get("animation_slider", False)
            animate_path(
                fig,
                frames,
                data["path_indices"],
                data["frame_duration"],
                animation_slider=animation_slider,
                update_layout=update_layout,
                rows=rows_list,
                cols=cols_list,
            )
        ranges = data["ranges"]
        if extra_data:
            ranges = get_scene_ranges(*frames[0]["data"], zoom=zoom)
        if update_layout:
            apply_fig_ranges(fig, ranges)
            fig.update_layout(legend_itemsizing="constant")
        fig.update_layout(layout)

    if return_fig and not show_fig:
        return fig
    if show_fig:
        fig.show(renderer=renderer)
    return None

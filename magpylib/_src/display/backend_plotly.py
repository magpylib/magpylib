""" plotly draw-functionalities"""
# pylint: disable=C0302
# pylint: disable=too-many-branches
import numbers
import warnings

try:
    import plotly.graph_objects as go
except ImportError as missing_module:  # pragma: no cover
    raise ModuleNotFoundError(
        """In order to use the plotly plotting backend, you need to install plotly via pip or conda,
        see https://github.com/plotly/plotly.py"""
    ) from missing_module

import numpy as np
from magpylib._src.defaults.defaults_classes import default_settings as Config
from magpylib._src.utility import format_obj_input
from magpylib._src.display.traces_utility import clean_legendgroups
from magpylib._src.display.traces_generic import (
    draw_frame,
    apply_fig_ranges,
    MagpyMarkers,
)
from magpylib._src.defaults.defaults_utility import SIZE_FACTORS_MATPLOTLIB_TO_PLOTLY
from magpylib._src.style import LINESTYLES_MATPLOTLIB_TO_PLOTLY
from magpylib._src.style import SYMBOLS_MATPLOTLIB_TO_PLOTLY


def animate_path(
    fig,
    objs,
    colorsequence=None,
    zoom=1,
    title="3D-Paths Animation",
    animation_time=3,
    animation_fps=30,
    animation_maxfps=50,
    animation_maxframes=200,
    animation_slider=False,
    **kwargs,
):
    """This is a helper function which attaches plotly frames to the provided `fig` object
    according to a certain zoom level. All three space direction will be equal and match the
    maximum of the ranges needed to display all objects, including their paths.

    Parameters
    ----------
    animation_time: float, default = 3
        Sets the animation duration

    animation_fps: float, default = 30
        This sets the maximum allowed frame rate. In case of path positions needed to be displayed
        exceeds the `animation_fps` the path position will be downsampled to be lower or equal
        the `animation_fps`. This is mainly depending on the pc/browser performance and is set to
        50 by default to avoid hanging the animation process.

    animation_slider: bool, default = False
        if True, an interactive slider will be displayed and stay in sync with the animation

    title: str, default = "3D-Paths Animation"
        When zoom=0 all objects are just inside the 3D-axes.

    colorsequence: list or array_like, iterable, default=
            ['#2E91E5', '#E15F99', '#1CA71C', '#FB0D0D', '#DA16FF', '#222A2A',
            '#B68100', '#750D86', '#EB663B', '#511CFB', '#00A08B', '#FB00D1',
            '#FC0080', '#B2828D', '#6C7C32', '#778AAE', '#862A16', '#A777F1',
            '#620042', '#1616A7', '#DA60CA', '#6C4516', '#0D2A63', '#AF0038']
        An iterable of color values used to cycle trough for every object displayed.
        A color and may be specified as:
      - A hex string (e.g. '#ff0000')
      - An rgb/rgba string (e.g. 'rgb(255,0,0)')
      - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
      - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
      - A named CSS color

    Returns
    -------
    None: NoneTyp
    """
    # make sure the number of frames does not exceed the max frames and max frame rate
    # downsample if necessary
    path_lengths = []
    for obj in objs:
        subobjs = [obj]
        if getattr(obj, "_object_type", None) == "Collection":
            subobjs.extend(obj.children)
        for subobj in subobjs:
            path_len = getattr(subobj, "_position", np.array((0.0, 0.0, 0.0))).shape[0]
            path_lengths.append(path_len)

    max_pl = max(path_lengths)
    if animation_fps > animation_maxfps:
        warnings.warn(
            f"The set `animation_fps` at {animation_fps} is greater than the max allowed of"
            f" {animation_maxfps}. `animation_fps` will be set to {animation_maxfps}. "
            f"You can modify the default value by setting it in "
            "`magpylib.defaults.display.animation.maxfps`"
        )
        animation_fps = animation_maxfps

    maxpos = min(animation_time * animation_fps, animation_maxframes)

    if max_pl <= maxpos:
        path_indices = np.arange(max_pl)
    else:
        round_step = max_pl / (maxpos - 1)
        ar = np.linspace(0, max_pl, max_pl, endpoint=False)
        path_indices = np.unique(np.floor(ar / round_step) * round_step).astype(
            int
        )  # downsampled indices
        path_indices[-1] = (
            max_pl - 1
        )  # make sure the last frame is the last path position

    # calculate exponent of last frame index to avoid digit shift in
    # frame number display during animation
    exp = (
        np.log10(path_indices.max()).astype(int) + 1
        if path_indices.ndim != 0 and path_indices.max() > 0
        else 1
    )

    frame_duration = int(animation_time * 1000 / path_indices.shape[0])
    new_fps = int(1000 / frame_duration)
    if max_pl > animation_maxframes:
        warnings.warn(
            f"The number of frames ({max_pl}) is greater than the max allowed "
            f"of {animation_maxframes}. The `animation_fps` will be set to {new_fps}. "
            f"You can modify the default value by setting it in "
            "`magpylib.defaults.display.animation.maxframes`"
        )

    if animation_slider:
        sliders_dict = {
            "active": 0,
            "yanchor": "top",
            "font": {"size": 10},
            "xanchor": "left",
            "currentvalue": {
                "prefix": f"Fps={new_fps}, Path index: ",
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

    # create frame for each path index or downsampled path index
    frames = []
    autosize = "return"
    for i, ind in enumerate(path_indices):
        kwargs["style_path_frames"] = [ind]
        frame = draw_frame(
            objs,
            colorsequence,
            zoom,
            autosize=autosize,
            output="list",
            **kwargs,
        )
        if i == 0:  # get the dipoles and sensors autosize from first frame
            traces, autosize = frame
        else:
            traces = frame
        frames.append(
            go.Frame(
                data=[generic_trace_to_plotly(trace) for trace in traces],
                name=str(ind + 1),
                layout=dict(title=f"""{title} - path index: {ind+1:0{exp}d}"""),
            )
        )
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
    fig.add_traces(frames[0].data)
    fig.update_layout(
        height=None,
        title=title,
        updatemenus=[buttons_dict],
        sliders=[sliders_dict] if animation_slider else None,
    )
    apply_fig_ranges(fig, zoom=zoom)


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


def display_plotly(
    *obj_list,
    markers=None,
    zoom=1,
    canvas=None,
    renderer=None,
    animation=False,
    colorsequence=None,
    **kwargs,
):

    """
    Display objects and paths graphically using the plotly library.

    Parameters
    ----------
    objects: sources, collections or sensors
        Objects to be displayed.

    markers: array_like, None, shape (N,3), default=None
        Display position markers in the global CS. By default no marker is displayed.

    zoom: float, default = 1
        Adjust plot zoom-level. When zoom=0 all objects are just inside the 3D-axes.

    fig: plotly Figure, default=None
        Display graphical output in a given figure:
        - plotly.graph_objects.Figure
        - plotly.graph_objects.FigureWidget
        By default a new `Figure` is created and displayed.

    renderer: str. default=None,
        The renderers framework is a flexible approach for displaying plotly.py figures in a variety
        of contexts.
        Available renderers are:
        ['plotly_mimetype', 'jupyterlab', 'nteract', 'vscode',
         'notebook', 'notebook_connected', 'kaggle', 'azure', 'colab',
         'cocalc', 'databricks', 'json', 'png', 'jpeg', 'jpg', 'svg',
         'pdf', 'browser', 'firefox', 'chrome', 'chromium', 'iframe',
         'iframe_connected', 'sphinx_gallery', 'sphinx_gallery_png']

    title: str, default = "3D-Paths Animation"
        When zoom=0 all objects are just inside the 3D-axes.

    colorsequence: list or array_like, iterable, default=
            ['#2E91E5', '#E15F99', '#1CA71C', '#FB0D0D', '#DA16FF', '#222A2A',
            '#B68100', '#750D86', '#EB663B', '#511CFB', '#00A08B', '#FB00D1',
            '#FC0080', '#B2828D', '#6C7C32', '#778AAE', '#862A16', '#A777F1',
            '#620042', '#1616A7', '#DA60CA', '#6C4516', '#0D2A63', '#AF0038']
        An iterable of color values used to cycle trough for every object displayed.
        A color and may be specified as:
      - A hex string (e.g. '#ff0000')
      - An rgb/rgba string (e.g. 'rgb(255,0,0)')
      - An hsl/hsla string (e.g. 'hsl(0,100%,50%)')
      - An hsv/hsva string (e.g. 'hsv(0,100%,100%)')
      - A named CSS color

    Returns
    -------
    None: NoneType
    """

    flat_obj_list = format_obj_input(obj_list)

    show_canvas = False
    if canvas is None:
        show_canvas = True
        canvas = go.Figure()

    # set animation and animation_time
    if isinstance(animation, numbers.Number) and not isinstance(animation, bool):
        kwargs["animation_time"] = animation
        animation = True
    if (
        not any(
            getattr(obj, "position", np.array([])).ndim > 1 for obj in flat_obj_list
        )
        and animation is not False
    ):  # check if some path exist for any object
        animation = False
        warnings.warn("No path to be animated detected, displaying standard plot")

    animation_kwargs = {
        k: v for k, v in kwargs.items() if k.split("_")[0] == "animation"
    }
    if animation is False:
        kwargs = {k: v for k, v in kwargs.items() if k not in animation_kwargs}
    else:
        for k, v in Config.display.animation.as_dict().items():
            anim_key = f"animation_{k}"
            if kwargs.get(anim_key, None) is None:
                kwargs[anim_key] = v

    if obj_list:
        style = getattr(obj_list[0], "style", None)
        label = getattr(style, "label", None)
        title = label if len(obj_list) == 1 else None
    else:
        title = "No objects to be displayed"

    if markers is not None and markers:
        obj_list = list(obj_list) + [MagpyMarkers(*markers)]

    if colorsequence is None:
        colorsequence = Config.display.colorsequence

    with canvas.batch_update():
        if animation is not False:
            title = "3D-Paths Animation" if title is None else title
            animate_path(
                fig=canvas,
                objs=obj_list,
                colorsequence=colorsequence,
                zoom=zoom,
                title=title,
                **kwargs,
            )
        else:
            generic_traces = draw_frame(
                obj_list, colorsequence, zoom, output="list", **kwargs
            )
            traces = [generic_trace_to_plotly(trace) for trace in generic_traces]
            canvas.add_traces(traces)
            canvas.update_layout(title_text=title)
            apply_fig_ranges(canvas, zoom=zoom)
        clean_legendgroups(canvas)
        canvas.update_layout(legend_itemsizing="constant")
    if show_canvas:
        canvas.show(renderer=renderer)

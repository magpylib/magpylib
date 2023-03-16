import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from magpylib._src.display.traces_generic import get_frames
from magpylib._src.display.traces_utility import place_and_orient_model3d
from magpylib._src.display.traces_utility import subdivide_mesh_by_facecolor

# from magpylib._src.utility import format_obj_input

SYMBOLS = {"circle": "o", "cross": "+", "diamond": "d", "square": "s", "x": "x"}

LINE_STYLES = {
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
    if trace["type"] == "mesh3d":
        subtraces = [trace]
        if trace.get("facecolor", None) is not None:
            subtraces = subdivide_mesh_by_facecolor(trace)
        for subtrace in subtraces:
            x, y, z = np.array([subtrace[k] for k in "xyz"], dtype=float)
            triangles = np.array([subtrace[k] for k in "ijk"]).T
            kwargs = {
                "triangles": triangles,
                "alpha": subtrace.get("opacity", None),
                "color": subtrace.get("color", None),
            }
            if not trace.get("flatshading", False):
                # flatshading is on for triangular meshes, to see triangles edges better
                # in other cases we don't want to see them
                kwargs.update(linewidth=0, antialiased=False)
            traces_mpl.append(
                {"constructor": "plot_trisurf", "args": (x, y, z), "kwargs": kwargs}
            )

    elif trace["type"] == "scatter3d":
        x, y, z = np.array([trace[k] for k in "xyz"], dtype=float)
        mode = trace.get("mode", None)
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
        if "ls" in props:
            props["ls"] = LINE_STYLES.get(props["ls"], props["ls"])
        if "marker" in props:
            props["marker"] = SYMBOLS.get(props["marker"], props["marker"])
        if mode is not None:
            if "lines" not in mode:
                props["ls"] = ""
            if "markers" not in mode:
                props["marker"] = None
            if "text" in mode and trace.get("text", False):
                for xs, ys, zs, txt in zip(x, y, z, trace["text"]):
                    traces_mpl.append(
                        {
                            "constructor": "text",
                            "args": (xs, ys, zs, txt),
                        }
                    )
        traces_mpl.append(
            {
                "constructor": "plot",
                "args": (x, y, z),
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


def display_matplotlib(
    *obj_list,
    zoom=1,
    canvas=None,
    animation=False,
    repeat=False,
    colorsequence=None,
    return_fig=False,
    return_animation=False,
    **kwargs,
):

    """Display objects and paths graphically using the matplotlib library."""
    data = get_frames(
        objs=obj_list,
        colorsequence=colorsequence,
        zoom=zoom,
        animation=animation,
        mag_color_grad_apt=False,
        extra_backend="matplotlib",
        **kwargs,
    )
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
    if canvas is None:
        show_canvas = True
        fig = plt.figure(dpi=80, figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect((1, 1, 1))
    else:
        ax = canvas
        fig = ax.get_figure()

    def draw_frame(ind):
        for tr in frames[ind]["data"]:
            constructor = tr["constructor"]
            args = tr.get("args", ())
            kwargs = tr.get("kwargs", {})
            getattr(ax, constructor)(*args, **kwargs)
        ax.set(
            **{f"{k}label": f"{k} [mm]" for k in "xyz"},
            **{f"{k}lim": r for k, r in zip("xyz", ranges)},
        )

    def animate(ind):  # pragma: no cover
        plt.cla()
        draw_frame(ind)
        return [ax]

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
        out += (anim,)
    if show_canvas:
        plt.show()

    if out:
        return out[0] if len(out) == 1 else out

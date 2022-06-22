import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from magpylib._src.display.traces_generic import get_frames
from magpylib._src.display.traces_generic import subdivide_mesh_by_facecolor

# from magpylib._src.utility import format_obj_input


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
            trace_mpl = {
                "constructor": "plot_trisurf",
                "args": (x, y, z),
                "kwargs": {
                    "triangles": triangles,
                    "alpha": subtrace.get("opacity", None),
                    "color": subtrace.get("color", None),
                },
            }
            traces_mpl.append(trace_mpl)
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
        if mode is not None and "lines" not in mode:
            props["ls"] = ""

        trace_mpl = {
            "constructor": "plot",
            "args": (x, y, z),
            "kwargs": {
                **{k: v for k, v in props.items() if v is not None},
                "alpha": trace.get("opacity", 1),
            },
        }
        traces_mpl.append(trace_mpl)
    else:
        raise ValueError(
            f"Trace type {trace['type']!r} cannot be transformed into matplotlib trace"
        )
    return traces_mpl


def display_matplotlib(
    *obj_list,
    zoom=1,
    canvas=None,
    animation=False,
    repeat=False,
    colorsequence=None,
    return_animation=False,
    **kwargs,
):

    """Display objects and paths graphically using the matplotlib library."""
    data = get_frames(
        objs=obj_list,
        colorsequence=colorsequence,
        zoom=zoom,
        animation=animation,
        mag_arrows=True,
        **kwargs,
    )
    frames = data["frames"]
    ranges = data["ranges"]

    for fr in frames:
        fr["data"] = [
            tr0 for tr1 in fr["data"] for tr0 in generic_trace_to_matplotlib(tr1)
        ]
    show_canvas = False
    if canvas is None:
        show_canvas = True
        fig = plt.figure(dpi=80, figsize=(8, 8))
        canvas = fig.add_subplot(111, projection="3d")
        canvas.set_box_aspect((1, 1, 1))

    def draw_frame(ind):
        for tr in frames[ind]["data"]:
            constructor = tr["constructor"]
            args = tr["args"]
            kwargs = tr["kwargs"]
            getattr(canvas, constructor)(*args, **kwargs)
        canvas.set(
            **{f"{k}label": f"{k} [mm]" for k in "xyz"},
            **{f"{k}lim": r for k, r in zip("xyz", ranges)},
        )

    def animate(ind):
        plt.cla()
        draw_frame(ind)
        return [canvas]

    if len(frames) == 1:
        draw_frame(0)
    else:
        anim = FuncAnimation(
            fig,
            animate,
            frames=range(len(frames)),
            interval=100,
            blit=False,
            repeat=repeat,
        )
    if return_animation and len(frames) != 1:
        return anim
    elif show_canvas:
        plt.show()

import warnings

import numpy as np
from matplotlib.colors import colorConverter
from mayavi import mlab

from magpylib._src.display.traces_generic import get_frames
from magpylib._src.display.traces_utility import subdivide_mesh_by_facecolor


# from magpylib._src.utility import format_obj_input


def generic_trace_to_mayavi(trace):
    """Transform a generic trace into a mayavi trace"""
    traces_mvi = []
    if trace["type"] == "mesh3d":
        subtraces = [trace]
        if trace.get("facecolor", None) is not None:
            subtraces = subdivide_mesh_by_facecolor(trace)
        for subtrace in subtraces:
            x, y, z = np.array([subtrace[k] for k in "xyz"], dtype=float)
            triangles = np.array([subtrace[k] for k in "ijk"]).T
            opacity = trace.get("opacity", 1)
            color = subtrace.get("color", None)
            color = (0.0, 0.0, 0.0, opacity) if color is None else color
            color = colorConverter.to_rgb(color)
            trace_mvi = {
                "constructor": "triangular_mesh",
                "mlab_source_names": {"x": x, "y": y, "z": z, "triangles": triangles},
                "args": (x, y, z, triangles),
                "kwargs": {
                    # "scalars": subtrace.get("intensity", None),
                    # "alpha": subtrace.get("opacity", None),
                    "color": color,
                    "opacity": opacity,
                },
            }
            traces_mvi.append(trace_mvi)
    elif trace["type"] == "scatter3d":
        x, y, z = np.array([trace[k] for k in "xyz"], dtype=float)
        opacity = trace.get("opacity", 1)
        color = trace.get("line", {}).get("color", trace.get("line_color", None))
        color = (0.0, 0.0, 0.0, opacity) if color is None else color
        color = colorConverter.to_rgb(color)
        trace_mvi = {
            "constructor": "plot3d",
            "mlab_source_names": {"x": x, "y": y, "z": z},
            "args": (x, y, z),
            "kwargs": {
                "color": color,
                "opacity": opacity,
            },
        }
        traces_mvi.append(trace_mvi)
    else:
        raise ValueError(
            f"Trace type {trace['type']!r} cannot be transformed into mayavi trace"
        )
    return traces_mvi


def display_mayavi(
    *obj_list,
    zoom=1,
    canvas=None,
    animation=False,
    colorsequence=None,
    **kwargs,
):

    """Display objects and paths graphically using the mayavi library."""

    # flat_obj_list = format_obj_input(obj_list)

    show_canvas = True
    if canvas == "hold":
        show_canvas = False
    elif canvas is not None:
        msg = (
            "The mayavi backend does not support a specific backend. You can specify "
            "`canvas='hold'` if you want to hold `on mlab.show()`"
        )
        warnings.warn(msg)

    data = get_frames(
        objs=obj_list,
        colorsequence=colorsequence,
        zoom=zoom,
        animation=animation,
        extra_backend="pyvista",
        mag_arrows=True,
        **kwargs,
    )
    frames = data["frames"]
    for fr in frames:
        new_data = []
        for tr in fr["data"]:
            new_data.extend(generic_trace_to_mayavi(tr))
        fr["data"] = new_data

    mayvi_traces = []

    def draw_frame(frame_ind):
        for trace_ind, tr1 in enumerate(frames[frame_ind]["data"]):
            if frame_ind == 0:
                constructor = tr1["constructor"]
                args = tr1["args"]
                kwargs = tr1["kwargs"]
                tr = getattr(mlab, constructor)(*args, **kwargs)
                mayvi_traces.append(tr)
            else:
                mlab_source = getattr(mayvi_traces[trace_ind], "mlab_source")
                mlab_source.trait_set(**tr1["mlab_source_names"])

    draw_frame(0)

    if animation:

        @mlab.animate(delay=data["frame_duration"])
        def anim():
            while 1:
                for frame_ind, _ in enumerate(frames):
                    if frame_ind > 0:
                        draw_frame(frame_ind)
                    yield

        anim()

    if show_canvas:
        mlab.show()

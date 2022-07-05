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

    if animation is not False:
        msg = "The mayavi backend does not support animation at the moment.\n"
        msg += "Use `backend=plotly` instead."
        warnings.warn(msg)
        animation = False

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

    frame = data["frames"][0]  # select first, since no animation supported

    for tr0 in frame["data"]:
        for tr1 in generic_trace_to_mayavi(tr0):
            constructor = tr1["constructor"]
            args = tr1["args"]
            kwargs = tr1["kwargs"]
            getattr(mlab, constructor)(*args, **kwargs)

    if show_canvas:
        mlab.show()

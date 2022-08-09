import warnings
from functools import lru_cache

import numpy as np
from matplotlib.colors import colorConverter
from mayavi import mlab

from magpylib._src.display.traces_generic import get_frames
from magpylib._src.display.traces_utility import subdivide_mesh_by_facecolor


# from magpylib._src.utility import format_obj_input

SYMBOLS = {
    "circle": "2dcircle",
    "cross": "2dcross",
    "diamond": "2ddiamond",
    "square": "2dsquare",
    "x": "2dcross",
    ".": "sphere",
    "o": "2dcircle",
    "+": "2dcross",
    "D": "2ddiamond",
    "d": "2ddiamond",
    "s": "2dsquare",
}


@lru_cache
def to_rgba_array(color, opacity=1):
    """Convert color to rgba_array"""
    return colorConverter.to_rgba_array(
        (0.0, 0.0, 0.0, opacity) if color is None else color
    )[0]


@lru_cache
def colorscale_to_lut(colorscale, opacity=1):
    "Convert plotly colorscale to vtk lut array."
    colors = np.array([to_rgba_array(v[1], opacity) * 255 for v in colorscale])
    print([int(256 * v[0]) for v in colorscale])
    repeat_inds = np.diff([int(256 * v[0]) for v in colorscale], prepend=0)
    return np.repeat(colors, repeat_inds, axis=0)


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
            color = colorConverter.to_rgb(
                (0.0, 0.0, 0.0, opacity) if color is None else color
            )
            colorscale = subtrace.get("colorscale", None)
            if colorscale is None:
                color_kwargs = {"color": color, "opacity": opacity}
            else:
                color_kwargs = {"lut": colorscale_to_lut(colorscale, opacity)}
            trace_mvi = {
                "constructor": "triangular_mesh",
                "mlab_source_names": {"x": x, "y": y, "z": z, "triangles": triangles},
                "args": (x, y, z, triangles),
                "kwargs": {
                    "scalars": subtrace.get("intensity", None),
                    **color_kwargs,
                },
            }
            traces_mvi.append(trace_mvi)
    elif trace["type"] == "scatter3d":
        x, y, z = np.array([trace[k] for k in "xyz"], dtype=float)
        opacity = trace.get("opacity", 1)
        line = trace.get("line", {})
        line_color = line.get("color", trace.get("line_color", None))
        line_color = colorConverter.to_rgb(
            (0.0, 0.0, 0.0, opacity) if line_color is None else line_color
        )
        marker_color = line.get("color", trace.get("marker_color", None))
        marker_color = colorConverter.to_rgb(
            line_color if marker_color is None else marker_color
        )
        trace_mvi_base = {
            "mlab_source_names": {"x": x, "y": y, "z": z},
            "args": (x, y, z),
        }
        kwargs = {"opacity": opacity, "color": line_color}
        mode = trace.get("mode", None)
        if mode is not None:
            if "markers" in mode:
                marker = trace.get("marker", {})
                marker_size = marker.get("size", trace.get("marker_size", 1))
                marker_symbol = marker.get(
                    "symbol", trace.get("marker_symbol", "2dcross")
                )
                marker_symbol = SYMBOLS.get(marker_symbol, "2dcross")
                trace_mvi1 = {"constructor": "points3d", **trace_mvi_base}
                trace_mvi1["kwargs"] = {
                    "scale_factor": 0.2 * marker_size,
                    "mode": marker_symbol,
                    "color": marker_color,
                    **kwargs,
                }
                traces_mvi.append(trace_mvi1)
            if "lines" in mode:
                trace_mvi2 = {"constructor": "plot3d", **trace_mvi_base}
                trace_mvi2["kwargs"] = {**kwargs}
                traces_mvi.append(trace_mvi2)
            if "text" in mode and trace.get("text", False):
                for xs, ys, zs, txt in zip(x, y, z, trace["text"]):
                    trace_mvi3 = {
                        "constructor": "text3d",
                        **trace_mvi_base,
                        "args": (xs, ys, zs, str(txt)),
                    }
                    trace_mvi3["kwargs"] = {**kwargs, "scale": 0.5}
                    traces_mvi.append(trace_mvi3)
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
        mag_arrows=False,
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
                lut = kwargs.pop("lut", None)
                tr = getattr(mlab, constructor)(*args, **kwargs)
                if lut is not None:
                    tr.module_manager.scalar_lut_manager.lut.table = lut
                mayvi_traces.append(tr)
            else:
                mlab_source = getattr(mayvi_traces[trace_ind], "mlab_source")
                mlab_source.trait_set(**tr1["mlab_source_names"])
        mlab.draw()

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

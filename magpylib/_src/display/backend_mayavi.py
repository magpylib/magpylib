import warnings

import numpy as np
from matplotlib.colors import colorConverter
from mayavi import mlab

from magpylib._src.display.traces_generic import draw_frame


# from magpylib._src.utility import format_obj_input


def subdivide_mesh_by_facecolor(trace):
    """Subdivide a mesh into a list of meshes based on facecolor"""
    # TODO so far the function keeps all x,y,z coords for all subtraces, which is convienient since
    # it does not require to recalculate the indices i,j,k. If many different colors, this is
    # become inpractical.
    facecolor = trace["facecolor"]
    subtraces = []
    last_ind = 0
    prev_color = facecolor[0]
    # pylint: disable=singleton-comparison
    facecolor[facecolor == None] = "black"
    for ind, color in enumerate(facecolor):
        if color != prev_color or ind == len(facecolor) - 1:
            new_trace = trace.copy()
            for k in "ijk":
                new_trace[k] = trace[k][last_ind:ind]
            new_trace["color"] = prev_color
            last_ind = ind
            prev_color = color
            subtraces.append(new_trace)
    return subtraces


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

    """
    Display objects and paths graphically using the mayavi library.

    Parameters
    ----------
    objects: sources, collections or sensors
        Objects to be displayed.

    markers: array_like, None, shape (N,3), default=None
        Display position markers in the global CS. By default no marker is displayed.

    zoom: float, default = 1
        Adjust plot zoom-level. When zoom=0 all objects are just inside the 3D-axes.

    canvas: #TODO what canvas
        Display graphical output in a given canvas
        By default a new `Figure` is created and displayed.

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
    """

    if animation is not False:
        msg = "The mayavi backend does not support animation at the moment.\n"
        msg += "Use `backend=plotly` instead."
        warnings.warn(msg)
        # animation = False

    # flat_obj_list = format_obj_input(obj_list)

    show_canvas = False
    if canvas is None:
        show_canvas = True
        canvas = mlab

    generic_traces = draw_frame(
        obj_list,
        colorsequence,
        zoom,
        output="list",
        mag_arrows=True,
        **kwargs,
    )
    for tr in generic_traces:
        for tr1 in generic_trace_to_mayavi(tr):
            constructor = tr1["constructor"]
            args = tr1["args"]
            kwargs = tr1["kwargs"]
            getattr(canvas, constructor)(*args, **kwargs)

    # apply_fig_ranges(canvas, zoom=zoom)
    if show_canvas:
        mlab.show()

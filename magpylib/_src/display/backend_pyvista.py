import warnings
from functools import lru_cache

import numpy as np

try:
    import pyvista as pv
except ImportError as missing_module:  # pragma: no cover
    raise ModuleNotFoundError(
        """In order to use the pyvista plotting backend, you need to install pyvista via pip or
        conda, see https://docs.pyvista.org/getting-started/installation.html"""
    ) from missing_module

from pyvista.plotting.colors import Color  # pylint: disable=import-error
from matplotlib.colors import LinearSegmentedColormap
from magpylib._src.display.traces_generic import get_frames

# from magpylib._src.utility import format_obj_input


@lru_cache(maxsize=32)
def colormap_from_colorscale(colorscale, name="plotly_to_mpl", N=256, gamma=1.0):
    """Create matplotlib colormap from plotly colorscale"""

    cs_rgb = [(v[0], Color(v[1]).float_rgb) for v in colorscale]
    cdict = {
        rgb_col: [
            (
                c[0],
                *[c[1][rgb_ind]] * 2,
            )
            for c in cs_rgb
        ]
        for rgb_ind, rgb_col in enumerate(("red", "green", "blue"))
    }
    return LinearSegmentedColormap(name, cdict, N, gamma)


def generic_trace_to_pyvista(trace, jupyter_backend=None):
    """Transform a generic trace into a pyvista trace"""
    traces_pv = []
    if trace["type"] == "mesh3d":
        vertices = np.array([trace[k] for k in "xyz"], dtype=float).T
        faces = np.array([trace[k] for k in "ijk"]).T.flatten()
        faces = np.insert(faces, range(0, len(faces), 3), 3)
        colorscale = trace.get("colorscale", None)
        mesh = pv.PolyData(vertices, faces)
        facecolor = trace.get("facecolor", None)
        trace_pv = {
            "mesh": mesh,
            "opacity": trace.get("opacity", None),
            "color": trace.get("color", None),
            "scalars": trace.get("intensity", None),
        }
        if facecolor is not None:
            # pylint: disable=unsupported-assignment-operation
            mesh.cell_data["colors"] = [
                Color(c, default_color=(0, 0, 0)).int_rgb for c in facecolor
            ]
            trace_pv.update(
                {
                    "scalars": "colors",
                    "rgb": True,
                    "preference": "cell",
                }
            )
        traces_pv.append(trace_pv)
        if colorscale is not None:
            # ipygany does not support custom colorsequences
            if (
                pv.global_theme.jupyter_backend == "ipygany"
                or jupyter_backend == "ipygany"
            ):
                trace_pv["cmap"] = "PiYG"
            else:
                trace_pv["cmap"] = colormap_from_colorscale(colorscale)
    elif trace["type"] == "scatter3d":
        points = np.array([trace[k] for k in "xyz"], dtype=float).T
        line = trace.get("line", {})
        line_color = line.get("color", trace.get("line_color", None))
        line_width = line.get("width", trace.get("line_width", None))
        trace_pv_line = {
            "mesh": pv.lines_from_points(points),
            "opacity": trace.get("opacity", None),
            "color": line_color,
            "line_width": line_width,
        }
        traces_pv.append(trace_pv_line)
        marker = trace.get("marker", {})
        marker_color = marker.get("color", trace.get("marker_color", None))
        # marker_symbol = marker.get("symbol", trace.get("marker_symbol", None))
        marker_size = marker.get("size", trace.get("marker_size", None))
        trace_pv_marker = {
            "mesh": pv.PolyData(points),
            "opacity": trace.get("opacity", None),
            "color": marker_color,
            "point_size": 1 if marker_size is None else marker_size,
        }
        traces_pv.append(trace_pv_marker)
    else:  # pragma: no cover
        raise ValueError(
            f"Trace type {trace['type']!r} cannot be transformed into pyvista trace"
        )
    return traces_pv


def display_pyvista(
    *obj_list,
    zoom=1,
    canvas=None,
    animation=False,
    colorsequence=None,
    return_fig=False,
    jupyter_backend=None,
    **kwargs,
):

    """Display objects and paths graphically using the pyvista library."""

    if animation is not False:
        warnings.warn(
            "The pyvista backend does not support animation at the moment.\n"
            "Use `backend=plotly` instead."
        )
        # animation = False

    # flat_obj_list = format_obj_input(obj_list)

    show_canvas = False
    if canvas is None:
        if not return_fig:
            show_canvas = True
        canvas = pv.Plotter()
    data = get_frames(
        objs=obj_list,
        colorsequence=colorsequence,
        zoom=zoom,
        animation=animation,
        extra_backend="pyvista",
        **kwargs,
    )

    frame = data["frames"][0]  # select first, since no animation supported

    for tr0 in frame["data"]:
        for tr1 in generic_trace_to_pyvista(tr0, jupyter_backend=jupyter_backend):
            canvas.add_mesh(**tr1)

    # apply_fig_ranges(canvas, zoom=zoom)
    try:
        canvas.remove_scalar_bar()
    except IndexError:
        pass

    # match other backends plotter properties
    canvas.set_background('gray', top='white')
    canvas.show_axes()
    canvas.camera.azimuth = -90

    if return_fig and not show_canvas:
        return canvas
    if show_canvas:
        canvas.show(jupyter_backend=jupyter_backend)
    return None

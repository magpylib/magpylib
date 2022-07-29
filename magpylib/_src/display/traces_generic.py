"""Generic trace drawing functionalities"""
# pylint: disable=C0302
# pylint: disable=too-many-branches
import numbers
import warnings
from itertools import combinations
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as RotScipy

from magpylib import _src
from magpylib._src.defaults.defaults_classes import default_settings as Config
from magpylib._src.defaults.defaults_utility import linearize_dict
from magpylib._src.display.sensor_mesh import get_sensor_mesh
from magpylib._src.display.traces_base import make_Arrow as make_BaseArrow
from magpylib._src.display.traces_base import make_Cuboid as make_BaseCuboid
from magpylib._src.display.traces_base import (
    make_CylinderSegment as make_BaseCylinderSegment,
)
from magpylib._src.display.traces_base import make_Ellipsoid as make_BaseEllipsoid
from magpylib._src.display.traces_base import make_Prism as make_BasePrism
from magpylib._src.display.traces_utility import draw_arrow_from_vertices
from magpylib._src.display.traces_utility import draw_arrowed_circle
from magpylib._src.display.traces_utility import draw_arrowed_line
from magpylib._src.display.traces_utility import get_flatten_objects_properties
from magpylib._src.display.traces_utility import get_rot_pos_from_path
from magpylib._src.display.traces_utility import get_scene_ranges
from magpylib._src.display.traces_utility import getColorscale
from magpylib._src.display.traces_utility import getIntensity
from magpylib._src.display.traces_utility import group_traces
from magpylib._src.display.traces_utility import merge_mesh3d
from magpylib._src.display.traces_utility import merge_traces
from magpylib._src.display.traces_utility import place_and_orient_model3d
from magpylib._src.input_checks import check_excitations
from magpylib._src.style import get_style
from magpylib._src.style import Markers
from magpylib._src.utility import format_obj_input
from magpylib._src.utility import unit_prefix


class MagpyMarkers:
    """A class that stores markers 3D-coordinates"""

    def __init__(self, *markers):
        self.style = Markers()
        self.markers = np.array(markers)

    def _draw_func(self, color=None, style=None, **kwargs):
        """Create the plotly mesh3d parameters for a Sensor object in a dictionary based on the
        provided arguments."""
        style = self.style if style is None else style
        x, y, z = self.markers.T
        marker_kwargs = {
            f"marker_{k}": v
            for k, v in style.marker.as_dict(flatten=True, separator="_").items()
        }
        marker_kwargs["marker_color"] = (
            style.marker.color if style.marker.color is not None else color
        )
        trace = dict(
            type="scatter3d",
            x=x,
            y=y,
            z=z,
            mode="markers",
            **marker_kwargs,
            **kwargs,
        )
        default_name = "Marker" if len(x) == 1 else "Markers"
        default_suffix = "" if len(x) == 1 else f" ({len(x)} points)"
        update_trace_name(trace, default_name, default_suffix, style)
        return trace


def make_DefaultTrace(
    obj,
    position=(0.0, 0.0, 0.0),
    orientation=None,
    color=None,
    style=None,
    **kwargs,
) -> dict:
    """
    Creates the plotly scatter3d parameters for an object with no specifically supported
    representation. The object will be represented by a scatter point and text above with object
    name.
    """
    style = obj.style if style is None else style
    trace = dict(
        type="scatter3d",
        x=[0.0],
        y=[0.0],
        z=[0.0],
        mode="markers+text",
        marker_size=10,
        marker_color=color,
        marker_symbol="diamond",
    )
    update_trace_name(trace, f"{type(obj).__name__}", "", style)
    trace["text"] = trace["name"]
    return place_and_orient_model3d(
        trace, orientation=orientation, position=position, **kwargs
    )


def make_Line(
    obj,
    position=(0.0, 0.0, 0.0),
    orientation=None,
    color=None,
    style=None,
    **kwargs,
) -> dict:
    """
    Creates the plotly scatter3d parameters for a Line current in a dictionary based on the
    provided arguments.
    """
    style = obj.style if style is None else style
    current = obj.current
    vertices = obj.vertices
    show_arrows = style.arrow.show
    arrow_size = style.arrow.size
    if show_arrows:
        vertices = draw_arrow_from_vertices(vertices, current, arrow_size)
    else:
        vertices = np.array(vertices).T
    x, y, z = vertices
    trace = dict(
        type="scatter3d",
        x=x,
        y=y,
        z=z,
        mode="lines",
        line_width=style.arrow.width,
        line_color=color,
    )
    default_suffix = (
        f" ({unit_prefix(current)}A)"
        if current is not None
        else " (Current not initialized)"
    )
    update_trace_name(trace, "Line", default_suffix, style)
    return place_and_orient_model3d(
        trace, orientation=orientation, position=position, **kwargs
    )


def make_Loop(
    obj,
    position=(0.0, 0.0, 0.0),
    orientation=None,
    color=None,
    style=None,
    vertices=50,
    **kwargs,
):
    """
    Creates the plotly scatter3d parameters for a Loop current in a dictionary based on the
    provided arguments.
    """
    style = obj.style if style is None else style
    current = obj.current
    diameter = obj.diameter
    arrow_size = style.arrow.size if style.arrow.show else 0
    vertices = draw_arrowed_circle(current, diameter, arrow_size, vertices)
    x, y, z = vertices
    trace = dict(
        type="scatter3d",
        x=x,
        y=y,
        z=z,
        mode="lines",
        line_width=style.arrow.width,
        line_color=color,
    )
    default_suffix = (
        f" ({unit_prefix(current)}A)"
        if current is not None
        else " (Current not initialized)"
    )
    update_trace_name(trace, "Loop", default_suffix, style)
    return place_and_orient_model3d(
        trace, orientation=orientation, position=position, **kwargs
    )


def make_Dipole(
    obj,
    position=(0.0, 0.0, 0.0),
    orientation=None,
    color=None,
    style=None,
    autosize=None,
    **kwargs,
) -> dict:
    """
    Create the plotly mesh3d parameters for a Loop current in a dictionary based on the
    provided arguments.
    """
    style = obj.style if style is None else style
    moment = obj.moment
    moment_mag = np.linalg.norm(moment)
    size = style.size
    if autosize is not None:
        size *= autosize
    trace = make_BaseArrow(
        "plotly-dict",
        base=10,
        diameter=0.3 * size,
        height=size,
        pivot=style.pivot,
        color=color,
    )
    default_suffix = f" (moment={unit_prefix(moment_mag)}mT mm³)"
    update_trace_name(trace, "Dipole", default_suffix, style)
    nvec = np.array(moment) / moment_mag
    zaxis = np.array([0, 0, 1])
    cross = np.cross(nvec, zaxis)
    n = np.linalg.norm(cross)
    if n == 0:
        n = 1
        cross = np.array([-np.sign(nvec[-1]), 0, 0])
    dot = np.dot(nvec, zaxis)
    t = np.arccos(dot)
    vec = -t * cross / n
    mag_orient = RotScipy.from_rotvec(vec)
    orientation = orientation * mag_orient
    return place_and_orient_model3d(
        trace, orientation=orientation, position=position, **kwargs
    )


def make_Cuboid(
    obj,
    position=(0.0, 0.0, 0.0),
    orientation=None,
    color=None,
    style=None,
    **kwargs,
) -> dict:
    """
    Create the plotly mesh3d parameters for a Cuboid Magnet in a dictionary based on the
    provided arguments.
    """
    style = obj.style if style is None else style
    dimension = obj.dimension
    d = [unit_prefix(d / 1000) for d in dimension]
    trace = make_BaseCuboid("plotly-dict", dimension=dimension, color=color)
    default_suffix = f" ({d[0]}m|{d[1]}m|{d[2]}m)"
    update_trace_name(trace, "Cuboid", default_suffix, style)
    update_magnet_mesh(
        trace, mag_style=style.magnetization, magnetization=obj.magnetization
    )
    return place_and_orient_model3d(
        trace, orientation=orientation, position=position, **kwargs
    )


def make_Cylinder(
    obj,
    position=(0.0, 0.0, 0.0),
    orientation=None,
    color=None,
    style=None,
    base=50,
    **kwargs,
) -> dict:
    """
    Create the plotly mesh3d parameters for a Cylinder Magnet in a dictionary based on the
    provided arguments.
    """
    style = obj.style if style is None else style
    diameter, height = obj.dimension
    d = [unit_prefix(d / 1000) for d in (diameter, height)]
    trace = make_BasePrism(
        "plotly-dict", base=base, diameter=diameter, height=height, color=color
    )
    default_suffix = f" (D={d[0]}m, H={d[1]}m)"
    update_trace_name(trace, "Cylinder", default_suffix, style)
    update_magnet_mesh(
        trace, mag_style=style.magnetization, magnetization=obj.magnetization
    )
    return place_and_orient_model3d(
        trace, orientation=orientation, position=position, **kwargs
    )


def make_CylinderSegment(
    obj,
    position=(0.0, 0.0, 0.0),
    orientation=None,
    color=None,
    style=None,
    vertices=25,
    **kwargs,
):
    """
    Create the plotly mesh3d parameters for a Cylinder Segment Magnet in a dictionary based on the
    provided arguments.
    """
    style = obj.style if style is None else style
    dimension = obj.dimension
    d = [unit_prefix(d / (1000 if i < 3 else 1)) for i, d in enumerate(dimension)]
    trace = make_BaseCylinderSegment(
        "plotly-dict", dimension=dimension, vert=vertices, color=color
    )
    default_suffix = f" (r={d[0]}m|{d[1]}m, h={d[2]}m, φ={d[3]}°|{d[4]}°)"
    update_trace_name(trace, "CylinderSegment", default_suffix, style)
    update_magnet_mesh(
        trace, mag_style=style.magnetization, magnetization=obj.magnetization
    )
    return place_and_orient_model3d(
        trace, orientation=orientation, position=position, **kwargs
    )


def make_Sphere(
    obj,
    position=(0.0, 0.0, 0.0),
    orientation=None,
    color=None,
    style=None,
    vertices=15,
    **kwargs,
) -> dict:
    """
    Create the plotly mesh3d parameters for a Sphere Magnet in a dictionary based on the
    provided arguments.
    """
    style = obj.style if style is None else style
    diameter = obj.diameter
    vertices = min(max(vertices, 3), 20)
    trace = make_BaseEllipsoid(
        "plotly-dict", vert=vertices, dimension=[diameter] * 3, color=color
    )
    default_suffix = f" (D={unit_prefix(diameter / 1000)}m)"
    update_trace_name(trace, "Sphere", default_suffix, style)
    update_magnet_mesh(
        trace, mag_style=style.magnetization, magnetization=obj.magnetization
    )
    return place_and_orient_model3d(
        trace, orientation=orientation, position=position, **kwargs
    )


def make_Pixels(positions, size=1) -> dict:
    """
    Create the plotly mesh3d parameters for Sensor pixels based on pixel positions and chosen size
    For now, only "cube" shape is provided.
    """
    pixels = [
        make_BaseCuboid("plotly-dict", position=p, dimension=[size] * 3)
        for p in positions
    ]
    return merge_mesh3d(*pixels)


def make_Sensor(
    obj,
    position=(0.0, 0.0, 0.0),
    orientation=None,
    color=None,
    style=None,
    autosize=None,
    **kwargs,
):
    """
    Create the plotly mesh3d parameters for a Sensor object in a dictionary based on the
    provided arguments.

    size_pixels: float, default=1
        A positive number. Adjusts automatic display size of sensor pixels. When set to 0,
        pixels will be hidden, when greater than 0, pixels will occupy half the ratio of the minimum
        distance between any pixel of the same sensor, equal to `size_pixel`.
    """
    style = obj.style if style is None else style
    dimension = getattr(obj, "dimension", style.size)
    pixel = obj.pixel
    pixel = np.array(pixel).reshape((-1, 3))
    style_arrows = style.arrows.as_dict(flatten=True, separator="_")
    sensor = get_sensor_mesh(**style_arrows, center_color=color)
    vertices = np.array([sensor[k] for k in "xyz"]).T
    if color is not None:
        sensor["facecolor"][sensor["facecolor"] == "rgb(238,238,238)"] = color
    dim = np.array(
        [dimension] * 3 if isinstance(dimension, (float, int)) else dimension[:3],
        dtype=float,
    )
    if autosize is not None:
        dim *= autosize
    if pixel.shape[0] == 1:
        dim_ext = dim
    else:
        hull_dim = pixel.max(axis=0) - pixel.min(axis=0)
        dim_ext = max(np.mean(dim), np.min(hull_dim))
    cube_mask = (vertices < 1).all(axis=1)
    vertices[cube_mask] = 0 * vertices[cube_mask]
    vertices[~cube_mask] = dim_ext * vertices[~cube_mask]
    vertices /= 2  # sensor_mesh vertices are of length 2
    x, y, z = vertices.T
    sensor.update(x=x, y=y, z=z)
    meshes_to_merge = [sensor]
    if pixel.shape[0] != 1:
        pixel_color = style.pixel.color
        pixel_size = style.pixel.size
        combs = np.array(list(combinations(pixel, 2)))
        vecs = np.diff(combs, axis=1)
        dists = np.linalg.norm(vecs, axis=2)
        pixel_dim = np.min(dists) / 2
        if pixel_size > 0:
            pixel_dim *= pixel_size
            pixels_mesh = make_Pixels(positions=pixel, size=pixel_dim)
            pixels_mesh["facecolor"] = np.repeat(pixel_color, len(pixels_mesh["i"]))
            meshes_to_merge.append(pixels_mesh)
        hull_pos = 0.5 * (pixel.max(axis=0) + pixel.min(axis=0))
        hull_dim[hull_dim == 0] = pixel_dim / 2
        hull_mesh = make_BaseCuboid(
            "plotly-dict", position=hull_pos, dimension=hull_dim
        )
        hull_mesh["facecolor"] = np.repeat(color, len(hull_mesh["i"]))
        meshes_to_merge.append(hull_mesh)
    trace = merge_mesh3d(*meshes_to_merge)
    default_suffix = (
        f""" ({'x'.join(str(p) for p in pixel.shape[:-1])} pixels)"""
        if pixel.ndim != 1
        else ""
    )
    update_trace_name(trace, "Sensor", default_suffix, style)
    return place_and_orient_model3d(
        trace, orientation=orientation, position=position, **kwargs
    )


def update_magnet_mesh(mesh_dict, mag_style=None, magnetization=None):
    """
    Updates an existing plotly mesh3d dictionary of an object which has a magnetic vector. The
    object gets colorized, positioned and oriented based on provided arguments.
    """
    mag_color = mag_style.color
    if magnetization is not None and mag_style.show:
        vertices = np.array([mesh_dict[k] for k in "xyz"]).T
        color_middle = mag_color.middle
        if mag_color.mode == "tricycle":
            color_middle = mesh_dict["color"]
        elif mag_color.mode == "bicolor":
            color_middle = False
        mesh_dict["colorscale"] = getColorscale(
            color_transition=mag_color.transition,
            color_north=mag_color.north,
            color_middle=color_middle,
            color_south=mag_color.south,
        )
        mesh_dict["intensity"] = getIntensity(
            vertices=vertices,
            axis=magnetization,
        )
        mesh_dict["showscale"] = False
    return mesh_dict


def update_trace_name(trace, default_name, default_suffix, style):
    """provides legend entry based on name and suffix"""
    name = default_name if style.label is None else style.label
    if style.description.show and style.description.text is None:
        name_suffix = default_suffix
    elif not style.description.show:
        name_suffix = ""
    else:
        name_suffix = f" ({style.description.text})"
    trace.update(name=f"{name}{name_suffix}")
    return trace


def make_mag_arrows(obj, style, legendgroup, kwargs):
    """draw direction of magnetization of faced magnets

    Parameters
    ----------
    - faced_objects(list of src objects): with magnetization vector to be drawn
    - colors: colors of faced_objects
    - show_path(bool or int): draw on every position where object is displayed
    """
    # pylint: disable=protected-access

    # add src attributes position and orientation depending on show_path
    rots, _, inds = get_rot_pos_from_path(obj, style.path.frames)

    # vector length, color and magnetization
    if hasattr(obj, "diameter"):
        length = obj.diameter  # Sphere
    else:  # Cuboid, Cylinder, CylinderSegment
        length = np.amax(obj.dimension[:3])
    length *= 1.8 * style.magnetization.size
    mag = obj.magnetization
    # collect all draw positions and directions
    points = []
    for rot, ind in zip(rots, inds):
        pos = getattr(obj, "_barycenter", obj._position)[ind]
        direc = mag / (np.linalg.norm(mag) + 1e-6) * length
        vec = rot.apply(direc)
        pts = draw_arrowed_line(vec, pos, sign=1, arrow_pos=1, pivot="tail")
        points.append(pts)
    # insert empty point to avoid connecting line between arrows
    points = np.array(points)
    points = np.insert(points, points.shape[-1], np.nan, axis=2)
    # remove last nan after insert with [:-1]
    x, y, z = np.concatenate(points.swapaxes(1, 2))[:-1].T
    trace = {
        "type": "scatter3d",
        "mode": "lines",
        "line_color": kwargs["color"],
        "opacity": kwargs["opacity"],
        "x": x,
        "y": y,
        "z": z,
        "legendgroup": legendgroup,
        "showlegend": False,
    }
    return trace


def make_path(input_obj, style, legendgroup, kwargs):
    """draw obj path based on path style properties"""
    x, y, z = np.array(input_obj.position).T
    txt_kwargs = (
        {"mode": "markers+text+lines", "text": list(range(len(x)))}
        if style.path.numbering
        else {"mode": "markers+lines"}
    )
    marker = style.path.marker.as_dict()
    marker["symbol"] = marker["symbol"]
    marker["color"] = kwargs["color"] if marker["color"] is None else marker["color"]
    line = style.path.line.as_dict()
    line["dash"] = line["style"]
    line["color"] = kwargs["color"] if line["color"] is None else line["color"]
    line = {k: v for k, v in line.items() if k != "style"}
    scatter_path = dict(
        type="scatter3d",
        x=x,
        y=y,
        z=z,
        name=f"Path: {input_obj}",
        showlegend=False,
        legendgroup=legendgroup,
        **{f"marker_{k}": v for k, v in marker.items()},
        **{f"line_{k}": v for k, v in line.items()},
        **txt_kwargs,
        opacity=kwargs["opacity"],
    )
    return scatter_path


def get_generic_traces(
    input_obj,
    color=None,
    autosize=None,
    legendgroup=None,
    showlegend=None,
    legendtext=None,
    mag_arrows=False,
    extra_backend=False,
    row=1,
    col=1,
    **kwargs,
) -> list:
    """
    This is a helper function providing the plotly traces for any object of the magpylib library. If
    the object is not supported, the trace representation will fall back to a single scatter point
    with the object name marked above it.

    - If the object has a path (multiple positions), the function will return both the object trace
    and the corresponding path trace. The legend entry of the path trace will be hidden but both
    traces will share the same `legendgroup` so that a legend entry click will hide/show both traces
    at once. From the user's perspective, the traces will be merged.

    - The argument caught by the kwargs dictionary must all be arguments supported both by
    `scatter3d` and `mesh3d` plotly objects, otherwise an error will be raised.
    """

    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-nested-blocks
    # pylint: disable=protected-access

    # parse kwargs into style and non style args
    style = get_style(input_obj, Config, **kwargs)
    kwargs = {k: v for k, v in kwargs.items() if not k.startswith("style")}
    kwargs["style"] = style
    style_color = getattr(style, "color", None)
    kwargs["color"] = style_color if style_color is not None else color
    kwargs["opacity"] = style.opacity
    legendgroup = f"{input_obj}" if legendgroup is None else legendgroup

    # check excitations validity
    for param in ("magnetization", "arrow"):
        if getattr(getattr(style, param, None), "show", False):
            check_excitations([input_obj])

    label = getattr(getattr(input_obj, "style", None), "label", None)
    label = label if label is not None else str(type(input_obj).__name__)

    make_func = input_obj._draw_func
    make_func_kwargs = kwargs.copy()
    if getattr(input_obj, "_autosize", False):
        make_func_kwargs["autosize"] = autosize

    traces = []
    path_traces = []
    path_traces_extra_generic = {}
    path_traces_extra_specific_backend = []
    has_path = hasattr(input_obj, "position") and hasattr(input_obj, "orientation")
    if not has_path:
        tr = make_func(**make_func_kwargs)
        tr["row"] = row
        tr["col"] = col
        traces = [tr]
        out = (traces,)
        if extra_backend is not False:
            out += (path_traces_extra_specific_backend,)
        return out[0] if len(out) == 1 else out

    extra_model3d_traces = style.model3d.data if style.model3d.data is not None else []
    orientations, positions, _ = get_rot_pos_from_path(input_obj, style.path.frames)
    for pos_orient_ind, (orient, pos) in enumerate(zip(orientations, positions)):
        if style.model3d.showdefault and make_func is not None:
            path_traces.append(
                make_func(position=pos, orientation=orient, **make_func_kwargs)
            )
        for extr in extra_model3d_traces:
            if extr.show:
                extr.update(extr.updatefunc())
                if extr.backend == "generic":
                    trace3d = {"opacity": kwargs["opacity"], "row": row, "col": col}
                    ttype = extr.constructor.lower()
                    obj_extr_trace = (
                        extr.kwargs() if callable(extr.kwargs) else extr.kwargs
                    )
                    obj_extr_trace = {"type": ttype, **obj_extr_trace}
                    if ttype == "scatter3d":
                        for k in ("marker", "line"):
                            trace3d[f"{k}_color"] = trace3d.get(
                                f"{k}_color", kwargs["color"]
                            )
                    elif ttype == "mesh3d":
                        trace3d["showscale"] = trace3d.get("showscale", False)
                        if "facecolor" in obj_extr_trace:
                            ttype = "mesh3d_facecolor"
                        trace3d["color"] = trace3d.get("color", kwargs["color"])
                    else:  # pragma: no cover
                        raise ValueError(
                            f"{ttype} is not supported, only 'scatter3d' and 'mesh3d' are"
                        )
                    trace3d.update(
                        linearize_dict(
                            place_and_orient_model3d(
                                model_kwargs=obj_extr_trace,
                                orientation=orient,
                                position=pos,
                                scale=extr.scale,
                            ),
                            separator="_",
                        )
                    )
                    if ttype not in path_traces_extra_generic:
                        path_traces_extra_generic[ttype] = []
                    path_traces_extra_generic[ttype].append(trace3d)
                elif extr.backend == extra_backend:
                    showleg = (
                        showlegend
                        and pos_orient_ind == 0
                        and not style.model3d.showdefault
                    )
                    showleg = True if showleg is None else showleg
                    trace3d = {
                        "model3d": extr,
                        "position": pos,
                        "orientation": orient,
                        "kwargs": {
                            "opacity": kwargs["opacity"],
                            "color": kwargs["color"],
                            "legendgroup": legendgroup,
                            "name": label,
                            "showlegend": showleg,
                            "row": row,
                            "col": col,
                        },
                    }
                    path_traces_extra_specific_backend.append(trace3d)
    trace = merge_traces(*path_traces)
    for ind, traces_extra in enumerate(path_traces_extra_generic.values()):
        extra_model3d_trace = merge_traces(*traces_extra)
        extra_model3d_trace.update(
            {
                "legendgroup": legendgroup,
                "showlegend": showlegend and ind == 0 and not trace,
                "name": label,
            }
        )
        traces.append(extra_model3d_trace)

    if trace:
        trace.update(
            {
                "legendgroup": legendgroup,
                "showlegend": True if showlegend is None else showlegend,
            }
        )
        if legendtext is not None:
            trace["name"] = legendtext
        traces.append(trace)

    if np.array(input_obj.position).ndim > 1 and style.path.show:
        scatter_path = make_path(input_obj, style, legendgroup, kwargs)
        traces.append(scatter_path)

    if mag_arrows and getattr(input_obj, "magnetization", None) is not None:
        if style.magnetization.show:
            traces.append(make_mag_arrows(input_obj, style, legendgroup, kwargs))
    for tr in traces:
        tr["row"] = row
        tr["col"] = col
    out = (traces,)
    if extra_backend is not False:
        out += (path_traces_extra_specific_backend,)
    return out[0] if len(out) == 1 else out


def clean_legendgroups(frames):
    """removes legend duplicates for a plotly figure"""
    for fr in frames:
        legendgroups = []
        for tr in fr["data"]:
            lg = tr.get("legendgroup", None)
            if lg is not None and lg not in legendgroups:
                legendgroups.append(lg)
            elif lg is not None:  # and tr.legendgrouptitle.text is None:
                tr["showlegend"] = False


def process_animation_kwargs(obj_list, animation=False, **kwargs):
    """Update animation kwargs"""
    flat_obj_list = format_obj_input(obj_list)
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

    anim_def = Config.display.animation.copy()
    anim_def.update({k[10:]: v for k, v in kwargs.items()}, _match_properties=False)
    animation_kwargs = {f"animation_{k}": v for k, v in anim_def.as_dict().items()}
    kwargs = {k: v for k, v in kwargs.items() if not k.startswith("animation")}
    return kwargs, animation, animation_kwargs


def extract_animation_properties(
    objs,
    *,
    animation_maxfps,
    animation_time,
    animation_fps,
    animation_maxframes,
    # pylint: disable=unused-argument
    animation_slider,
):
    """Exctract animation properties"""
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
            f" {animation_maxfps}. `animation_fps` will be set to"
            f" {animation_maxfps}. "
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

    return path_indices, exp, frame_duration


def draw_frame(
    objs,
    colorsequence=None,
    zoom=0.0,
    autosize=None,
    output="dict",
    mag_arrows=False,
    extra_backend=False,
    **kwargs,
) -> Tuple:
    """
    Creates traces from input `objs` and provided parameters, updates the size of objects like
    Sensors and Dipoles in `kwargs` depending on the canvas size.

    Returns
    -------
    traces_dicts, kwargs: dict, dict
        returns the traces in a obj/traces_list dictionary and updated kwargs
    """
    # pylint: disable=protected-access
    obj_list_semi_flat = format_obj_input(
        [o["objects"] for o in objs], allow="sources+sensors+collections"
    )
    if colorsequence is None:
        colorsequence = Config.display.colorsequence
    extra_backend_traces = []
    Sensor = _src.obj_classes.class_Sensor.Sensor
    Dipole = _src.obj_classes.class_misc_Dipole.Dipole
    traces_out = {}
    # dipoles and sensors use autosize, the trace building has to be put at the back of the queue.
    # autosize is calculated from the other traces overall scene range
    markers = (
        []
        if not objs
        else objs[-1]["objects"][-1:]
        if isinstance(objs[-1]["objects"], MagpyMarkers)
        else []
    )

    row_cols = get_row_cols(objs)
    flat_objs_props = get_flatten_objects_properties(
        *obj_list_semi_flat, *markers, colorsequence=colorsequence
    )
    traces_to_resize = {}
    for obj, params in flat_objs_props.items():
        params.update(kwargs)
        if isinstance(obj, (Dipole, Sensor)):
            traces_to_resize[obj] = {**params}
            # temporary coordinates to be able to calculate ranges
            x, y, z = obj._position.T
            traces_out[obj] = [dict(x=x, y=y, z=z)]
        else:
            traces_out[obj] = []
            for row_col in row_cols.get(obj, [(1, 1)]):
                params["row"], params["col"] = row_col
                out_traces = get_generic_traces(
                    obj,
                    mag_arrows=mag_arrows,
                    extra_backend=extra_backend,
                    **params,
                )
                if extra_backend is not False:
                    out_traces, ebt = out_traces
                    extra_backend_traces.extend(ebt)
                traces_out[obj].extend(out_traces)
    traces = [t for tr in traces_out.values() for t in tr]
    ranges = get_scene_ranges(*traces, zoom=zoom)
    if autosize is None or autosize == "return":
        autosize = np.mean(np.diff(ranges)) / Config.display.autosizefactor
    for obj, params in traces_to_resize.items():
        traces_out[obj] = []
        for row_col in row_cols.get(obj, [(1, 1)]):
            params["row"], params["col"] = row_col
            out_traces = get_generic_traces(
                obj,
                autosize=autosize,
                mag_arrows=mag_arrows,
                extra_backend=extra_backend,
                **params,
            )
            if extra_backend is not False:
                out_traces, ebt = out_traces
                extra_backend_traces.extend(ebt)
            traces_out[obj].extend(out_traces)
    if output == "list":
        traces = [t for tr in traces_out.values() for t in tr]
        traces_out = group_traces(*traces)
    return traces_out, autosize, ranges, extra_backend_traces


def get_row_cols(objs):
    """Return row_col dict with objs as keys and tuple (row,col) as values"""
    # pylint: disable=import-outside-toplevel

    from magpylib._src.obj_classes.class_Collection import Collection

    row_cols = {}
    for obj in objs:
        sub_objs = []
        for sub_obj in obj["objects"]:
            sub_objs.append(sub_obj)
            if isinstance(sub_obj, Collection):
                sub_objs.extend(sub_obj.children_all)
        for sub_obj in sub_objs:
            if sub_obj not in row_cols:
                row_cols[sub_obj] = []
            row_cols[sub_obj].extend([(obj["row"], obj["col"])])
    return row_cols


def get_frames(
    objs,
    colorsequence=None,
    zoom=1,
    title=None,
    animation=False,
    mag_arrows=False,
    extra_backend=False,
    **kwargs,
):
    """This is a helper function which generates frames with generic traces to be provided to
    the chosen backend. According to a certain zoom level, all three space direction will be equal
    and match the maximum of the ranges needed to display all objects, including their paths.
    """
    # infer title if necessary
    if objs:
        style = getattr(objs[0]["objects"][0], "style", None)
        label = getattr(style, "label", None)
        title = label if len(objs[0]["objects"]) == 1 else None
    else:
        title = "No objects to be displayed"

    # make sure the number of frames does not exceed the max frames and max frame rate
    # downsample if necessary
    obj_list_semi_flat = format_obj_input(
        [o["objects"] for o in objs], allow="sources+sensors+collections"
    )
    kwargs, animation, animation_kwargs = process_animation_kwargs(
        obj_list_semi_flat, animation=animation, **kwargs
    )
    path_indices = [-1]
    if animation:
        path_indices, exp, frame_duration = extract_animation_properties(
            obj_list_semi_flat, **animation_kwargs
        )

    # create frame for each path index or downsampled path index
    frames = []
    autosize = "return"
    title_str = title
    for i, ind in enumerate(path_indices):
        extra_backend_traces = []
        if animation:
            kwargs["style_path_frames"] = [ind]
            title = "Animation 3D - " if title is None else title
            title_str = f"""{title}path index: {ind+1:0{exp}d}"""
        traces, autosize_init, ranges, extra_backend_traces = draw_frame(
            objs,
            colorsequence,
            zoom,
            autosize=autosize,
            output="list",
            mag_arrows=mag_arrows,
            extra_backend=extra_backend,
            **kwargs,
        )
        if i == 0:  # get the dipoles and sensors autosize from first frame
            autosize = autosize_init
        frames.append(
            dict(
                data=traces,
                name=str(ind + 1),
                layout=dict(title=title_str),
                extra_backend_traces=extra_backend_traces,
            )
        )

    clean_legendgroups(frames)
    traces = [t for frame in frames for t in frame["data"]]
    ranges = get_scene_ranges(*traces, zoom=zoom)
    out = {
        "frames": frames,
        "ranges": ranges,
    }
    if animation:
        out.update(
            {
                "frame_duration": frame_duration,
                "path_indices": path_indices,
                "animation_slider": animation_kwargs["animation_slider"],
            }
        )
    return out

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
from magpylib._src.display.traces_base import make_Tetrahedron as make_BaseTetrahedron
from magpylib._src.display.traces_utility import draw_arrow_from_vertices
from magpylib._src.display.traces_utility import draw_arrowed_circle
from magpylib._src.display.traces_utility import draw_arrowed_line
from magpylib._src.display.traces_utility import get_flatten_objects_properties
from magpylib._src.display.traces_utility import get_rot_pos_from_path
from magpylib._src.display.traces_utility import getColorscale
from magpylib._src.display.traces_utility import getIntensity
from magpylib._src.display.traces_utility import MagpyMarkers
from magpylib._src.display.traces_utility import merge_mesh3d
from magpylib._src.display.traces_utility import merge_traces
from magpylib._src.display.traces_utility import place_and_orient_model3d
from magpylib._src.input_checks import check_excitations
from magpylib._src.style import get_style
from magpylib._src.utility import format_obj_input
from magpylib._src.utility import unit_prefix


def make_Line(
    current=0.0,
    vertices=((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    position=(0.0, 0.0, 0.0),
    orientation=None,
    color=None,
    style=None,
    **kwargs,
) -> dict:
    """
    Creates the plotly scatter3d parameters for a Line current in a dictionary based on the
    provided arguments
    """
    default_suffix = (
        f" ({unit_prefix(current)}A)"
        if current is not None
        else " (Current not initialized)"
    )
    name, name_suffix = get_name_and_suffix("Line", default_suffix, style)
    show_arrows = style.arrow.show
    arrow_size = style.arrow.size
    if show_arrows:
        vertices = draw_arrow_from_vertices(vertices, current, arrow_size)
    else:
        vertices = np.array(vertices).T
    if orientation is not None:
        vertices = orientation.apply(vertices.T).T
    x, y, z = (vertices.T + position).T
    line_width = style.arrow.width
    line = dict(
        type="scatter3d",
        x=x,
        y=y,
        z=z,
        name=f"""{name}{name_suffix}""",
        mode="lines",
        line_width=line_width,
        line_color=color,
    )
    return {**line, **kwargs}


def make_Loop(
    current=0.0,
    diameter=1.0,
    position=(0.0, 0.0, 0.0),
    vert=50,
    orientation=None,
    color=None,
    style=None,
    **kwargs,
):
    """
    Creates the plotly scatter3d parameters for a Loop current in a dictionary based on the
    provided arguments
    """
    default_suffix = (
        f" ({unit_prefix(current)}A)"
        if current is not None
        else " (Current not initialized)"
    )
    name, name_suffix = get_name_and_suffix("Loop", default_suffix, style)
    arrow_size = style.arrow.size if style.arrow.show else 0
    vertices = draw_arrowed_circle(current, diameter, arrow_size, vert)
    if orientation is not None:
        vertices = orientation.apply(vertices.T).T
    x, y, z = (vertices.T + position).T
    line_width = style.arrow.width
    circular = dict(
        type="scatter3d",
        x=x,
        y=y,
        z=z,
        name=f"""{name}{name_suffix}""",
        mode="lines",
        line_width=line_width,
        line_color=color,
    )
    return {**circular, **kwargs}


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

    default_suffix = ""
    name, name_suffix = get_name_and_suffix(
        f"{type(obj).__name__}", default_suffix, style
    )
    vertices = np.array([position])
    if orientation is not None:
        vertices = orientation.apply(vertices).T
    x, y, z = vertices
    trace = dict(
        type="scatter3d",
        x=x,
        y=y,
        z=z,
        name=f"""{name}{name_suffix}""",
        text=name,
        mode="markers+text",
        marker_size=10,
        marker_color=color,
        marker_symbol="diamond",
    )
    return {**trace, **kwargs}


def make_Dipole(
    moment=(0.0, 0.0, 1.0),
    position=(0.0, 0.0, 0.0),
    orientation=None,
    style=None,
    autosize=None,
    **kwargs,
) -> dict:
    """
    Creates the plotly mesh3d parameters for a Loop current in a dictionary based on the
    provided arguments
    """
    moment_mag = np.linalg.norm(moment)
    default_suffix = f" (moment={unit_prefix(moment_mag)}mT mm³)"
    name, name_suffix = get_name_and_suffix("Dipole", default_suffix, style)
    size = style.size
    if autosize is not None:
        size *= autosize
    dipole = make_BaseArrow(
        "plotly-dict", base=10, diameter=0.3 * size, height=size, pivot=style.pivot
    )
    nvec = np.array(moment) / moment_mag
    zaxis = np.array([0, 0, 1])
    cross = np.cross(nvec, zaxis)
    dot = np.dot(nvec, zaxis)
    n = np.linalg.norm(cross)
    t = np.arccos(dot)
    vec = -t * cross / n if n != 0 else (0, 0, 0)
    mag_orient = RotScipy.from_rotvec(vec)
    orientation = orientation * mag_orient
    mag = np.array((0, 0, 1))
    return _update_mag_mesh(
        dipole,
        name,
        name_suffix,
        mag,
        orientation,
        position,
        style,
        **kwargs,
    )


def make_Cuboid(
    mag=(0.0, 0.0, 1000.0),
    dimension=(1.0, 1.0, 1.0),
    position=(0.0, 0.0, 0.0),
    orientation=None,
    style=None,
    **kwargs,
) -> dict:
    """
    Creates the plotly mesh3d parameters for a Cuboid Magnet in a dictionary based on the
    provided arguments
    """
    d = [unit_prefix(d / 1000) for d in dimension]
    default_suffix = f" ({d[0]}m|{d[1]}m|{d[2]}m)"
    name, name_suffix = get_name_and_suffix("Cuboid", default_suffix, style)
    cuboid = make_BaseCuboid("plotly-dict", dimension=dimension)
    return _update_mag_mesh(
        cuboid,
        name,
        name_suffix,
        mag,
        orientation,
        position,
        style,
        **kwargs,
    )


def make_Cylinder(
    mag=(0.0, 0.0, 1000.0),
    base=50,
    diameter=1.0,
    height=1.0,
    position=(0.0, 0.0, 0.0),
    orientation=None,
    style=None,
    **kwargs,
) -> dict:
    """
    Creates the plotly mesh3d parameters for a Cylinder Magnet in a dictionary based on the
    provided arguments
    """
    d = [unit_prefix(d / 1000) for d in (diameter, height)]
    default_suffix = f" (D={d[0]}m, H={d[1]}m)"
    name, name_suffix = get_name_and_suffix("Cylinder", default_suffix, style)
    cylinder = make_BasePrism(
        "plotly-dict",
        base=base,
        diameter=diameter,
        height=height,
    )
    return _update_mag_mesh(
        cylinder,
        name,
        name_suffix,
        mag,
        orientation,
        position,
        style,
        **kwargs,
    )


def make_CylinderSegment(
    mag=(0.0, 0.0, 1000.0),
    dimension=(1.0, 2.0, 1.0, 0.0, 90.0),
    position=(0.0, 0.0, 0.0),
    orientation=None,
    vert=25,
    style=None,
    **kwargs,
):
    """
    Creates the plotly mesh3d parameters for a Cylinder Segment Magnet in a dictionary based on the
    provided arguments
    """
    d = [unit_prefix(d / (1000 if i < 3 else 1)) for i, d in enumerate(dimension)]
    default_suffix = f" (r={d[0]}m|{d[1]}m, h={d[2]}m, φ={d[3]}°|{d[4]}°)"
    name, name_suffix = get_name_and_suffix("CylinderSegment", default_suffix, style)
    cylinder_segment = make_BaseCylinderSegment(
        "plotly-dict", dimension=dimension, vert=vert
    )
    return _update_mag_mesh(
        cylinder_segment,
        name,
        name_suffix,
        mag,
        orientation,
        position,
        style,
        **kwargs,
    )


def make_Sphere(
    mag=(0.0, 0.0, 1000.0),
    vert=15,
    diameter=1,
    position=(0.0, 0.0, 0.0),
    orientation=None,
    style=None,
    **kwargs,
) -> dict:
    """
    Creates the plotly mesh3d parameters for a Sphere Magnet in a dictionary based on the
    provided arguments
    """
    default_suffix = f" (D={unit_prefix(diameter / 1000)}m)"
    name, name_suffix = get_name_and_suffix("Sphere", default_suffix, style)
    vert = min(max(vert, 3), 20)
    sphere = make_BaseEllipsoid("plotly-dict", vert=vert, dimension=[diameter] * 3)
    return _update_mag_mesh(
        sphere,
        name,
        name_suffix,
        mag,
        orientation,
        position,
        style,
        **kwargs,
    )

def make_Tetrahedron(
    mag=(0.0, 0.0, 1000.0),
    vertices=None,
    position=(0.0, 0.0, 0.0),
    orientation=None,
    style=None,
    **kwargs,
) -> dict:
    """
    Creates the plotly mesh3d parameters for a Sphere Magnet in a dictionary based on the
    provided arguments
    """
    name, name_suffix = get_name_and_suffix("Sphere", "", style)
    sphere = make_BaseTetrahedron("plotly-dict", vertices=vertices)
    return _update_mag_mesh(
        sphere,
        name,
        name_suffix,
        mag,
        orientation,
        position,
        style,
        **kwargs,
    )

def make_Pixels(positions, size=1) -> dict:
    """
    Creates the plotly mesh3d parameters for Sensor pixels based on pixel positions and chosen size
    For now, only "cube" shape is provided.
    """
    pixels = [
        make_BaseCuboid("plotly-dict", position=p, dimension=[size] * 3)
        for p in positions
    ]
    return merge_mesh3d(*pixels)


def make_Sensor(
    pixel=(0.0, 0.0, 0.0),
    dimension=(1.0, 1.0, 1.0),
    position=(0.0, 0.0, 0.0),
    orientation=None,
    color=None,
    style=None,
    autosize=None,
    **kwargs,
):
    """
    Creates the plotly mesh3d parameters for a Sensor object in a dictionary based on the
    provided arguments

    size_pixels: float, default=1
        A positive number. Adjusts automatic display size of sensor pixels. When set to 0,
        pixels will be hidden, when greater than 0, pixels will occupy half the ratio of the minimum
        distance between any pixel of the same sensor, equal to `size_pixel`.
    """
    pixel = np.array(pixel).reshape((-1, 3))
    default_suffix = (
        f""" ({'x'.join(str(p) for p in pixel.shape[:-1])} pixels)"""
        if pixel.ndim != 1
        else ""
    )
    name, name_suffix = get_name_and_suffix("Sensor", default_suffix, style)
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
    sensor = merge_mesh3d(*meshes_to_merge)
    return _update_mag_mesh(
        sensor, name, name_suffix, orientation=orientation, position=position, **kwargs
    )


def _update_mag_mesh(
    mesh_dict,
    name,
    name_suffix,
    magnetization=None,
    orientation=None,
    position=None,
    style=None,
    **kwargs,
):
    """
    Updates an existing plotly mesh3d dictionary of an object which has a magnetic vector. The
    object gets colorized, positioned and oriented based on provided arguments
    """
    if hasattr(style, "magnetization"):
        color = style.magnetization.color
        if magnetization is not None and style.magnetization.show:
            vertices = np.array([mesh_dict[k] for k in "xyz"]).T
            color_middle = color.middle
            if color.mode == "tricycle":
                color_middle = kwargs.get("color", None)
            elif color.mode == "bicolor":
                color_middle = False
            mesh_dict["colorscale"] = getColorscale(
                color_transition=color.transition,
                color_north=color.north,
                color_middle=color_middle,
                color_south=color.south,
            )
            mesh_dict["intensity"] = getIntensity(
                vertices=vertices,
                axis=magnetization,
            )
    mesh_dict = place_and_orient_model3d(
        model_kwargs=mesh_dict,
        orientation=orientation,
        position=position,
        showscale=False,
        name=f"{name}{name_suffix}",
    )
    return {**mesh_dict, **kwargs}


def get_name_and_suffix(default_name, default_suffix, style):
    """provides legend entry based on name and suffix"""
    name = default_name if style.label is None else style.label
    if style.description.show and style.description.text is None:
        name_suffix = default_suffix
    elif not style.description.show:
        name_suffix = ""
    else:
        name_suffix = f" ({style.description.text})"
    return name, name_suffix


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
    if obj._object_type in ("Cuboid", "Cylinder"):
        length = 1.8 * np.amax(obj.dimension)
    elif obj._object_type == "CylinderSegment":
        length = 1.8 * np.amax(obj.dimension[:3])  # d1,d2,h
    elif obj._object_type == 'Tetrahedron':
        v = obj.vertices
        v = np.insert(v, 3, v[0], axis=0)
        length = np.linalg.norm(np.diff(v, axis=0),axis=1).mean()
    else:
        length = 1.8 * obj.diameter  # Sphere
    length *= style.magnetization.size
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
    x, y, z = np.concatenate(points.swapaxes(1, 2)).T
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


def get_generic_traces(
    input_obj,
    color=None,
    autosize=None,
    legendgroup=None,
    showlegend=None,
    legendtext=None,
    mag_arrows=False,
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

    Sensor = _src.obj_classes.Sensor
    Cuboid = _src.obj_classes.Cuboid
    Cylinder = _src.obj_classes.Cylinder
    CylinderSegment = _src.obj_classes.CylinderSegment
    Sphere = _src.obj_classes.Sphere
    Tetrahedron = _src.obj_classes.Tetrahedron
    Dipole = _src.obj_classes.Dipole
    Loop = _src.obj_classes.Loop
    Line = _src.obj_classes.Line

    # parse kwargs into style and non style args
    style = get_style(input_obj, Config, **kwargs)
    kwargs = {k: v for k, v in kwargs.items() if not k.startswith("style")}
    kwargs["style"] = style
    style_color = getattr(style, "color", None)
    kwargs["color"] = style_color if style_color is not None else color
    kwargs["opacity"] = style.opacity
    legendgroup = f"{input_obj}" if legendgroup is None else legendgroup

    if hasattr(style, "magnetization"):
        if style.magnetization.show:
            check_excitations([input_obj])

    if hasattr(style, "arrow"):
        if style.arrow.show:
            check_excitations([input_obj])

    traces = []
    if isinstance(input_obj, MagpyMarkers):
        x, y, z = input_obj.markers.T
        marker = style.marker.as_dict()
        default_name = "Marker" if len(x) == 1 else "Markers"
        default_suffix = "" if len(x) == 1 else f" ({len(x)} points)"
        name, name_suffix = get_name_and_suffix(default_name, default_suffix, style)
        trace = dict(
            type="scatter3d",
            name=f"{name}{name_suffix}",
            x=x,
            y=y,
            z=z,
            marker=marker,
            mode="markers",
            opacity=style.opacity,
        )
        traces.append(trace)
    else:
        if isinstance(input_obj, Sensor):
            kwargs.update(
                dimension=getattr(input_obj, "dimension", style.size),
                pixel=getattr(input_obj, "pixel", (0.0, 0.0, 0.0)),
                autosize=autosize,
            )
            make_func = make_Sensor
        elif isinstance(input_obj, Cuboid):
            kwargs.update(
                mag=input_obj.magnetization,
                dimension=input_obj.dimension,
            )
            make_func = make_Cuboid
        elif isinstance(input_obj, Cylinder):
            base = 50
            kwargs.update(
                mag=input_obj.magnetization,
                diameter=input_obj.dimension[0],
                height=input_obj.dimension[1],
                base=base,
            )
            make_func = make_Cylinder
        elif isinstance(input_obj, CylinderSegment):
            vert = 50
            kwargs.update(
                mag=input_obj.magnetization,
                dimension=input_obj.dimension,
                vert=vert,
            )
            make_func = make_CylinderSegment
        elif isinstance(input_obj, Sphere):
            kwargs.update(
                mag=input_obj.magnetization,
                diameter=input_obj.diameter,
            )
            make_func = make_Sphere
        elif isinstance(input_obj, Tetrahedron):
            kwargs.update(
                mag=input_obj.magnetization,
                vertices=input_obj.vertices,
            )
            make_func = make_Tetrahedron
        elif isinstance(input_obj, Dipole):
            kwargs.update(
                moment=input_obj.moment,
                autosize=autosize,
            )
            make_func = make_Dipole
        elif isinstance(input_obj, Line):
            kwargs.update(
                vertices=input_obj.vertices,
                current=input_obj.current,
            )
            make_func = make_Line
        elif isinstance(input_obj, Loop):
            kwargs.update(
                diameter=input_obj.diameter,
                current=input_obj.current,
            )
            make_func = make_Loop
        elif getattr(input_obj, "children", None) is not None:
            make_func = None
        else:
            kwargs.update(obj=input_obj)
            make_func = make_DefaultTrace

        path_traces = []
        path_traces_extra = {}
        extra_model3d_traces = (
            style.model3d.data if style.model3d.data is not None else []
        )
        rots, poss, _ = get_rot_pos_from_path(input_obj, style.path.frames)
        for orient, pos in zip(rots, poss):
            if style.model3d.showdefault and make_func is not None:
                path_traces.append(
                    make_func(position=pos, orientation=orient, **kwargs)
                )
            for extr in extra_model3d_traces:
                if extr.show:
                    extr.update(extr.updatefunc())
                    if extr.backend == "generic":
                        trace3d = {"opacity": kwargs["opacity"]}
                        ttype = extr.constructor.lower()
                        obj_extr_trace = (
                            extr.kwargs() if callable(extr.kwargs) else extr.kwargs
                        )
                        obj_extr_trace = {"type": ttype, **obj_extr_trace}
                        if ttype == "scatter3d":
                            for k in ("marker", "line"):
                                trace3d["{k}_color"] = trace3d.get(
                                    f"{k}_color", kwargs["color"]
                                )
                        elif ttype == "mesh3d":
                            trace3d["showscale"] = trace3d.get("showscale", False)
                            if "facecolor" in obj_extr_trace:
                                ttype = "mesh3d_facecolor"
                            trace3d["color"] = trace3d.get("color", kwargs["color"])
                        else:
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
                        if ttype not in path_traces_extra:
                            path_traces_extra[ttype] = []
                        path_traces_extra[ttype].append(trace3d)
        trace = merge_traces(*path_traces)
        for ind, traces_extra in enumerate(path_traces_extra.values()):
            extra_model3d_trace = merge_traces(*traces_extra)
            label = (
                input_obj.style.label
                if input_obj.style.label is not None
                else str(type(input_obj).__name__)
            )
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
            traces.append(make_mag_arrows(input_obj, style, legendgroup, kwargs))

    return traces


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


def draw_frame(
    obj_list_semi_flat,
    colorsequence=None,
    zoom=0.0,
    autosize=None,
    output="dict",
    return_ranges=False,
    mag_arrows=False,
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
    if colorsequence is None:
        colorsequence = Config.display.colorsequence

    return_autosize = False
    Sensor = _src.obj_classes.Sensor
    Dipole = _src.obj_classes.Dipole
    traces_out = {}
    # dipoles and sensors use autosize, the trace building has to be put at the back of the queue.
    # autosize is calculated from the other traces overall scene range
    traces_to_resize = {}
    flat_objs_props = get_flatten_objects_properties(
        *obj_list_semi_flat, colorsequence=colorsequence
    )
    for obj, params in flat_objs_props.items():
        params.update(kwargs)
        if isinstance(obj, (Dipole, Sensor)):
            traces_to_resize[obj] = {**params}
            # temporary coordinates to be able to calculate ranges
            x, y, z = obj._position.T
            traces_out[obj] = [dict(x=x, y=y, z=z)]
        else:
            traces_out[obj] = get_generic_traces(obj, mag_arrows=mag_arrows, **params)
    traces = [t for tr in traces_out.values() for t in tr]
    ranges = get_scene_ranges(*traces, zoom=zoom)
    if autosize is None or autosize == "return":
        if autosize == "return":
            return_autosize = True
        autosize = np.mean(np.diff(ranges)) / Config.display.autosizefactor
    for obj, params in traces_to_resize.items():
        traces_out[obj] = get_generic_traces(
            obj, autosize=autosize, mag_arrows=mag_arrows, **params
        )
    if output == "list":
        traces = [t for tr in traces_out.values() for t in tr]
        traces_out = group_traces(*traces)
    res = (traces_out,)
    if return_autosize:
        res += (autosize,)
    if return_ranges:
        res += (ranges,)
    return res[0] if len(res) == 1 else res


def group_traces(*traces):
    """Group and merge mesh traces with similar properties. This drastically improves
    browser rendering performance when displaying a lot of mesh3d objects."""
    mesh_groups = {}
    common_keys = ["legendgroup", "opacity"]
    # TODO grouping does not dectect line_width vs line=dict(with=...)
    spec_keys = {"mesh3d": ["colorscale"], "scatter3d": ["marker", "line", "mode"]}
    for tr in traces:
        gr = [tr["type"]]
        for k in common_keys + spec_keys[tr["type"]]:
            try:
                v = tr.get(k, "")
            except AttributeError:
                v = getattr(tr, k, "")
            gr.append(str(v))
        gr = "".join(gr)
        if gr not in mesh_groups:
            mesh_groups[gr] = []
        mesh_groups[gr].append(tr)

    traces = []
    for key, gr in mesh_groups.items():
        if key.startswith("mesh3d") or key.startswith("scatter3d"):
            tr = [merge_traces(*gr)]
        else:
            tr = gr
        traces.extend(tr)
    return traces


def subdivide_mesh_by_facecolor(trace):
    """Subdivide a mesh into a list of meshes based on facecolor"""
    facecolor = trace["facecolor"]
    subtraces = []
    # pylint: disable=singleton-comparison
    facecolor[facecolor == None] = "black"
    for color in np.unique(facecolor):
        mask = facecolor == color
        new_trace = trace.copy()
        uniq = np.unique(np.hstack([trace[k][mask] for k in "ijk"]))
        new_inds = np.arange(len(uniq))
        mapping_ar = np.zeros(uniq.max() + 1, dtype=new_inds.dtype)
        mapping_ar[uniq] = new_inds
        for k in "ijk":
            new_trace[k] = mapping_ar[trace[k][mask]]
        for k in "xyz":
            new_trace[k] = new_trace[k][uniq]
        new_trace["color"] = color
        new_trace.pop("facecolor")
        subtraces.append(new_trace)
    return subtraces


def get_scene_ranges(*traces, zoom=1) -> np.ndarray:
    """
    Returns 3x2 array of the min and max ranges in x,y,z directions of input traces. Traces can be
    any plotly trace object or a dict, with x,y,z numbered parameters.
    """
    if traces:
        ranges = {k: [] for k in "xyz"}
        for t in traces:
            for k, v in ranges.items():
                v.extend(
                    [
                        np.nanmin(np.array(t[k], dtype=float)),
                        np.nanmax(np.array(t[k], dtype=float)),
                    ]
                )
        r = np.array([[np.nanmin(v), np.nanmax(v)] for v in ranges.values()])
        size = np.diff(r, axis=1)
        size[size == 0] = 1
        m = size.max() / 2
        center = r.mean(axis=1)
        ranges = np.array([center - m * (1 + zoom), center + m * (1 + zoom)]).T
    else:
        ranges = np.array([[-1.0, 1.0]] * 3)
    return ranges


def process_animation_kwargs(obj_list, animation=False, **kwargs):
    """Update animation kwargs"""
    markers = [o for o in obj_list if isinstance(o, MagpyMarkers)]
    flat_obj_list = format_obj_input([o for o in obj_list if o not in markers])
    flat_obj_list.extend(markers)
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
    anim_def.update({k[10:]: v for k, v in kwargs.items()})
    animation_kwargs = {f"animation_{k}": v for k, v in anim_def.as_dict().items()}
    kwargs = {k: v for k, v in kwargs.items() if not k.startswith("animation")}
    return kwargs, animation, animation_kwargs


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


def get_frames(
    objs,
    colorsequence=None,
    zoom=1,
    title=None,
    animation=False,
    mag_arrows=False,
    **kwargs,
):
    """This is a helper function which generates frames with generic traces to be provided to
    the chosen backend. According to a certain zoom level, all three space direction will be equal
    and match the maximum of the ranges needed to display all objects, including their paths.
    """
    # infer title if necessary
    if objs:
        style = getattr(objs[0], "style", None)
        label = getattr(style, "label", None)
        title = label if len(objs) == 1 else None
    else:
        title = "No objects to be displayed"

    # make sure the number of frames does not exceed the max frames and max frame rate
    # downsample if necessary
    kwargs, animation, animation_kwargs = process_animation_kwargs(
        objs, animation=animation, **kwargs
    )
    path_indices = [-1]
    if animation:
        path_indices, exp, frame_duration = extract_animation_properties(
            objs, **animation_kwargs
        )

    # create frame for each path index or downsampled path index
    frames = []
    autosize = "return"
    title_str = title
    for i, ind in enumerate(path_indices):
        if animation:
            kwargs["style_path_frames"] = [ind]
            title = "Animation 3D - " if title is None else title
            title_str = f"""{title}path index: {ind+1:0{exp}d}"""
        frame = draw_frame(
            objs,
            colorsequence,
            zoom,
            autosize=autosize,
            output="list",
            mag_arrows=mag_arrows,
            **kwargs,
        )
        if i == 0:  # get the dipoles and sensors autosize from first frame
            traces, autosize = frame
        else:
            traces = frame
        frames.append(
            dict(
                data=traces,
                name=str(ind + 1),
                layout=dict(title=title_str),
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

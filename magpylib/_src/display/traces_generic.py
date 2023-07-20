"""Generic trace drawing functionalities"""
# pylint: disable=C0302
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=too-many-nested-blocks
# pylint: disable=cyclic-import
import numbers
import warnings
from collections import Counter
from itertools import combinations
from itertools import cycle
from typing import Tuple

import numpy as np
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as RotScipy

import magpylib as magpy
from magpylib._src.defaults.defaults_classes import default_settings as Config
from magpylib._src.defaults.defaults_utility import ALLOWED_LINESTYLES
from magpylib._src.defaults.defaults_utility import ALLOWED_SYMBOLS
from magpylib._src.defaults.defaults_utility import linearize_dict
from magpylib._src.display.sensor_mesh import get_sensor_mesh
from magpylib._src.display.traces_base import make_Arrow as make_BaseArrow
from magpylib._src.display.traces_base import make_Cuboid as make_BaseCuboid
from magpylib._src.display.traces_base import (
    make_CylinderSegment as make_BaseCylinderSegment,
)
from magpylib._src.display.traces_base import make_Ellipsoid as make_BaseEllipsoid
from magpylib._src.display.traces_base import make_Prism as make_BasePrism
from magpylib._src.display.traces_base import make_Pyramid as make_BasePyramid
from magpylib._src.display.traces_base import make_Tetrahedron as make_BaseTetrahedron
from magpylib._src.display.traces_base import (
    make_TriangularMesh as make_BaseTriangularMesh,
)
from magpylib._src.display.traces_utility import draw_arrow_from_vertices
from magpylib._src.display.traces_utility import draw_arrow_on_circle
from magpylib._src.display.traces_utility import draw_arrowed_line
from magpylib._src.display.traces_utility import get_flatten_objects_properties
from magpylib._src.display.traces_utility import get_label
from magpylib._src.display.traces_utility import get_rot_pos_from_path
from magpylib._src.display.traces_utility import get_scene_ranges
from magpylib._src.display.traces_utility import getColorscale
from magpylib._src.display.traces_utility import getIntensity
from magpylib._src.display.traces_utility import group_traces
from magpylib._src.display.traces_utility import merge_mesh3d
from magpylib._src.display.traces_utility import merge_traces
from magpylib._src.display.traces_utility import place_and_orient_model3d
from magpylib._src.display.traces_utility import slice_mesh_from_colorscale
from magpylib._src.display.traces_utility import triangles_area
from magpylib._src.style import DefaultMarkers
from magpylib._src.utility import format_obj_input
from magpylib._src.utility import unit_prefix


class MagpyMarkers:
    """A class that stores markers 3D-coordinates."""

    def __init__(self, *markers):
        self._style = DefaultMarkers()
        self.markers = np.array(markers)

    @property
    def style(self):
        """Style property"""
        return self._style

    def get_trace(self, **kwargs):
        """Create the plotly mesh3d parameters for a Sensor object in a dictionary based on the
        provided arguments."""
        style = self.style
        x, y, z = self.markers.T
        marker_kwargs = {
            f"marker_{k}": v
            for k, v in style.marker.as_dict(flatten=True, separator="_").items()
        }
        marker_kwargs["marker_color"] = (
            style.marker.color if style.marker.color is not None else style.color
        )
        trace = {
            "type": "scatter3d",
            "x": x,
            "y": y,
            "z": z,
            "mode": "markers",
            **marker_kwargs,
            **kwargs,
        }
        default_name = "Marker" if len(x) == 1 else "Markers"
        default_suffix = "" if len(x) == 1 else f" ({len(x)} points)"
        trace["name"] = get_label(self, default_suffix, default_name)
        return trace


def make_DefaultTrace(obj, **kwargs) -> dict:
    """
    Creates the plotly scatter3d parameters for an object with no specifically supported
    representation. The object will be represented by a scatter point and text above with object
    name.
    """
    style = obj.style
    trace = {
        "type": "scatter3d",
        "x": [0.0],
        "y": [0.0],
        "z": [0.0],
        "mode": "markers+text",
        "marker_size": 10,
        "marker_color": style.color,
        "marker_symbol": "diamond",
    }
    trace["name"] = get_label(obj)
    trace["text"] = trace["name"]
    return {**trace, **kwargs}


def make_Line(obj, **kwargs) -> dict:
    """
    Creates the plotly scatter3d parameters for a Line current in a dictionary based on the
    provided arguments.
    """
    style = obj.style
    default_suffix = (
        f" ({unit_prefix(obj.current)}A)"
        if obj.current is not None
        else " (Current not initialized)"
    )
    traces = []
    for kind in ("arrow", "line"):
        kind_style = getattr(style, kind)
        if kind_style.show:
            color = style.color if kind_style.color is None else kind_style.color
            if kind == "arrow":
                x, y, z = draw_arrow_from_vertices(
                    obj.vertices,
                    obj.current,
                    kind_style.size,
                    arrow_pos=style.arrow.offset,
                    include_line=False,
                ).T
            else:
                x, y, z = obj.vertices.T
            trace = {
                "type": "scatter3d",
                "x": x,
                "y": y,
                "z": z,
                "mode": "lines",
                "line_width": kind_style.width,
                "line_dash": kind_style.style,
                "line_color": color,
            }
            trace["name"] = get_label(obj, default_suffix=default_suffix)
            traces.append({**trace, **kwargs})
    return traces


def make_Loop(
    obj,
    style=None,
    vert_num=72,
    **kwargs,
):
    """
    Creates the plotly scatter3d parameters for a Loop current in a dictionary based on the
    provided arguments.
    """
    style = obj.style
    default_suffix = (
        f" ({unit_prefix(obj.current)}A)"
        if obj.current is not None
        else " (Current not initialized)"
    )
    traces = []
    for kind in ("arrow", "line"):
        kind_style = getattr(style, kind)
        if kind_style.show:
            color = style.color if kind_style.color is None else kind_style.color

            if kind == "arrow":
                angle_pos_deg = 360 * np.round(style.arrow.offset * vert_num) / vert_num
                vertices = draw_arrow_on_circle(
                    np.sign(obj.current),
                    obj.diameter,
                    style.arrow.size,
                    angle_pos_deg=angle_pos_deg,
                )
                x, y, z = vertices.T
            else:
                t = np.linspace(0, 2 * np.pi, vert_num)
                x = np.cos(t) * obj.diameter / 2
                y = np.sin(t) * obj.diameter / 2
                z = np.zeros(x.shape)
            trace = {
                "type": "scatter3d",
                "x": x,
                "y": y,
                "z": z,
                "mode": "lines",
                "line_width": kind_style.width,
                "line_dash": kind_style.style,
                "line_color": color,
            }
            trace["name"] = get_label(obj, default_suffix=default_suffix)
            traces.append({**trace, **kwargs})
    return traces


def make_Dipole(obj, autosize=None, **kwargs) -> dict:
    """
    Create the plotly mesh3d parameters for a dipole in a dictionary based on the
    provided arguments.
    """
    style = obj.style
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
        color=style.color,
    )
    default_suffix = f" (moment={unit_prefix(moment_mag)}mT mm³)"
    trace["name"] = get_label(obj, default_suffix=default_suffix)
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
    trace = place_and_orient_model3d(trace, orientation=mag_orient, **kwargs)
    return {**trace, **kwargs}


def get_closest_vertices(faces_subsets, vertices):
    """Get closest pairs of points between disconnected subsets of faces indices"""
    nparts = len(faces_subsets)
    inds_subsets = [np.unique(v) for v in faces_subsets]
    closest_verts_list = []
    if nparts > 1:
        connected = [np.min(inds_subsets[0])]
        while len(connected) < nparts:
            prev_min = float("inf")
            for i in connected:
                for j in range(nparts):
                    if j not in connected:
                        tr1, tr2 = inds_subsets[i], inds_subsets[j]
                        c1, c2 = vertices[tr1], vertices[tr2]
                        dist = distance.cdist(c1, c2)
                        i1, i2 = divmod(dist.argmin(), dist.shape[1])
                        min_dist = dist[i1, i2]
                        if min_dist < prev_min:
                            prev_min = min_dist
                            closest_verts = [c1[i1], c2[i2]]
                            connected_ind = j
            connected.append(connected_ind)
            closest_verts_list.append(closest_verts)
    return np.array(closest_verts_list)


def make_Cuboid(obj, **kwargs) -> dict:
    """
    Create the plotly mesh3d parameters for a Cuboid Magnet in a dictionary based on the
    provided arguments.
    """
    style = obj.style
    if obj.dimension is None:
        trace = {"type": "scatter3d", "x": [0], "y": [0], "z": [0]}
        default_suffix = " (no dimension)"
    else:
        trace = make_BaseCuboid(
            "plotly-dict", dimension=obj.dimension, color=style.color
        )
        d = [unit_prefix(d / 1000) for d in obj.dimension]
        default_suffix = f" ({d[0]}m|{d[1]}m|{d[2]}m)"
    trace["name"] = get_label(obj, default_suffix=default_suffix)
    return {**trace, **kwargs}


def make_Cylinder(obj, base=50, **kwargs) -> dict:
    """
    Create the plotly mesh3d parameters for a Cylinder Magnet in a dictionary based on the
    provided arguments.
    """
    style = obj.style
    if obj.dimension is None:
        trace = {"type": "scatter3d", "x": [0], "y": [0], "z": [0]}
        default_suffix = " (no dimension)"
    else:
        diameter, height = obj.dimension
        d = [unit_prefix(d / 1000) for d in (diameter, height)]
        trace = make_BasePrism(
            "plotly-dict",
            base=base,
            diameter=diameter,
            height=height,
            color=style.color,
        )
        default_suffix = f" (D={d[0]}m, H={d[1]}m)"
    trace["name"] = get_label(obj, default_suffix=default_suffix)
    return {**trace, **kwargs}


def make_CylinderSegment(obj, vertices=25, **kwargs):
    """
    Create the plotly mesh3d parameters for a Cylinder Segment Magnet in a dictionary based on the
    provided arguments.
    """
    style = obj.style
    if obj.dimension is None:
        trace = {"type": "scatter3d", "x": [0], "y": [0], "z": [0]}
        default_suffix = " (no dimension)"
    else:
        d = [
            unit_prefix(d / (1000 if i < 3 else 1)) for i, d in enumerate(obj.dimension)
        ]
        trace = make_BaseCylinderSegment(
            "plotly-dict", dimension=obj.dimension, vert=vertices, color=style.color
        )
        default_suffix = f" (r={d[0]}m|{d[1]}m, h={d[2]}m, φ={d[3]}°|{d[4]}°)"
    trace["name"] = get_label(obj, default_suffix=default_suffix)
    return {**trace, **kwargs}


def make_Sphere(obj, vertices=15, **kwargs) -> dict:
    """
    Create the plotly mesh3d parameters for a Sphere Magnet in a dictionary based on the
    provided arguments.
    """
    style = obj.style

    if obj.diameter is None:
        trace = {"type": "scatter3d", "x": [0], "y": [0], "z": [0]}
        default_suffix = " (no dimension)"
    else:
        vertices = min(max(vertices, 3), 20)
        trace = make_BaseEllipsoid(
            "plotly-dict",
            vert=vertices,
            dimension=[obj.diameter] * 3,
            color=style.color,
        )
        default_suffix = f" (D={unit_prefix(obj.diameter / 1000)}m)"
    trace["name"] = get_label(obj, default_suffix=default_suffix)
    return {**trace, **kwargs}


def make_Tetrahedron(obj, **kwargs) -> dict:
    """
    Create the plotly mesh3d parameters for a Tetrahedron Magnet in a dictionary based on the
    provided arguments.
    """
    style = obj.style
    trace = make_BaseTetrahedron(
        "plotly-dict", vertices=obj.vertices, color=style.color
    )
    trace["name"] = get_label(obj)
    return {**trace, **kwargs}


def make_triangle_orientations(obj, **kwargs) -> dict:
    """
    Create the plotly mesh3d parameters for a triangle orientation cone or arrow3d in a dictionary
    based on the provided arguments.
    """
    # pylint: disable=protected-access
    style = obj.style
    orient = style.orientation
    size = orient.size
    symbol = orient.symbol
    offset = orient.offset
    color = style.color if orient.color is None else orient.color
    vertices = obj.mesh if hasattr(obj, "mesh") else [obj.vertices]
    traces = []
    for vert in vertices:
        vec = np.cross(vert[1] - vert[0], vert[2] - vert[1])
        nvec = vec / np.linalg.norm(vec)
        # arrow length proportional to square root of triangle
        length = np.sqrt(triangles_area(np.expand_dims(vert, axis=0))[0]) * 0.2
        zaxis = np.array([0, 0, 1])
        cross = np.cross(nvec, zaxis)
        n = np.linalg.norm(cross)
        if n == 0:
            n = 1
            cross = np.array([-np.sign(nvec[-1]), 0, 0])
        dot = np.dot(nvec, zaxis)
        t = np.arccos(dot)
        vec = -t * cross / n
        orient = RotScipy.from_rotvec(vec)
        make_fn = make_BasePyramid if symbol == "cone" else make_BaseArrow
        vmean = np.mean(vert, axis=0)
        vmean -= (1 - offset) * length * nvec * size
        tr = make_fn(
            "plotly-dict",
            base=10,
            diameter=0.5 * size * length,
            height=size * length,
            pivot="tail",
            color=color,
            position=vmean,
            orientation=orient,
            **kwargs,
        )
        traces.append(tr)
    trace = merge_mesh3d(*traces)
    return trace


def make_mesh_lines(obj, mode, **kwargs):
    """Draw mesh lines and vertices"""
    # pylint: disable=protected-access
    kwargs.pop("color", None)
    legendgroup = kwargs.pop("legendgroup", obj)
    style = obj.style
    mesh = getattr(style.mesh, mode)
    marker, line = mesh.marker, mesh.line
    tr, vert = obj.faces, obj.vertices
    if mode == "disconnected":
        subsets = obj.get_faces_subsets()
        lines = get_closest_vertices(subsets, vert)
    else:
        if mode == "selfintersecting":
            tr = obj.faces[obj.get_selfintersecting_faces()]
        edges = np.concatenate([tr[:, 0:2], tr[:, 1:3], tr[:, ::2]], axis=0)
        if mode == "open":
            edges = obj.get_open_edges()
        else:
            edges = np.unique(edges, axis=0)
        lines = vert[edges]

    if lines.size == 0:
        return {}
    lines = np.insert(lines, 2, None, axis=1).reshape(-1, 3)
    traces = []
    x, y, z = lines.T
    trace = {
        "type": "scatter3d",
        "x": x,
        "y": y,
        "z": z,
        "marker_color": marker.color,
        "marker_size": marker.size,
        "marker_symbol": marker.symbol,
        "line_color": line.color,
        "line_width": line.width,
        "line_dash": line.style,
        "legendgroup": f"{legendgroup} - {mode}edges",
        "name_suffix": f" - {mode}-edges",
        "name": get_label(obj),
    }
    traces.append(trace)
    return {**merge_traces(*traces)[0], **kwargs}


def make_Triangle(obj, **kwargs) -> dict:
    """
    Creates the plotly mesh3d parameters for a Trianglular facet in a dictionary based on the
    provided arguments.
    """
    style = obj.style
    vert = obj.vertices
    vec = np.cross(vert[1] - vert[0], vert[2] - vert[1])
    faces = np.array([[0, 1, 2]])
    # if magnetization is normal to the triangle, add a second triangle slightly above to enable
    # proper color gradient visualization. Otherwise only the middle color is shown.
    if np.all(np.cross(obj.magnetization, vec) == 0):
        epsilon = 1e-3 * vec
        vert = np.concatenate([vert - epsilon, vert + epsilon])
        side_faces = [
            [0, 1, 3],
            [1, 2, 4],
            [2, 0, 5],
            [1, 4, 3],
            [2, 5, 4],
            [0, 3, 5],
        ]
        faces = np.concatenate([faces, [[3, 4, 5]], side_faces])

    trace = make_BaseTriangularMesh(
        "plotly-dict", vertices=vert, faces=faces, color=style.color
    )
    trace["name"] = get_label(obj)
    traces = [{**trace, **kwargs}]
    if style.orientation.show:
        traces.append(make_triangle_orientations(obj, **kwargs))
    return traces


def make_TriangularMesh_single(obj, **kwargs) -> dict:
    """
    Creates the plotly mesh3d parameters for a Trianglular facet mesh in a dictionary based on the
    provided arguments.
    """
    style = obj.style
    trace = make_BaseTriangularMesh(
        "plotly-dict", vertices=obj.vertices, faces=obj.faces, color=style.color
    )
    ntri = len(obj.faces)
    default_suffix = f" ({ntri} face{'s'[:ntri^1]})"
    trace["name"] = get_label(obj, default_suffix=default_suffix)
    # make edges sharper in plotly
    trace.update(flatshading=True, lighting_facenormalsepsilon=0, lighting_ambient=0.7)
    return {**trace, **kwargs}


def make_TriangularMesh(obj, **kwargs) -> dict:
    """
    Creates the plotly mesh3d parameters for a Trianglular facet mesh in a dictionary based on the
    provided arguments.
    """
    # pylint: disable=protected-access
    style = obj.style
    is_disconnected = False
    for mode in ("open", "disconnected", "selfintersecting"):
        show_mesh = getattr(style.mesh, mode).show
        if mode == "open" and show_mesh:
            if obj.status_open is None:
                warnings.warn(
                    f"Unchecked open mesh status in {obj!r} detected, before attempting "
                    "to show potential open edges, which may take a while to compute "
                    "when the mesh has many faces, now applying operation..."
                )
                obj.check_open()
        elif mode == "disconnected" and show_mesh:
            if obj.status_disconnected is None:
                warnings.warn(
                    f"Unchecked disconnected mesh status in {obj!r} detected, before "
                    "attempting to show possible disconnected parts, which may take a while "
                    "to compute when the mesh has many faces, now applying operation..."
                )
            is_disconnected = obj.check_disconnected()
        elif mode == "selfintersecting":
            if obj._status_selfintersecting is None:
                warnings.warn(
                    f"Unchecked selfintersecting mesh status in {obj!r} detected, before "
                    "attempting to show possible disconnected parts, which may take a while "
                    "to compute when the mesh has many faces, now applying operation..."
                )
                obj.check_selfintersecting()

    if is_disconnected:
        tria_orig = obj._faces
        obj.style.magnetization.mode = "arrow"
        traces = []
        for ind, (tri, dis_color) in enumerate(
            zip(
                obj.get_faces_subsets(),
                cycle(obj.style.mesh.disconnected.colorsequence),
            )
        ):
            # temporary mutate faces from subset
            obj._faces = tri
            obj.style.magnetization.show = False
            tr = make_TriangularMesh_single(obj, **{**kwargs, "color": dis_color})
            # match first group with path scatter trace
            lg_suff = "" if ind == 0 else f"- part_{ind+1:02d}"
            tr["legendgroup"] = f"{kwargs.get('legendgroup', obj)}{lg_suff}"
            tr["name_suffix"] = f" - part_{ind+1:02d}"
            traces.append(tr)
            if style.orientation.show:
                traces.append(
                    make_triangle_orientations(
                        obj,
                        **{**kwargs, "legendgroup": tr["legendgroup"]},
                    )
                )
        obj._faces = tria_orig
    else:
        traces = [make_TriangularMesh_single(obj, **kwargs)]
        if style.orientation.show:
            traces.append(
                make_triangle_orientations(
                    obj,
                    **kwargs,
                )
            )
    for mode in ("grid", "open", "disconnected", "selfintersecting"):
        if getattr(style.mesh, mode).show:
            trace = make_mesh_lines(obj, mode, **kwargs)
            if trace:
                traces.append(trace)
    return traces


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


def make_Sensor(obj, autosize=None, **kwargs):
    """
    Create the plotly mesh3d parameters for a Sensor object in a dictionary based on the
    provided arguments.

    size_pixels: float, default=1
        A positive number. Adjusts automatic display size of sensor pixels. When set to 0,
        pixels will be hidden, when greater than 0, pixels will occupy half the ratio of the minimum
        distance between any pixel of the same sensor, equal to `size_pixel`.
    """
    style = obj.style
    dimension = getattr(obj, "dimension", style.size)
    pixel = obj.pixel
    pixel = np.unique(np.array(pixel).reshape((-1, 3)), axis=0)
    style_arrows = style.arrows.as_dict(flatten=True, separator="_")
    sensor = get_sensor_mesh(**style_arrows, center_color=style.color)
    vertices = np.array([sensor[k] for k in "xyz"]).T
    if style.color is not None:
        sensor["facecolor"][sensor["facecolor"] == "rgb(238,238,238)"] = style.color
    dim = np.array(
        [dimension] * 3 if isinstance(dimension, (float, int)) else dimension[:3],
        dtype=float,
    )
    no_pix = pixel.shape[0] == 1 and (pixel == 0).all()
    one_pix = pixel.shape[0] == 1 and not (pixel == 0).all()
    if autosize is not None:
        dim *= autosize
    if no_pix:
        dim_ext = dim
    else:
        if one_pix:
            pixel = np.concatenate([[[0, 0, 0]], pixel])
        hull_dim = pixel.max(axis=0) - pixel.min(axis=0)
        dim_ext = max(np.mean(dim), np.min(hull_dim))
    cube_mask = (vertices < 1).all(axis=1)
    vertices[cube_mask] = 0 * vertices[cube_mask]
    vertices[~cube_mask] = dim_ext * vertices[~cube_mask]
    vertices /= 2  # sensor_mesh vertices are of length 2
    x, y, z = vertices.T
    sensor.update(x=x, y=y, z=z)
    meshes_to_merge = [sensor]
    if not no_pix:
        pixel_color = style.pixel.color
        pixel_size = style.pixel.size
        combs = np.array(list(combinations(pixel, 2)))
        vecs = np.diff(combs, axis=1)
        dists = np.linalg.norm(vecs, axis=2)
        min_dist = np.min(dists)
        pixel_dim = dim_ext / 5 if min_dist == 0 else min_dist / 2
        if pixel_size > 0:
            pixel_dim *= pixel_size
            poss = pixel[1:] if one_pix else pixel
            pixels_mesh = make_Pixels(positions=poss, size=pixel_dim)
            pixels_mesh["facecolor"] = np.repeat(pixel_color, len(pixels_mesh["i"]))
            meshes_to_merge.append(pixels_mesh)
        hull_pos = 0.5 * (pixel.max(axis=0) + pixel.min(axis=0))
        hull_dim[hull_dim == 0] = pixel_dim / 2
        hull_mesh = make_BaseCuboid(
            "plotly-dict", position=hull_pos, dimension=hull_dim
        )
        hull_mesh["facecolor"] = np.repeat(style.color, len(hull_mesh["i"]))
        meshes_to_merge.append(hull_mesh)
    trace = merge_mesh3d(*meshes_to_merge)
    default_suffix = (
        f" ({'x'.join(str(p) for p in obj.pixel.shape[:-1])} pixels)"
        if obj.pixel.ndim != 1
        else f" ({pixel[1:].shape[0]} pixel)"
        if one_pix
        else ""
    )
    trace["name"] = get_label(obj, default_suffix=default_suffix)
    return {**trace, **kwargs}


def update_magnet_mesh(
    mesh_dict, mag_style=None, magnetization=None, color_slicing=False
):
    """
    Updates an existing plotly mesh3d dictionary of an object which has a magnetic vector. The
    object gets colorized, positioned and oriented based on provided arguments.
    Slicing allows for matplotlib to show colorgradients approximations by slicing the mesh into
    the colorscales colors, remesh it and merge with assigning facecolor for each part.
    """
    mag_color = mag_style.color
    if magnetization is not None and mag_style.show:
        vertices = np.array([mesh_dict[k] for k in "xyz"]).T
        color_middle = mag_color.middle
        if mag_color.mode == "tricycle":
            color_middle = mesh_dict["color"]
        elif mag_color.mode == "bicolor":
            color_middle = False
        ct = mag_color.transition
        cs = getColorscale(
            color_transition=0 if color_slicing else ct,
            color_north=mag_color.north,
            color_middle=color_middle,
            color_south=mag_color.south,
        )
        if color_slicing:
            tr = slice_mesh_from_colorscale(mesh_dict, magnetization, cs)
            mesh_dict.update(tr)
        else:
            mesh_dict["colorscale"] = cs
            mesh_dict["intensity"] = getIntensity(
                vertices=vertices,
                axis=magnetization,
            )
        mesh_dict["showscale"] = False
        mesh_dict.pop("color_slicing", None)
    return mesh_dict


def make_mag_arrows(obj, pos_orient_inds):
    """draw direction of magnetization of faced magnets

    Parameters
    ----------
    - obj: object with magnetization vector to be drawn
    - colors: colors of faced_objects
    - show_path(bool or int): draw on every position where object is displayed
    """
    # pylint: disable=protected-access

    # vector length, color and magnetization
    style = obj.style
    if hasattr(obj, "diameter"):
        length = obj.diameter  # Sphere
    elif isinstance(obj, magpy.misc.Triangle):
        length = np.amax(obj.vertices) - np.amin(obj.vertices)
    elif hasattr(obj, "mesh"):
        length = np.amax(np.ptp(obj.mesh.reshape(-1, 3), axis=0))
    elif hasattr(obj, "vertices"):
        length = np.amax(np.ptp(obj.vertices, axis=0))
    else:  # Cuboid, Cylinder, CylinderSegment
        length = np.amax(obj.dimension[:3])
    length *= 1.8 * style.magnetization.size
    mag = obj.magnetization
    # collect all draw positions and directions
    points = []
    for ind in pos_orient_inds:
        pos = getattr(obj, "_barycenter", obj._position)[ind]
        direc = mag / (np.linalg.norm(mag) + 1e-6) * length
        vec = obj._orientation[ind].apply(direc)
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
        "line_color": style.color,
        "opacity": style.opacity,
        "x": x,
        "y": y,
        "z": z,
        "showlegend": False,
    }
    return trace


def make_path(input_obj):
    """draw obj path based on path style properties"""
    style = input_obj.style
    x, y, z = np.array(input_obj.position).T
    txt_kwargs = (
        {"mode": "markers+text+lines", "text": list(range(len(x)))}
        if style.path.numbering
        else {"mode": "markers+lines"}
    )
    marker = style.path.marker.as_dict()
    marker["symbol"] = marker["symbol"]
    marker["color"] = style.color if marker["color"] is None else marker["color"]
    line = style.path.line.as_dict()
    line["dash"] = line["style"]
    line["color"] = style.color if line["color"] is None else line["color"]
    line = {k: v for k, v in line.items() if k != "style"}
    scatter_path = {
        "type": "scatter3d",
        "x": x,
        "y": y,
        "z": z,
        "name": f"Path: {input_obj}",
        "showlegend": False,
        **{f"marker_{k}": v for k, v in marker.items()},
        **{f"line_{k}": v for k, v in line.items()},
        **txt_kwargs,
        "opacity": style.opacity,
    }
    return scatter_path


def get_trace2D_dict(
    BH,
    *,
    field_str,
    coords_str,
    obj_lst_str,
    frame_focus_inds,
    frames_indices,
    mode,
    label_suff,
    color,
    symbol,
    linestyle,
    **kwargs,
):
    """return a 2d trace based on field and parameters"""
    coords_inds = ["xyz".index(k) for k in coords_str]
    y = BH.T[list(coords_inds)]
    if len(coords_inds) == 1:
        y = y[0]
    else:
        y = np.linalg.norm(y, axis=0)
    marker_size = np.array([2] * len(frames_indices))
    marker_size[frame_focus_inds] = 15
    title = f"{field_str}{''.join(coords_str)}"
    trace = {
        "mode": "lines+markers",
        "legendgrouptitle_text": f"{title}"
        + (f" ({label_suff})" if label_suff else ""),
        "text": mode,
        "hovertemplate": (
            "<b>Path index</b>: %{x}    "
            f"<b>{title}</b>: " + "%{y:.3s}T<br>"
            f"<b>{'sources'}</b>:<br>{obj_lst_str['sources']}<br>"
            f"<b>{'sensors'}</b>:<br>{obj_lst_str['sensors']}"
            # "<extra></extra>",
        ),
        "x": frames_indices,
        "y": y[frames_indices],
        "line_dash": linestyle,
        "line_color": color,
        "marker_size": marker_size,
        "marker_color": color,
        "marker_symbol": symbol,
        "showlegend": True,
        "legendgroup": f"{title}{label_suff}",
        **kwargs,
    }
    return trace


def get_generic_traces_2D(
    *,
    objects,
    output=("Bx", "By", "Bz"),
    row=None,
    col=None,
    sumup=True,
    pixel_agg=None,
    style_path_frames=None,
    flat_objs_props=None,
):
    """draws and animates sensor values over a path in a subplot"""
    # pylint: disable=import-outside-toplevel
    from magpylib._src.fields.field_wrap_BH import getBH_level2

    sources = format_obj_input(objects, allow="sources+collections")
    sources = [
        s
        for s in sources
        if not (isinstance(s, magpy.Collection) and not s.sources_all)
    ]
    sensors = format_obj_input(objects, allow="sensors+collections")
    sensors = [
        sub_s
        for s in sensors
        for sub_s in (s.sensors_all if isinstance(s, magpy.Collection) else [s])
    ]

    if not isinstance(output, (list, tuple)):
        output = [output]
    output_params = {}
    for out, linestyle in zip(output, cycle(ALLOWED_LINESTYLES[:6])):
        field_str, *coords_str = out
        if not coords_str:
            coords_str = list("xyz")
        if field_str not in ("B", "H") and set(coords_str).difference(set("xyz")):
            raise ValueError(
                "The `output` parameter must start with 'B' or 'H' "
                "and be followed by a combination of 'x', 'y', 'z' (e.g. 'Bxy' or ('Bxy', 'Hz') )"
                f"\nreceived {out!r} instead"
            )
        output_params[out] = {
            "field_str": field_str,
            "coords_str": coords_str,
            "linestyle": linestyle,
        }
    BH_array = getBH_level2(
        sources,
        sensors,
        sumup=sumup,
        squeeze=False,
        field=field_str,
        pixel_agg=pixel_agg,
        output="ndarray",
    )
    BH_array = BH_array.swapaxes(1, 2)  # swap axes to have sensors first, path second

    frames_indices = np.arange(0, BH_array.shape[2])
    frame_focus_inds = [-1] if style_path_frames is None else style_path_frames
    if isinstance(frame_focus_inds, numbers.Number):
        # pylint: disable=invalid-unary-operand-type
        frame_focus_inds = frames_indices[::-style_path_frames]

    def get_obj_list_str(objs):
        if len(objs) < 8:
            obj_lst_str = "<br>".join(f" - {s}" for s in objs)
        else:
            counts = Counter(s.__class__.__name__ for s in objs)
            obj_lst_str = "<br>".join(f" {v}x {k}" for k, v in counts.items())
        return obj_lst_str

    def get_label_and_color(obj):
        props = flat_objs_props.get(obj, {})
        style = props.get("style", None)
        style = obj.style
        label = getattr(style, "label", None)
        label = repr(obj) if not label else label
        color = getattr(style, "color", None)
        return label, color

    obj_lst_str = {
        "sources": get_obj_list_str(sources),
        "sensors": get_obj_list_str(sensors),
    }
    mode = "sources" if sumup else "sensors"

    traces = []
    for src_ind, src in enumerate(sources):
        if src_ind == 1 and sumup:
            break
        label_src, color_src = get_label_and_color(src)
        symbols = cycle(ALLOWED_SYMBOLS[:6])
        for sens_ind, sens in enumerate(sensors):
            label_sens, color_sens = get_label_and_color(sens)
            label_suff = label_sens
            if mode == "sensors":
                label, color = label_src, color_src
            else:
                label_suff = (
                    f"{label_src}" if len(sources) == 1 else f"{len(sources)} sources"
                )
                label, color = label_sens, color_sens
            num_of_pix = (
                len(sens.pixel.reshape(-1, 3))
                if (not isinstance(sens, magpy.Collection)) and sens.pixel.ndim != 1
                else 1
            )
            pix_suff = ""
            num_of_pix_to_show = 1 if pixel_agg else num_of_pix
            for pix_ind in range(num_of_pix_to_show):
                symbol = next(symbols)
                BH = BH_array[src_ind, sens_ind, :, pix_ind]
                if num_of_pix > 1:
                    if pixel_agg:
                        pix_suff = f" ({num_of_pix} pixels {pixel_agg})"
                    else:
                        pix_suff = f" (pixel {pix_ind})"
                for param in output_params.values():
                    traces.append(
                        get_trace2D_dict(
                            BH,
                            **param,
                            obj_lst_str=obj_lst_str,
                            frame_focus_inds=frame_focus_inds,
                            frames_indices=frames_indices,
                            mode=mode,
                            label_suff=label_suff,
                            name=f"{label}{pix_suff}",
                            color=color,
                            symbol=symbol,
                            type="scatter",
                            row=row,
                            col=col,
                        )
                    )
    return traces


def get_generic_traces(
    input_obj,
    autosize=None,
    legendgroup=None,
    legendtext=None,
    supports_colorgradient=True,
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
    # pylint: disable=import-outside-toplevel

    style = input_obj.style
    is_mag_arrows = False
    is_mag = hasattr(input_obj, "magnetization") and hasattr(style, "magnetization")
    if is_mag and style.magnetization.show:
        mag = style.magnetization
        if mag.mode == "auto":
            mag.mode = "color"  # if mag_color_grad_apt else "arrow"
        is_mag_arrows = "arrow" in mag.mode
        mag.show = "color" in mag.mode

    make_func = getattr(input_obj, "get_trace", None)
    make_func_kwargs = {"legendgroup": legendgroup, **kwargs}
    if getattr(input_obj, "_autosize", False):
        make_func_kwargs["autosize"] = autosize

    all_generic_traces = []
    traces_generic = []
    path_traces_extra_non_generic_backend = []
    has_path = hasattr(input_obj, "position") and hasattr(input_obj, "orientation")
    if not has_path and make_func is not None:
        tr = make_func(**make_func_kwargs)
        tr["row"] = row
        tr["col"] = col
        out = {"generic": [tr]}
        if extra_backend:
            out.update({extra_backend: path_traces_extra_non_generic_backend})
        return out

    orientations, positions, pos_orient_inds = get_rot_pos_from_path(
        input_obj, style.path.frames
    )
    if pos_orient_inds.size != 0:
        if style.model3d.showdefault and make_func is not None:
            p_trs = make_func(**make_func_kwargs)
            p_trs = [p_trs] if isinstance(p_trs, dict) else p_trs
            for p_tr in p_trs:
                if is_mag and p_tr.get("type", "") == "mesh3d":
                    p_tr = update_magnet_mesh(
                        p_tr,
                        mag_style=style.magnetization,
                        magnetization=input_obj.magnetization,
                        color_slicing=not supports_colorgradient,
                    )

                traces_generic.append(p_tr)

        extra_model3d_traces = (
            style.model3d.data if style.model3d.data is not None else []
        )
        for extr in extra_model3d_traces:
            if not extr.show:
                continue
            extr.update(extr.updatefunc())  # update before checking backend
            if extr.backend == "generic":
                extr.update(extr.updatefunc())
                tr_generic = {"opacity": style.opacity}
                ttype = extr.constructor.lower()
                obj_extr_trace = extr.kwargs() if callable(extr.kwargs) else extr.kwargs
                obj_extr_trace = {"type": ttype, **obj_extr_trace}
                if ttype == "scatter3d":
                    for k in ("marker", "line"):
                        tr_generic[f"{k}_color"] = tr_generic.get(
                            f"{k}_color", style.color
                        )
                elif ttype == "mesh3d":
                    tr_generic["showscale"] = tr_generic.get("showscale", False)
                    if "facecolor" in obj_extr_trace:
                        ttype = "mesh3d_facecolor"
                    tr_generic["color"] = tr_generic.get("color", style.color)
                else:  # pragma: no cover
                    raise ValueError(
                        f"{ttype} is not supported, only 'scatter3d' and 'mesh3d' are"
                    )
                tr_generic.update(linearize_dict(obj_extr_trace, separator="_"))
                traces_generic.append(tr_generic)

    path_traces_generic = []
    for tr in traces_generic:
        temp_rot_traces = []
        name_suff = tr.pop("name_suffix", None)
        name = tr.get("name", "") if legendtext is None else legendtext
        for orient, pos in zip(orientations, positions):
            tr1 = place_and_orient_model3d(tr, orientation=orient, position=pos)
            if name_suff is not None:
                tr1["name"] = f"{name}{name_suff}"
            temp_rot_traces.append(tr1)
        path_traces_generic.extend(group_traces(*temp_rot_traces))

    for extr in extra_model3d_traces:
        if not extr.show:
            continue
        extr.update(extr.updatefunc())  # update before checking backend
        if extr.backend == extra_backend:
            for orient, pos in zip(orientations, positions):
                tr_generic = {
                    "model3d": extr,
                    "position": pos,
                    "orientation": orient,
                    "kwargs": {
                        "opacity": style.opacity,
                        "color": style.color,
                        "legendgroup": legendgroup,
                        "name": legendtext,
                        "row": row,
                        "col": col,
                    },
                }
                path_traces_extra_non_generic_backend.append(tr_generic)

    all_generic_traces.extend(group_traces(*path_traces_generic))

    if np.array(input_obj.position).ndim > 1 and style.path.show:
        scatter_path = make_path(input_obj)
        all_generic_traces.append(scatter_path)

    if is_mag_arrows:
        all_generic_traces.append(make_mag_arrows(input_obj, pos_orient_inds))

    for tr in all_generic_traces:
        tr.update(row=row, col=col)
        if tr.get("opacity", None) is None:
            tr["opacity"] = style.opacity
        if tr.get("legendgroup", None) is None:
            # allow invalid trimesh traces to have their own legendgroup
            tr["legendgroup"] = legendgroup
        if legendtext is not None:
            tr["name"] = legendtext
        elif "name" not in tr:
            tr["name"] = style.label
        if tr.get("facecolor", None) is not None:
            # this allows merging of 3d meshes, ignoring different colors
            tr["color"] = None
    out = {"generic": all_generic_traces}
    if extra_backend:
        out.update({extra_backend: path_traces_extra_non_generic_backend})
    return out


def clean_legendgroups(frames, clean_2d=False):
    """removes legend duplicates for a plotly figure"""
    for fr in frames:
        legendgroups = []
        for tr in fr["data"]:
            if "z" in tr or clean_2d:
                lg = tr.get("legendgroup", None)
                if lg is not None and lg not in legendgroups:
                    legendgroups.append(lg)
                    tr["showlegend"] = True
                elif lg is not None:  # and tr.legendgrouptitle.text is None:
                    tr["showlegend"] = False
        for tr in fr["extra_backend_traces"]:
            lg = tr["kwargs"].get("legendgroup", None)
            if lg is not None and lg not in legendgroups:
                legendgroups.append(lg)
                tr["kwargs"]["showlegend"] = True
            elif lg is not None:  # and tr.legendgrouptitle.text is None:
                tr["kwargs"]["showlegend"] = False


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
    animation_output,
):
    """Exctract animation properties"""
    # pylint: disable=import-outside-toplevel
    from magpylib._src.obj_classes.class_Collection import Collection

    path_lengths = []
    for obj in objs:
        subobjs = [obj]
        if isinstance(obj, Collection):
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


def draw_frame(objs, colorsequence=None, zoom=0.0, autosize=None, **kwargs) -> Tuple:
    """
    Creates traces from input `objs` and provided parameters, updates the size of objects like
    Sensors and Dipoles in `kwargs` depending on the canvas size.

    Returns
    -------
    traces_dicts, kwargs: dict, dict
        returns the traces in a obj/traces_list dictionary and updated kwargs
    """
    if colorsequence is None:
        colorsequence = Config.display.colorsequence
    # dipoles and sensors use autosize, the trace building has to be put at the back of the queue.
    # autosize is calculated from the other traces overall scene range

    style_path_frames = kwargs.get(
        "style_path_frames", [-1]
    )  # get before next func strips style
    flat_objs_props, kwargs = get_flatten_objects_properties(
        *objs, colorsequence=colorsequence, **kwargs
    )
    traces_dict, traces_to_resize_dict, extra_backend_traces = get_row_col_traces(
        flat_objs_props, **kwargs
    )
    traces = [t for tr in traces_dict.values() for t in tr]
    ranges = get_scene_ranges(*traces, zoom=zoom)
    if autosize is None or autosize == "return":
        autosize = np.mean(np.diff(ranges)) / Config.display.autosizefactor

    traces_dict_2, _, extra_backend_traces2 = get_row_col_traces(
        traces_to_resize_dict, autosize=autosize, **kwargs
    )
    traces_dict.update(traces_dict_2)
    extra_backend_traces.extend(extra_backend_traces2)
    traces = group_traces(*[t for tr in traces_dict.values() for t in tr])
    obj_list_2d = [o for o in objs if o["output"] != "model3d"]
    for objs_2d in obj_list_2d:
        traces2d = get_generic_traces_2D(
            **objs_2d,
            style_path_frames=style_path_frames,
            flat_objs_props=flat_objs_props,
        )
        traces.extend(traces2d)
    return traces, autosize, ranges, extra_backend_traces


def get_row_col_traces(flat_objs_props, extra_backend=False, autosize=None, **kwargs):
    """Return traces, traces to resize and extra_backend_traces"""
    # pylint: disable=protected-access
    extra_backend_traces = []
    traces_dict = {}
    traces_to_resize_dict = {}
    for obj, params in flat_objs_props.items():
        params.update(kwargs)
        if autosize is None and getattr(obj, "_autosize", False):
            traces_to_resize_dict[obj] = {**params}
            # temporary coordinates to be able to calculate ranges
            x, y, z = obj._position.T
            traces_dict[obj] = [{"x": x, "y": y, "z": z}]
        else:
            traces_dict[obj] = []
            rco_obj = params.pop("row_cols")
            for rco in rco_obj:
                params["row"], params["col"], output_typ = rco
                if output_typ == "model3d":
                    orig_style = None
                    try:
                        # temporary replace style attribute
                        orig_style = obj._style
                        obj._style = params.pop("style", None)
                        out_traces = get_generic_traces(
                            obj,
                            extra_backend=extra_backend,
                            autosize=autosize,
                            **params,
                        )
                    finally:
                        obj._style = orig_style
                    if extra_backend:
                        extra_backend_traces.extend(out_traces.get(extra_backend, []))
                    traces_dict[obj].extend(out_traces["generic"])
    return traces_dict, traces_to_resize_dict, extra_backend_traces


def get_frames(
    objs,
    colorsequence=None,
    zoom=1,
    title=None,
    animation=False,
    supports_colorgradient=True,
    backend="generic",
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
            supports_colorgradient=supports_colorgradient,
            extra_backend=backend,
            **kwargs,
        )
        if i == 0:  # get the dipoles and sensors autosize from first frame
            autosize = autosize_init
        frames.append(
            {
                "data": traces,
                "name": str(ind + 1),
                "layout": {"title": title_str},
                "extra_backend_traces": extra_backend_traces,
            }
        )

    clean_legendgroups(frames)
    traces = [t for frame in frames for t in frame["data"]]
    ranges = get_scene_ranges(*traces, zoom=zoom)
    out = {
        "frames": frames,
        "ranges": ranges,
        "input_kwargs": {**kwargs, **animation_kwargs},
    }
    if animation:
        out.update(
            {
                "frame_duration": frame_duration,
                "path_indices": path_indices,
            }
        )
    return out

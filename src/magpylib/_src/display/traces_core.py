"""Generic trace drawing functionalities"""

# pylint: disable=C0302
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=too-many-nested-blocks
# pylint: disable=cyclic-import
import warnings
from itertools import combinations
from itertools import cycle
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import numpy as np
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as RotScipy

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
from magpylib._src.display.traces_utility import create_null_dim_trace
from magpylib._src.display.traces_utility import draw_arrow_from_vertices
from magpylib._src.display.traces_utility import draw_arrow_on_circle
from magpylib._src.display.traces_utility import get_legend_label
from magpylib._src.display.traces_utility import merge_mesh3d
from magpylib._src.display.traces_utility import place_and_orient_model3d
from magpylib._src.display.traces_utility import triangles_area


def make_DefaultTrace(obj, **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
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
        "mode": "markers",
        "marker_size": 10,
        "marker_color": style.color,
        "marker_symbol": "diamond",
    }
    return {**trace, **kwargs}


def make_Polyline(obj, **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Creates the plotly scatter3d parameters for a Polyline current in a dictionary based on the
    provided arguments.
    """
    style = obj.style
    if obj.vertices is None:
        trace = create_null_dim_trace(color=style.color)
        return {**trace, **kwargs}

    traces = []
    for kind in ("arrow", "line"):
        kind_style = getattr(style, kind)
        if kind_style.show:
            color = style.color if kind_style.color is None else kind_style.color
            if kind == "arrow":
                current = 0 if obj.current is None else obj.current
                x, y, z = draw_arrow_from_vertices(
                    vertices=obj.vertices,
                    sign=np.sign(current),
                    arrow_size=kind_style.size,
                    arrow_pos=style.arrow.offset,
                    scaled=kind_style.sizemode == "scaled",
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
            traces.append({**trace, **kwargs})
    return traces


def make_Circle(obj, base=72, **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Creates the plotly scatter3d parameters for a Circle current in a dictionary based on the
    provided arguments.
    """
    style = obj.style
    if obj.diameter is None:
        trace = create_null_dim_trace(color=style.color)
        return {**trace, **kwargs}
    traces = []
    for kind in ("arrow", "line"):
        kind_style = getattr(style, kind)
        if kind_style.show:
            color = style.color if kind_style.color is None else kind_style.color

            if kind == "arrow":
                angle_pos_deg = 360 * np.round(style.arrow.offset * base) / base
                current = 0 if obj.current is None else obj.current
                vertices = draw_arrow_on_circle(
                    sign=np.sign(current),
                    diameter=obj.diameter,
                    arrow_size=style.arrow.size,
                    scaled=kind_style.sizemode == "scaled",
                    angle_pos_deg=angle_pos_deg,
                )
                x, y, z = vertices.T
            else:
                t = np.linspace(0, 2 * np.pi, base)
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
            traces.append({**trace, **kwargs})
    return traces


def make_Dipole(obj, autosize=None, **kwargs) -> Dict[str, Any]:
    """
    Create the plotly mesh3d parameters for a dipole in a dictionary based on the
    provided arguments.
    """
    style = obj.style
    moment = np.array([0.0, 0.0, 0.0]) if obj.moment is None else obj.moment
    moment_mag = np.linalg.norm(moment)
    size = style.size
    if autosize is not None and style.sizemode == "scaled":
        size *= autosize
    if moment_mag == 0:
        trace = create_null_dim_trace(color=style.color)
    else:
        trace = make_BaseArrow(
            "plotly-dict",
            base=10,
            diameter=0.3 * size,
            height=size,
            pivot=style.pivot,
            color=style.color,
        )
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


def make_Cuboid(obj, **kwargs) -> Dict[str, Any]:
    """
    Create the plotly mesh3d parameters for a Cuboid Magnet in a dictionary based on the
    provided arguments.
    """
    style = obj.style
    if obj.dimension is None:
        trace = create_null_dim_trace(color=style.color)
    else:
        trace = make_BaseCuboid(
            "plotly-dict", dimension=obj.dimension, color=style.color
        )
    return {**trace, **kwargs}


def make_Cylinder(obj, base=50, **kwargs) -> Dict[str, Any]:
    """
    Create the plotly mesh3d parameters for a Cylinder Magnet in a dictionary based on the
    provided arguments.
    """
    style = obj.style
    if obj.dimension is None:
        trace = create_null_dim_trace(color=style.color)
    else:
        diameter, height = obj.dimension
        trace = make_BasePrism(
            "plotly-dict",
            base=base,
            diameter=diameter,
            height=height,
            color=style.color,
        )
    return {**trace, **kwargs}


def make_CylinderSegment(obj, vertices=25, **kwargs) -> Dict[str, Any]:
    """
    Create the plotly mesh3d parameters for a Cylinder Segment Magnet in a dictionary based on the
    provided arguments.
    """
    style = obj.style
    if obj.dimension is None:
        trace = create_null_dim_trace(color=style.color)
    else:
        trace = make_BaseCylinderSegment(
            "plotly-dict", dimension=obj.dimension, vert=vertices, color=style.color
        )
    return {**trace, **kwargs}


def make_Sphere(obj, vertices=15, **kwargs) -> Dict[str, Any]:
    """
    Create the plotly mesh3d parameters for a Sphere Magnet in a dictionary based on the
    provided arguments.
    """
    style = obj.style

    if obj.diameter is None:
        trace = create_null_dim_trace(color=style.color)
    else:
        vertices = min(max(vertices, 3), 20)
        trace = make_BaseEllipsoid(
            "plotly-dict",
            vert=vertices,
            dimension=[obj.diameter] * 3,
            color=style.color,
        )
    return {**trace, **kwargs}


def make_Tetrahedron(obj, **kwargs) -> Dict[str, Any]:
    """
    Create the plotly mesh3d parameters for a Tetrahedron Magnet in a dictionary based on the
    provided arguments.
    """
    style = obj.style
    if obj.vertices is None:
        trace = create_null_dim_trace(color=style.color)
    else:
        trace = make_BaseTetrahedron(
            "plotly-dict", vertices=obj.vertices, color=style.color
        )
    return {**trace, **kwargs}


def make_triangle_orientations(obj, **kwargs) -> Dict[str, Any]:
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
    trace["ismagnet"] = False  # neede to avoid updating mag mesh
    return trace


def get_closest_vertices(faces_subsets, vertices):
    """Get closest pairs of points between disconnected subsets of faces indices"""
    # pylint: disable=used-before-assignment
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


def make_mesh_lines(obj, mode, **kwargs) -> Dict[str, Any]:
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
    x, y, z = lines.T
    trace = {
        "type": "scatter3d",
        "x": x,
        "y": y,
        "z": z,
        "mode": "markers+lines",
        "marker_color": marker.color,
        "marker_size": marker.size,
        "marker_symbol": marker.symbol,
        "line_color": line.color,
        "line_width": line.width,
        "line_dash": line.style,
        "legendgroup": f"{legendgroup} - {mode}edges",
        "name_suffix": f" - {mode}-edges",
        "name": get_legend_label(obj, suffix=False),
    }
    return {**trace, **kwargs}


def make_Triangle(obj, **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Creates the plotly mesh3d parameters for a Trianglular facet in a dictionary based on the
    provided arguments.
    """
    style = obj.style
    vert = obj.vertices

    if vert is None:
        trace = create_null_dim_trace(color=style.color)
    else:
        vec = np.cross(vert[1] - vert[0], vert[2] - vert[1])
        faces = np.array([[0, 1, 2]])
        # if magnetization is normal to the triangle, add a second triangle slightly above to enable
        # proper color gradient visualization. Otherwise only the middle color is shown.
        magnetization = (
            np.array([0.0, 0.0, 0.0])
            if obj.magnetization is None
            else obj.magnetization
        )
        if np.all(np.cross(magnetization, vec) == 0):
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
    traces = [{**trace, **kwargs}]
    if vert is not None and style.orientation.show:
        traces.append(make_triangle_orientations(obj, **kwargs))
    return traces


def make_TriangularMesh_single(obj, **kwargs) -> Dict[str, Any]:
    """
    Creates the plotly mesh3d parameters for a Trianglular facet mesh in a dictionary based on the
    provided arguments.
    """
    style = obj.style
    trace = make_BaseTriangularMesh(
        "plotly-dict", vertices=obj.vertices, faces=obj.faces, color=style.color
    )
    trace["name"] = get_legend_label(obj)
    # make edges sharper in plotly
    trace.update(flatshading=True, lighting_facenormalsepsilon=0, lighting_ambient=0.7)
    return {**trace, **kwargs}


def make_TriangularMesh(obj, **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
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
        subsets = obj.get_faces_subsets()
        col_seq = cycle(obj.style.mesh.disconnected.colorsequence)
        exponent = np.log10(len(subsets)).astype(int) + 1
        for ind, (tri, dis_color) in enumerate(zip(subsets, col_seq)):
            # temporary mutate faces from subset
            obj._faces = tri
            obj.style.magnetization.show = False
            tr = make_TriangularMesh_single(obj, **{**kwargs, "color": dis_color})
            # match first group with path scatter trace
            lg_suff = "" if ind == 0 else f"- part_{ind+1:02d}"
            tr["legendgroup"] = f"{kwargs.get('legendgroup', obj)}{lg_suff}"
            tr["name_suffix"] = f" - part_{ind+1:0{exponent}d}"
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


def make_Pixels(positions, size=1) -> Dict[str, Any]:
    """
    Create the plotly mesh3d parameters for Sensor pixels based on pixel positions and chosen size
    For now, only "cube" shape is provided.
    """
    pixels = [
        make_BaseCuboid("plotly-dict", position=p, dimension=[size] * 3)
        for p in positions
    ]
    return merge_mesh3d(*pixels)


def make_Sensor(obj, autosize=None, **kwargs) -> Dict[str, Any]:
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
    no_pix = pixel is None
    if not no_pix:
        pixel = np.unique(np.array(pixel).reshape((-1, 3)), axis=0)
    one_pix = not no_pix and pixel.shape[0] == 1
    style_arrows = style.arrows.as_dict(flatten=True, separator="_")
    sensor = get_sensor_mesh(
        **style_arrows, center_color=style.color, handedness=obj.handedness
    )
    vertices = np.array([sensor[k] for k in "xyz"]).T
    if style.color is not None:
        sensor["facecolor"][sensor["facecolor"] == "rgb(238,238,238)"] = style.color
    dim = np.array(
        [dimension] * 3 if isinstance(dimension, (float, int)) else dimension[:3],
        dtype=float,
    )
    if autosize is not None and style.sizemode == "scaled":
        dim *= autosize
    if no_pix:
        dim_ext = dim
    else:
        if one_pix:
            pixel = np.concatenate([[[0, 0, 0]], pixel])
        hull_dim = pixel.max(axis=0) - pixel.min(axis=0)
        dim_ext = max(np.mean(dim), np.min(hull_dim))
    cube_mask = (abs(vertices) < 1).all(axis=1)
    vertices[cube_mask] = 0 * vertices[cube_mask]
    vertices[~cube_mask] = dim_ext * vertices[~cube_mask]
    vertices /= 2  # sensor_mesh vertices are of length 2
    x, y, z = vertices.T
    sensor.update(x=x, y=y, z=z)
    meshes_to_merge = [sensor]
    if not no_pix:
        pixel_color = style.pixel.color
        pixel_size = style.pixel.size
        pixel_dim = 1
        if style.pixel.sizemode == "scaled":
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
    return {**trace, **kwargs}

""" plolty draw-functionalities"""

from itertools import cycle

try:
    import plotly.graph_objects as go
except ImportError as missing_module:
    raise ModuleNotFoundError(
        """In order to use the plotly plotting backend, you need to install plotly via pip or conda,
        see https://github.com/plotly/plotly.py"""
    ) from missing_module
import numpy as np
from scipy.spatial.transform import Rotation as RotScipy
from magpylib import _lib
from magpylib._lib.config import Config
from magpylib._lib.display.sensor_plotly_mesh import get_sensor_mesh

# Defaults
COLORMAP = [
    "#2E91E5",
    "#E15F99",
    "#1CA71C",
    "#FB0D0D",
    "#DA16FF",
    "#222A2A",
    "#B68100",
    "#750D86",
    "#EB663B",
    "#511CFB",
    "#00A08B",
    "#FB00D1",
    "#FC0080",
    "#B2828D",
    "#6C7C32",
    "#778AAE",
    "#862A16",
    "#A777F1",
    "#620042",
    "#1616A7",
    "#DA60CA",
    "#6C4516",
    "#0D2A63",
    "#AF0038",
]

_UNIT_PREFIX = {
    -24: "y",  # yocto
    -21: "z",  # zepto
    -18: "a",  # atto
    -15: "f",  # femto
    -12: "p",  # pico
    -9: "n",  # nano
    -6: "µ",  # micro
    -3: "m",  # milli
    0: "",
    3: "k",  # kilo
    6: "M",  # mega
    9: "G",  # giga
    12: "T",  # tera
    15: "P",  # peta
    18: "E",  # exa
    21: "Z",  # zetta
    24: "Y",  # yotta
}


def unit_prefix(number, unit="", precision=3, char_between=""):
    from math import log10

    digits = int(log10(abs(number))) // 3 * 3 if number != 0 else 0
    prefix = _UNIT_PREFIX.get(digits, "")
    if prefix != "":
        new_number_str = "{:.{}g}".format(number / 10 ** digits, precision)
    else:
        new_number_str = "{:.{}g}".format(number, precision)
    return f"{new_number_str}{char_between}{prefix}{unit}"


def _getIntensity(vertices, mag, pos):
    """vertices: [x,y,z] array"""
    if not all(m == 0 for m in mag):
        p = np.array(vertices)
        pos = np.array(pos)
        m = np.array(mag) / np.linalg.norm(mag)
        a = (p[0] - pos[0]) * m[0] + (p[1] - pos[1]) * m[1] + (p[2] - pos[2]) * m[2]
        b = (p[0] - pos[0]) ** 2 + (p[1] - pos[1]) ** 2 + (p[2] - pos[2]) ** 2
        return a / np.sqrt(b)
    else:
        return vertices[0] * 0


def _getColorscale(
    color_transition=0.1, north_color=None, middle_color=None, south_color=None
):
    if north_color is None:
        north_color = Config.NORTH_COLOR
    if south_color is None:
        south_color = Config.SOUTH_COLOR
    if middle_color is None:
        middle_color = Config.MIDDLE_COLOR
    if middle_color is False:
        return [
            [0.0, south_color],
            [0.5 * (1 - color_transition), south_color],
            [0.5 * (1 + color_transition), north_color],
            [1, north_color],
        ]
    else:
        return [
            [0.0, south_color],
            [0.2 - 0.2 * (color_transition), south_color],
            [0.2 + 0.3 * (color_transition), middle_color],
            [0.8 - 0.3 * (color_transition), middle_color],
            [0.8 + 0.2 * (color_transition), north_color],
            [1.0, north_color],
        ]


def make_BaseCuboid(dim=(1.0, 1.0, 1.0), pos=(0.0, 0.0, 0.0)):
    return dict(
        type="mesh3d",
        i=np.array([7, 0, 0, 0, 4, 4, 2, 6, 4, 0, 3, 7]),
        j=np.array([3, 4, 1, 2, 5, 6, 5, 5, 0, 1, 2, 2]),
        k=np.array([0, 7, 2, 3, 6, 7, 1, 2, 5, 5, 7, 6]),
        x=np.array([-1, -1, 1, 1, -1, -1, 1, 1]) * 0.5 * dim[0] + pos[0],
        y=np.array([-1, 1, 1, -1, -1, 1, 1, -1]) * 0.5 * dim[1] + pos[1],
        z=np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * 0.5 * dim[2] + pos[2],
    )


def make_BasePrism(base_vertices=3, diameter=1, height=1, pos=(0.0, 0.0, 0.0)):
    N = base_vertices
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    c1 = np.array([1 * np.cos(t), 1 * np.sin(t), t * 0 - 1]) * 0.5
    c2 = np.array([1 * np.cos(t), 1 * np.sin(t), t * 0 + 1]) * 0.5
    c3 = np.array([[0, 0], [0, 0], [-1, 1]]) * 0.5
    c = np.concatenate([c1, c2, c3], axis=1)
    c = c.T * np.array([diameter, diameter, height]) + np.array(pos)
    i1 = np.arange(N)
    j1 = i1 + 1
    j1[-1] = 0
    k1 = i1 + N

    i2 = i1 + N
    j2 = j1 + N
    j2[-1] = N
    k2 = i1 + 1
    k2[-1] = 0

    i3 = i1
    j3 = j1
    k3 = i1 * 0 + 2 * N

    i4 = i2
    j4 = j2
    k4 = k3 + 1

    # k2&j2 and k3&j3 inverted because of face orientation
    i = np.concatenate([i1, i2, i3, i4])
    j = np.concatenate([j1, k2, k3, j4])
    k = np.concatenate([k1, j2, j3, k4])

    x, y, z = c.T
    return dict(type="mesh3d", x=x, y=y, z=z, i=i, j=j, k=k)


def make_Ellipsoid(dim=(1.0, 1.0, 1.0), pos=(0.0, 0.0, 0.0), Nvert=15):
    N = min(max(Nvert, 3), 20)
    phi = np.linspace(0, 2 * np.pi, Nvert, endpoint=False)
    theta = np.linspace(-np.pi / 2, np.pi / 2, Nvert, endpoint=True)
    phi, theta = np.meshgrid(phi, theta)

    x = np.cos(theta) * np.sin(phi) * dim[0] * 0.5 + pos[0]
    y = np.cos(theta) * np.cos(phi) * dim[1] * 0.5 + pos[1]
    z = np.sin(theta) * dim[2] * 0.5 + pos[2]

    x, y, z = x.flatten()[N - 1 :], y.flatten()[N - 1 :], z.flatten()[N - 1 :]

    i1 = [0] * N
    j1 = np.array([N] + list(range(1, N)), dtype=int)
    k1 = np.array(list(range(1, N)) + [N], dtype=int)

    i2 = np.concatenate([k1 + i * N for i in range(N - 2)])
    j2 = np.concatenate([j1 + i * N for i in range(N - 2)])
    k2 = np.concatenate([j1 + (i + 1) * N for i in range(N - 2)])

    i3 = np.concatenate([k1 + i * N for i in range(N - 2)])
    j3 = np.concatenate([j1 + (i + 1) * N for i in range(N - 2)])
    k3 = np.concatenate([k1 + (i + 1) * N for i in range(N - 2)])

    i = np.concatenate([i1, i2, i3])
    j = np.concatenate([j1, j2, j3])
    k = np.concatenate([k1, k2, k3])

    return dict(type="mesh3d", x=x, y=y, z=z, i=i, j=j, k=k)


def make_BaseCylinderSegment(d1=1, d2=2, h=1, phi1=0, phi2=90, Nvert=30):
    N = Nvert
    phi = np.linspace(phi1, phi2, N)
    x = np.cos(np.deg2rad(phi))
    y = np.sin(np.deg2rad(phi))
    z = np.zeros(N)
    c1 = np.array([d1 / 2 * x, d1 / 2 * y, z + h / 2])
    c2 = np.array([d2 / 2 * x, d2 / 2 * y, z + h / 2])
    c3 = np.array([d1 / 2 * x, d1 / 2 * y, z - h / 2])
    c4 = np.array([d2 / 2 * x, d2 / 2 * y, z - h / 2])
    x, y, z = np.concatenate([c1, c2, c3, c4], axis=1)

    i1 = np.arange(N - 1)
    j1 = i1 + N
    k1 = i1 + 1

    i2 = k1
    j2 = j1
    k2 = j1 + 1

    i3 = i1
    j3 = k1
    k3 = j1 + N

    i4 = k3 + 1
    j4 = k3
    k4 = k1

    i5 = np.array([0, N])
    j5 = np.array([2 * N, 0])
    k5 = np.array([3 * N, 3 * N])

    i = np.hstack(
        [i1, i2, i1 + 2 * N, i2 + 2 * N, i3, i4, i3 + N, i4 + N, i5, i5 + N - 1]
    )
    j = np.hstack(
        [j1, j2, k1 + 2 * N, k2 + 2 * N, j3, j4, k3 + N, k4 + N, j5, k5 + N - 1]
    )
    k = np.hstack(
        [k1, k2, j1 + 2 * N, j2 + 2 * N, k3, k4, j3 + N, j4 + N, k5, j5 + N - 1]
    )

    return dict(type="mesh3d", x=x, y=y, z=z, i=i, j=j, k=k)


def make_Cone(base_vertices=3, diameter=1, height=1, pos=(0.0, 0.0, 0.0)):
    N = base_vertices
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    c = np.array([np.cos(t), np.sin(t), t * 0 - 1]) * 0.5
    tp = np.array([[0, 0, 0.5]]).T
    c = np.concatenate([c, tp], axis=1)
    c = c.T * np.array([diameter, diameter, height]) + np.array(pos)
    x, y, z = c.T

    i = np.arange(N, dtype=int)
    j = i + 1
    j[-1] = 0
    k = np.array([N] * N, dtype=int)
    return dict(type="mesh3d", x=x, y=y, z=z, i=i, j=j, k=k)


def make_BaseArrow(base_vertices=30, diameter=0.3, height=1):
    h, d = height, diameter
    cone = make_Cone(
        base_vertices=base_vertices, diameter=d, height=d, pos=(0.0, 0.0, (h - d) / 2)
    )
    prism = make_BasePrism(
        base_vertices=base_vertices,
        diameter=d / 2,
        height=h - d,
        pos=(0.0, 0.0, -d / 2),
    )
    arrow = merge_mesh3d(cone, prism)
    return arrow


def draw_arrow(vec, pos, sign=1):
    hy = sign * 0.1
    hx = 0.06
    norm = np.linalg.norm(vec)
    arrow = (
        np.array(
            [
                [0, -0.5, 0],
                [0, 0, 0],
                [-hx, 0 - hy, 0],
                [0, 0, 0],
                [hx, 0 - hy, 0],
                [0, 0, 0],
                [0, 0.5, 0],
            ]
        )
        * norm
    )
    nvec = np.array(vec) / norm
    yaxis = np.array([0, 1, 0])
    cross = np.cross(nvec, yaxis)
    dot = np.dot(nvec, yaxis)
    n = np.linalg.norm(cross)
    if n != 0:
        t = np.arccos(dot)
        R = RotScipy.from_rotvec(-t * cross / n)
        arrow = R.apply(arrow)
    x, y, z = (arrow + pos).T
    return x, y, z


def make_Line(
    current=0.0,
    vertices=((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    pos=(0.0, 0.0, 0.0),
    show_arrows=True,
    orientation=None,
    name=None,
    name_suffix=None,
    color=None,
    **kwargs,
):
    name = "Line Curent" if name is None else name
    name_suffix = (
        " ({}A)".format(unit_prefix(current))
        if name_suffix is None
        else f" ({name_suffix})"
    )
    if show_arrows:
        vectors = np.diff(vertices, axis=0)
        positions = vertices[:-1] + vectors / 2
        vertices = np.concatenate(
            [
                draw_arrow(vec, pos, np.sign(current))
                for vec, pos in zip(vectors, positions)
            ],
            axis=1,
        )
    else:
        vertices = np.array(vertices).T
    if orientation is not None:
        vertices = orientation.apply(vertices.T).T
    x, y, z = (vertices.T + pos).T
    line = dict(
        type="scatter3d",
        x=x,
        y=y,
        z=z,
        name=f"""{name}{name_suffix}""",
        mode="lines",
        line_width=5,
        line_color=color,
    )
    return {**line, **kwargs}


def make_Circular(
    current=0.0,
    diameter=1.0,
    pos=(0.0, 0.0, 0.0),
    Nvert=50,
    show_arrows=True,
    orientation=None,
    name=None,
    name_suffix=None,
    color=None,
    **kwargs,
):
    name = "Circular Curent" if name is None else name
    name_suffix = (
        " ({}A)".format(unit_prefix(current))
        if name_suffix is None
        else f" ({name_suffix})"
    )
    t = np.linspace(0, 2 * np.pi, Nvert)
    x = np.cos(t)
    y = np.sin(t)
    if show_arrows:
        hy = 0.2 * np.sign(current)
        hx = 0.15
        x = np.hstack([x, [1 + hx, 1, 1 - hx]])
        y = np.hstack([y, [-hy, 0, -hy]])
    x = x * diameter / 2
    y = y * diameter / 2
    z = np.zeros(x.shape)
    vertices = np.array([x, y, z])
    if orientation is not None:
        vertices = orientation.apply(vertices.T).T
    x, y, z = (vertices.T + pos).T
    circular = dict(
        type="scatter3d",
        x=x,
        y=y,
        z=z,
        name=f"""{name}{name_suffix}""",
        mode="lines",
        line_width=5,
        line_color=color,
    )
    return {**circular, **kwargs}


def make_UnsupportedObject(
    pos=(0.0, 0.0, 0.0),
    orientation=None,
    name=None,
    name_suffix=None,
    color=None,
    **kwargs,
):
    name = "Unkwnon obj" if name is None else name
    name_suffix = (
        " (Unsupported visualisation)" if name_suffix is None else f" ({name_suffix})"
    )
    vertices = np.array([pos])
    if orientation is not None:
        vertices = orientation.apply(vertices).T
    x, y, z = vertices
    obj = dict(
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
    return {**obj, **kwargs}


def make_Dipole(
    moment=(0.0, 0.0, 1.0),
    pos=(0.0, 0.0, 0.0),
    size=1.0,
    orientation=None,
    color_transition=0.0,
    name=None,
    name_suffix=None,
    north_color=None,
    middle_color=None,
    south_color=None,
    **kwargs,
):
    name = "Dipole" if name is None else name
    moment_mag = np.linalg.norm(moment)
    name_suffix = (
        " (moment={}T/m³)".format(unit_prefix(moment_mag))
        if name_suffix is None
        else f" ({name_suffix})"
    )
    dipole = make_BaseArrow(base_vertices=10, diameter=0.3 * size, height=size)
    nvec = np.array(moment) / moment_mag
    zaxis = np.array([0, 0, 1])
    cross = np.cross(nvec, zaxis)
    dot = np.dot(nvec, zaxis)
    n = np.linalg.norm(cross)
    t = np.arccos(dot)
    vec = -t * cross / n if n != 0 else (0, 0, 0)
    mag_orient = RotScipy.from_rotvec(vec)
    if orientation is not None:
        orientation = orientation * mag_orient
    else:
        orientation = mag_orient
    mag = (0, 0, 1)
    return _update_mag_mesh(
        dipole,
        name,
        name_suffix,
        mag,
        orientation,
        pos,
        color_transition,
        north_color,
        middle_color,
        south_color,
        **kwargs,
    )


def make_Cuboid(
    mag=(0.0, 0.0, 1000.0),
    dim=(1.0, 1.0, 1.0),
    pos=(0.0, 0.0, 0.0),
    orientation=None,
    color_transition=0.0,
    name=None,
    name_suffix=None,
    north_color=None,
    middle_color=None,
    south_color=None,
    **kwargs,
):
    name = "Cuboid" if name is None else name
    name_suffix = (
        " ({}mx{}mx{}m)".format(*(unit_prefix(d / 1000) for d in dim))
        if name_suffix is None
        else f" ({name_suffix})"
    )
    cuboid = make_BaseCuboid(dim=dim, pos=(0.0, 0.0, 0.0))
    return _update_mag_mesh(
        cuboid,
        name,
        name_suffix,
        mag,
        orientation,
        pos,
        color_transition,
        north_color,
        middle_color,
        south_color,
        **kwargs,
    )


def make_Cylinder(
    mag=(0.0, 0.0, 1000.0),
    base_vertices=50,
    diameter=1,
    height=1.0,
    pos=(0.0, 0.0, 0.0),
    orientation=None,
    color_transition=0.0,
    name=None,
    name_suffix=None,
    north_color=None,
    middle_color=None,
    south_color=None,
    **kwargs,
):
    name = "Cylinder" if name is None else name
    name_suffix = (
        " (D={}m, H={}m)".format(*(unit_prefix(d / 1000) for d in (diameter, height)))
        if name_suffix is None
        else f" ({name_suffix})"
    )
    cylinder = make_BasePrism(
        base_vertices=base_vertices,
        diameter=diameter,
        height=height,
        pos=(0.0, 0.0, 0.0),
    )
    return _update_mag_mesh(
        cylinder,
        name,
        name_suffix,
        mag,
        orientation,
        pos,
        color_transition,
        north_color,
        middle_color,
        south_color,
        **kwargs,
    )


def make_CylinderSegment(
    mag=(0.0, 0.0, 1000.0),
    dimension=(1.0, 2.0, 1.0, 0.0, 90.0),
    pos=(0.0, 0.0, 0.0),
    orientation=None,
    Nvert=25.0,
    color_transition=0.0,
    name=None,
    name_suffix=None,
    north_color=None,
    middle_color=None,
    south_color=None,
    **kwargs,
):
    name = "CylinderSegment" if name is None else name
    name_suffix = (
        " (d1={}m, d2={}m, h={}m, phi1={}°, phi2={}°)".format(
            *(unit_prefix(d / (1000 if i < 3 else 1)) for i, d in enumerate(dimension))
        )
        if name_suffix is None
        else f" ({name_suffix})"
    )
    cylinder_segment = make_BaseCylinderSegment(*dimension, Nvert=Nvert)
    return _update_mag_mesh(
        cylinder_segment,
        name,
        name_suffix,
        mag,
        orientation,
        pos,
        color_transition,
        north_color,
        middle_color,
        south_color,
        **kwargs,
    )


def make_Sphere(
    mag=(0.0, 0.0, 1000.0),
    Nvert=15,
    diameter=1,
    pos=(0.0, 0.0, 0.0),
    orientation=None,
    color_transition=0.0,
    name=None,
    name_suffix=None,
    north_color=None,
    middle_color=None,
    south_color=None,
    **kwargs,
):
    name = "Sphere" if name is None else name
    name_suffix = (
        " (D={}m)".format(unit_prefix(diameter / 1000))
        if name_suffix is None
        else f" ({name_suffix})"
    )
    sphere = make_Ellipsoid(Nvert=Nvert, dim=[diameter] * 3, pos=(0.0, 0.0, 0.0))
    return _update_mag_mesh(
        sphere,
        name,
        name_suffix,
        mag,
        orientation,
        pos,
        color_transition,
        north_color,
        middle_color,
        south_color,
        **kwargs,
    )


def make_Sensor(
    pixel=(0.0, 0.0, 0.0),
    dim=(1.0, 1.0, 1.0),
    pos=(0.0, 0.0, 0.0),
    orientation=None,
    name=None,
    name_suffix=None,
    color=None,
    **kwargs,
):
    name = "Sensor" if name is None else name
    pixel = np.array(pixel)
    pixel_str = (
        f""" ({'x'.join(str(p) for p in pixel.shape[:-1])} pixels)"""
        if pixel.ndim != 1
        else ""
    )
    name_suffix = pixel_str if name_suffix is None else f" ({name_suffix})"
    sensor = get_sensor_mesh()
    vertices = np.array([sensor[k] for k in "xyz"]).T
    if color is not None:
        sensor["facecolor"][sensor["facecolor"] == "rgb(238,238,238)"] = color
    if pixel.ndim == 1:
        dim_ext = dim
        dim = 0
    else:
        dim = pixel.max(axis=0) - pixel.min(axis=0)
        mask = np.diff(pixel, axis=0).sum(axis=0) == 0
        dim_ext = np.mean(dim)
        dim[mask] = dim_ext / 10
    cube_mask = (vertices < 1).all(axis=1)
    vertices[cube_mask] = dim * vertices[cube_mask]
    vertices[~cube_mask] = dim_ext * vertices[~cube_mask]
    x, y, z = vertices.T
    sensor.update(x=x, y=y, z=z)
    return _update_mag_mesh(
        sensor, name, name_suffix, orientation=orientation, position=pos, **kwargs
    )


def _update_mag_mesh(
    mesh_dict,
    name,
    name_suffix,
    magnetization=None,
    orientation=None,
    position=None,
    color_transition=None,
    north_color=None,
    middle_color=None,
    south_color=None,
    **kwargs,
):
    vertices = np.array([mesh_dict[k] for k in "xyz"])
    if magnetization is not None and color_transition >= 0:
        if middle_color == "auto":
            middle_color = kwargs.get("color", None)
        mesh_dict["colorscale"] = _getColorscale(
            color_transition,
            north_color=north_color,
            middle_color=middle_color,
            south_color=south_color,
        )
        mesh_dict["intensity"] = _getIntensity(
            vertices=vertices, mag=magnetization, pos=(0.0, 0.0, 0.0)
        )
    if orientation is not None:
        vertices = orientation.apply(vertices.T).T
    x, y, z = (vertices.T + position).T
    mesh_dict.update(
        x=x, y=y, z=z, showscale=False, name=f"""{name}{name_suffix}""", **kwargs
    )
    return {**mesh_dict, **kwargs}


def merge_mesh3d(*traces, concat_xyz=True):
    merged_trace = dict()
    L = np.array([0] + [len(b["x"]) for b in traces[:-1]]).cumsum()
    for k in "ijk":
        if k in traces[0]:
            merged_trace[k] = np.hstack([b[k] + l for b, l in zip(traces, L)])
    for k in "xyz":
        if concat_xyz:
            merged_trace[k] = np.concatenate([b[k] for b in traces])
        else:
            merged_trace[k] = np.array([b[k] for b in traces])
    for k in ("intensity", "facecolor"):
        if k in traces[0] and traces[0][k] is not None:
            merged_trace[k] = np.hstack([b[k] for b in traces])
    for k, v in traces[0].items():
        if k not in merged_trace:
            merged_trace[k] = v
    return merged_trace


def merge_scatter3d(*traces):
    merged_trace = dict()
    for k in "xyz":
        merged_trace[k] = np.hstack([pts for b in traces for pts in [[None], b[k]]])
    for k, v in traces[0].items():
        if k not in merged_trace:
            merged_trace[k] = v
    return merged_trace


def merge_traces(*traces, concat_xyz=True):
    if traces[0]["type"] == "mesh3d":
        trace = merge_mesh3d(*traces, concat_xyz=concat_xyz)
    elif traces[0]["type"] == "scatter3d":
        trace = merge_scatter3d(*traces)
    return trace


def getTraces(
    input_obj,
    show_path=False,
    size_dipoles=1,
    size_sensors=1,
    show_arrows=True,
    show_path_numbering=False,
    opacity=None,
    color_transition=None,
    north_color=None,
    middle_color=None,
    south_color=None,
    **kwargs,
):

    Sensor = _lib.obj_classes.Sensor
    Cuboid = _lib.obj_classes.Cuboid
    Cylinder = _lib.obj_classes.Cylinder
    CylinderSegment = _lib.obj_classes.CylinderSegment
    Sphere = _lib.obj_classes.Sphere
    Dipole = _lib.obj_classes.Dipole
    Circular = _lib.obj_classes.Circular
    Line = _lib.obj_classes.Line

    if color_transition is None:
        color_transition = Config.COLOR_TRANSITION

    mag_color_kwargs = dict(
        color_transition=color_transition,
        north_color=north_color,
        middle_color=middle_color,
        south_color=south_color,
    )

    kwargs["opacity"] = 1 if opacity is None else opacity
    color = kwargs.get("color", None)

    haspath = input_obj.position.ndim > 1
    traces = []
    if isinstance(input_obj, Sensor):
        kwargs.update(
            dim=getattr(input_obj, "dimension", size_sensors),
            pixel=getattr(input_obj, "pixel", (0.0, 0.0, 0.0)),
        )
        make_func = make_Sensor
    elif isinstance(input_obj, Cuboid):
        kwargs.update(
            mag=input_obj.magnetization,
            dim=input_obj.dimension,
            **mag_color_kwargs,
        )
        make_func = make_Cuboid
    elif isinstance(input_obj, Cylinder):
        base_vertices = min(
            50, Config.ITER_CYLINDER
        )  # no need to render more than 50 vertices
        kwargs.update(
            mag=input_obj.magnetization,
            diameter=input_obj.dimension[0],
            height=input_obj.dimension[0],
            base_vertices=base_vertices,
            **mag_color_kwargs,
        )
        make_func = make_Cylinder
    elif isinstance(input_obj, CylinderSegment):
        Nvert = min(50, Config.ITER_CYLINDER)  # no need to render more than 50 vertices
        kwargs.update(
            mag=input_obj.magnetization,
            dimension=input_obj.dimension,
            Nvert=Nvert,
            **mag_color_kwargs,
        )
        make_func = make_CylinderSegment
    elif isinstance(input_obj, Sphere):
        kwargs.update(
            mag=input_obj.magnetization,
            diameter=input_obj.diameter,
            **mag_color_kwargs,
        )
        make_func = make_Sphere
    elif isinstance(input_obj, Dipole):
        kwargs.update(
            moment=input_obj.moment,
            size=size_dipoles,
            **mag_color_kwargs,
        )
        make_func = make_Dipole
    elif isinstance(input_obj, Line):
        kwargs.update(
            vertices=input_obj.vertices,
            current=input_obj.current,
            show_arrows=show_arrows,
        )
        make_func = make_Line
    elif isinstance(input_obj, Circular):
        kwargs.update(
            diameter=input_obj.diameter,
            current=input_obj.current,
            show_arrows=show_arrows,
        )
        make_func = make_Circular
    else:
        kwargs.update(name=type(input_obj).__name__)
        make_func = make_UnsupportedObject

    if haspath:
        path_len = input_obj.position.shape[0]
        if show_path is True or show_path is False:
            inds = [-1]
        elif isinstance(show_path, int):
            inds = slice(None, None, -show_path)
        else:
            inds = np.array(show_path)
            inds = inds[inds < path_len]
            if inds.size == 0:
                inds = np.array([path_len - 1])
        path_traces = []
        for pos, orient in zip(input_obj.position[inds], input_obj.orientation[inds]):
            path_traces.append(make_func(pos=pos, orientation=orient, **kwargs))
        trace = merge_traces(*path_traces)
    else:
        trace = make_func(
            pos=input_obj.position, orientation=input_obj.orientation, **kwargs
        )
    trace.update({"legendgroup": f"{input_obj}", "showlegend": True})
    traces.append(trace)
    if haspath and show_path is not False:
        x, y, z = input_obj.position.T
        txt_kwargs = (
            {"mode": "markers+text+lines", "text": list(range(len(x)))}
            if show_path_numbering
            else {"mode": "markers+lines"}
        )
        scatter_path = dict(
            type="scatter3d",
            x=x,
            y=y,
            z=z,
            name=f"Path: {input_obj}",
            showlegend=False,
            legendgroup=f"{input_obj}",
            marker_size=1,
            marker_color=color,
            line_color=color,
            line_width=1,
            **txt_kwargs,
        )
        traces.append(scatter_path)

    return traces


def display_plotly(
    *objs,
    show_path=False,
    fig=None,
    renderer=None,
    duration=5,
    max_frame_rate=10,
    zoom=1,
    backtofirst=False,
    **kwargs,
):
    show_fig = False
    if fig is None:
        show_fig = True
        fig = go.Figure()

    title = getattr(objs[0], "name", None) if len(objs) == 1 else None

    with fig.batch_update():
        if (
            not any(obj.position.ndim > 1 for obj in objs) and show_path == "animate"
        ):  # check if some path exist for any object
            show_path = True
            import warnings

            warnings.warn("No path to be animated detected, displaying standard plot")
        if show_path == "animate":
            title = "3D-Paths Animation" if title is None else title
            animate_path(
                fig,
                objs,
                title=title,
                duration=duration,
                max_frame_rate=max_frame_rate,
                zoom=zoom,
                backtofirst=backtofirst,
                renderer=renderer,
                **kwargs,
            )
        else:
            traces_dicts = {
                obj: getTraces(obj, show_path=show_path, color=color, **kwargs)
                for obj, color in zip(objs, cycle(COLORMAP))
            }
            traces = [t for tr in traces_dicts.values() for t in tr]
            fig.add_traces(traces)
            fig.update_layout(
                title_text=title,
                **{f"scene_{k}axis_title": f"{k} [mm]" for k in "xyz"},
                scene_aspectmode="data",
            )
    if show_fig:
        fig.show(renderer=renderer)


def animate_path(
    fig,
    objs,
    title="3D-Paths Animation",
    duration=5,
    max_frame_rate=50,
    zoom=1,
    backtofirst=False,
    **kwargs,
):
    # make sure the number of frames does not exceed the max_frame_rate
    # downsample if necessary
    path_lengths = [
        obj.position.shape[0] if obj.position.ndim > 1 else 0 for obj in objs
    ]
    N = max(path_lengths)
    maxpos = duration * max_frame_rate
    if N <= maxpos:
        path_indices = np.arange(N)
    else:
        round_step = N / (maxpos - 1)
        ar = np.linspace(0, N, N, endpoint=False)
        path_indices = np.unique(np.floor(ar / round_step) * round_step).astype(
            int
        )  # downsampled indices
        path_indices[-1] = N - 1  # make sure the last frame is the last path position

    # calculate exponent of last frame index to avoid digit shift in frame number display during animation
    exp = (
        np.log10(path_indices.max()).astype(int) + 1
        if path_indices.ndim != 0 and path_indices.max() > 0
        else 1
    )

    # create frame for each path index or downsampled path index
    frames = []
    for ind in path_indices:
        traces_dicts = {
            obj: getTraces(obj, show_path=[ind], color=color, **kwargs)
            for obj, color in zip(objs, cycle(COLORMAP))
        }
        traces = [t for tr in traces_dicts.values() for t in tr]
        frames.append(
            go.Frame(
                data=traces, layout=dict(title=f"""{title} - frame: {ind+1:0{exp}d}""")
            )
        )

    if backtofirst:
        frames += frames[
            :1
        ]  # add first frame again, so that the last frame shows starting state

    # calculate max ranges to avoid ranges moving during animation
    range_factor = max(1, zoom)
    ranges = {k: [] for k in ("x", "y", "z")}
    for frame in frames:
        for t in frame.data:
            for k, v in ranges.items():
                v.extend([np.min(t[k]), np.max(t[k])])
    for k, v in ranges.items():
        ranges[k] = np.array([np.min(v), np.max(v)]) * range_factor

    # update fig
    frame_duration = int(duration * 1000 / path_indices.shape[0])
    fig.frames = frames
    fig.add_traces(frames[0].data)
    fig.update_layout(
        height=None,
        **{
            f"scene_{k}axis": dict(range=ranges[k], autorange=False, title=f"{k} [mm]")
            for k in "xyz"
        },
        scene_aspectratio={
            k: (np.diff(v) / np.diff(ranges["x"]))[0] for k, v in ranges.items()
        },
        scene_aspectmode="manual",
        title=title,
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": frame_duration}}],
                    )
                ],
                x=0.01,
                xanchor="left",
                y=1.01,
                yanchor="bottom",
            )
        ],
    )

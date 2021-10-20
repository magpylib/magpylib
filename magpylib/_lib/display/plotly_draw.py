""" plolty draw-functionalities"""
# pylint: disable=C0302

from itertools import cycle, combinations
from math import log10
from typing import Tuple
import warnings

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


def unit_prefix(number, unit="", precision=3, char_between="") -> str:
    """
    displays a number with given unit and precision and uses unit prefixes for the exponents from
    yotta (y) to Yocto (Y). If the exponent is smaller or bigger, falls back to scientific notation.

    Parameters
    ----------
    number : int, float
        can be any number
    unit : str, optional
        unit symbol can be any string, by default ""
    precision : int, optional
        gives the number of significant digits, by default 3
    char_between : str, optional
        character to insert between number of prefix. Can be " " or any string, if a space is wanted
        before the unit symbol , by default ""

    Returns
    -------
    str
        returns formatted number as string
    """
    digits = int(log10(abs(number))) // 3 * 3 if number != 0 else 0
    prefix = _UNIT_PREFIX.get(digits, "")
    # pylint: disable=consider-using-f-string
    if prefix != "":
        new_number_str = "{:.{}g}".format(number / 10 ** digits, precision)
    else:
        new_number_str = "{:.{}g}".format(number, precision)
    return f"{new_number_str}{char_between}{prefix}{unit}"


class Markers:
    """A class that stores markers 3D-coordinates"""

    def __init__(self, *markers, color=None):
        self.markers = np.array(markers)
        self.color = color


def _getIntensity(vertices, axis) -> np.ndarray:
    """
    Calculates the intensity values for vertices based on the distance of the vertices to the mean
    vertices position in the provided axis direction. It can be used for plotting
    fields on meshes. If `mag` See more infos here:https://plotly.com/python/3d-mesh/

    Parameters
    ----------
    vertices : ndarray Nx3
        the N vertices of the mesh object
    axis : ndarray 3
        direction vector

    Returns
    -------
    ndarray N
        returns 1D array of length N
    """
    if all(m == 0 for m in axis):
        intensity = np.array(vertices).T[0] * 0
    else:
        p = np.array(vertices).T
        pos = np.mean(p, axis=1)
        m = np.array(axis) / np.linalg.norm(axis)
        a = (p[0] - pos[0]) * m[0] + (p[1] - pos[1]) * m[1] + (p[2] - pos[2]) * m[2]
        b = (p[0] - pos[0]) ** 2 + (p[1] - pos[1]) ** 2 + (p[2] - pos[2]) ** 2
        intensity = a / np.sqrt(b)
    return intensity


def _getColorscale(
    color_transition=0.1, north_color=None, middle_color=None, south_color=None
) -> list:
    """
    Provides the colorscale for a plotly mesh3d trace.
    The colorscale must be an array containing arrays mapping a normalized value to an rgb, rgba,
    hex, hsl, hsv, or named color string. At minimum, a mapping for the lowest (0) and highest (1)
    values are required. For example, `[[0, 'rgb(0,0,255)'], [1,'rgb(255,0,0)']]`.
    In this case the colorscale is created depending on the north/middle/south poles colors. If the
    middle color is `None` the colorscale will only have north and south pole colors.
    By default colors as set from the global magpylib Config class.

    Parameters
    ----------
    color_transition : float, optional
        A value between 0 and 1. Sets the smoothness of the color transitions from adjacent colors
        visualization., by default 0.1
    north_color : [type], optional
        magnetic north pole color , by default None
    middle_color : [type], optional
        middle between south and north pole color, by default None
    south_color : [type], optional
        magnetic north pole color , by default None

    Returns
    -------
    list
        returns colorscale as list of tuples
    """

    if north_color is None:
        north_color = Config.NORTH_COLOR
    if south_color is None:
        south_color = Config.SOUTH_COLOR
    if middle_color is None:
        middle_color = Config.MIDDLE_COLOR
    if middle_color is False:
        colorscale = [
            [0.0, south_color],
            [0.5 * (1 - color_transition), south_color],
            [0.5 * (1 + color_transition), north_color],
            [1, north_color],
        ]
    else:
        colorscale = [
            [0.0, south_color],
            [0.2 - 0.2 * (color_transition), south_color],
            [0.2 + 0.3 * (color_transition), middle_color],
            [0.8 - 0.3 * (color_transition), middle_color],
            [0.8 + 0.2 * (color_transition), north_color],
            [1.0, north_color],
        ]
    return colorscale


def make_BaseCuboid(dim=(1.0, 1.0, 1.0), pos=(0.0, 0.0, 0.0)) -> dict:
    """
    Provides the base plotly cuboid mesh3d parameters in a dictionary based on dimension and
    position
    The zero position is in the barycenter of the vertices.
    """
    return dict(
        type="mesh3d",
        i=np.array([7, 0, 0, 0, 4, 4, 2, 6, 4, 0, 3, 7]),
        j=np.array([3, 4, 1, 2, 5, 6, 5, 5, 0, 1, 2, 2]),
        k=np.array([0, 7, 2, 3, 6, 7, 1, 2, 5, 5, 7, 6]),
        x=np.array([-1, -1, 1, 1, -1, -1, 1, 1]) * 0.5 * dim[0] + pos[0],
        y=np.array([-1, 1, 1, -1, -1, 1, 1, -1]) * 0.5 * dim[1] + pos[1],
        z=np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * 0.5 * dim[2] + pos[2],
    )


def make_BasePrism(base_vertices=3, diameter=1, height=1, pos=(0.0, 0.0, 0.0)) -> dict:
    """
    Provides the base plotly prism mesh3d parameters in a dictionary based on number of vertices of
    the base, the diameter the height and position.
    The zero position is in the barycenter of the vertices.
    """
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


def make_Ellipsoid(
    dim=(1.0, 1.0, 1.0), pos=(0.0, 0.0, 0.0), Nvert=15, min_vert=3, max_vert=20
) -> dict:
    """
    Provides the base plotly ellipsoid mesh3d parameters in a dictionary based on number of vertices
    of the circumference, the 3 dimensions and position.
    The zero position is in the barycenter of the vertices.
    `Nvert` will be forced automatically in the range [`min_vert`, `max_vert`]
    """
    N = min(max(Nvert, min_vert), max_vert)
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


def make_BaseCylinderSegment(d1=1, d2=2, h=1, phi1=0, phi2=90, Nvert=30) -> dict:
    """
    Provides the base plotly CylinderSegment mesh3d parameters in a dictionary based on inner
    and outer diameters, height, start angle and end angles in degrees.
    The zero position is in the barycenter of the vertices.
    """
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


def make_BaseCone(base_vertices=3, diameter=1, height=1, pos=(0.0, 0.0, 0.0)) -> dict:
    """
    Provides the base plotly Cone mesh3d parameters in a dictionary based on number of vertices of
    the base, the diameter the height and position.
    The zero position is in the barycenter of the vertices.
    """
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


def make_BaseArrow(base_vertices=30, diameter=0.3, height=1) -> dict:
    """
    Provides the base plotly 3D Arrow mesh3d parameters in a dictionary based on number of vertices
    of the base, the diameter the height and position.
    The zero position is in the barycenter of the vertices.
    """
    h, d = height, diameter
    cone = make_BaseCone(
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


def draw_arrow(vec, pos, sign=1) -> Tuple:
    """
    Provides x,y,z coordinates of an arrow drawn in the x-y-plane (z=0), showing up the y-axis and
    centered in x,y,z=(0,0,0). The arrow vertices are then turned in the direction of `vec` and
    moved to position `pos`.
    """
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
) -> dict:
    """
    Creates the plotly scatter3d parameters for a Line current in a dictionary based on the
    provided arguments
    """
    name = "Line Curent" if name is None else name
    name_suffix = (
        f" ({unit_prefix(current)}A)" if name_suffix is None else f" ({name_suffix})"
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
    """
    Creates the plotly scatter3d parameters for a Circular current in a dictionary based on the
    provided arguments
    """
    name = "Circular Curent" if name is None else name
    name_suffix = (
        f" ({unit_prefix(current)}A)" if name_suffix is None else f" ({name_suffix})"
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
) -> dict:
    """
    Creates the plotly scatter3d parameters for an object with no specifically supported
    representation. The object will be reprensented by a scatter point and text above with object
    name.
    """
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
) -> dict:
    """
    Creates the plotly mesh3d parameters for a Circular current in a dictionary based on the
    provided arguments
    """
    name = "Dipole" if name is None else name
    moment_mag = np.linalg.norm(moment)
    name_suffix = (
        f" (moment={unit_prefix(moment_mag)}T/m³)".format()
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
    mag = np.array((0, 0, 1))
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
) -> dict:
    """
    Creates the plotly mesh3d parameters for a Cuboid Magnet in a dictionary based on the
    provided arguments
    """
    name = "Cuboid" if name is None else name
    # pylint: disable=consider-using-f-string
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
    diameter=1.0,
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
) -> dict:
    """
    Creates the plotly mesh3d parameters for a Cylinder Magnet in a dictionary based on the
    provided arguments
    """
    name = "Cylinder" if name is None else name
    # pylint: disable=consider-using-f-string
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
    """
    Creates the plotly mesh3d parameters for a Cylinder Segment Magnet in a dictionary based on the
    provided arguments
    """
    name = "CylinderSegment" if name is None else name
    # pylint: disable=consider-using-f-string
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
) -> dict:
    """
    Creates the plotly mesh3d parameters for a Sphere Magnet in a dictionary based on the
    provided arguments
    """
    name = "Sphere" if name is None else name
    name_suffix = (
        f" (D={unit_prefix(diameter / 1000)}m)"
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


def make_Pixels(positions, size=1, shape="cube") -> dict:
    """
    Creates the plotly mesh3d parameters for Sensor pixels based on pixel positions and chosen size
    For now, only "cube" shape is provided.
    """
    if shape == "cube":
        pixels = [make_BaseCuboid(pos=p, dim=[size] * 3) for p in positions]
    else:
        raise NotImplementedError(
            "pixel shape parameter only supports `cube` at the moment"
        )
    return merge_mesh3d(*pixels)


def make_Sensor(
    pixel=(0.0, 0.0, 0.0),
    dim=(1.0, 1.0, 1.0),
    pos=(0.0, 0.0, 0.0),
    orientation=None,
    name=None,
    name_suffix=None,
    color=None,
    size_pixels=1,
    pixel_color=None,
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
    else:
        dim = pixel.max(axis=0) - pixel.min(axis=0)
        dim_ext = np.mean(dim)
    cube_mask = (vertices < 1).all(axis=1)
    vertices[cube_mask] = 0 * vertices[cube_mask]
    vertices[~cube_mask] = dim_ext * vertices[~cube_mask]
    x, y, z = vertices.T
    sensor.update(x=x, y=y, z=z)
    meshes_to_merge = [sensor]
    if pixel.ndim != 1:
        if size_pixels >= 0:
            if pixel_color is None:
                pixel_color = Config.PIXEL_COLOR
            combs = np.array(list(combinations(pixel, 2)))
            vecs = np.diff(combs, axis=1)
            dists = np.linalg.norm(vecs, axis=2)
            pixel_dim = np.min(dists) * size_pixels/2
            pixels_mesh = make_Pixels(positions=pixel, size=pixel_dim)
            pixels_mesh["facecolor"] = np.repeat(pixel_color, len(pixels_mesh["i"]))
            meshes_to_merge.append(pixels_mesh)
        hull_pos = 0.5 * (pixel.max(axis=0) + pixel.min(axis=0))
        dim[dim == 0] = pixel_dim / 2 if size_pixels != 0 else dim_ext / 10
        hull_mesh = make_BaseCuboid(pos=hull_pos, dim=dim)
        hull_mesh["facecolor"] = np.repeat(color, len(hull_mesh["i"]))
        meshes_to_merge.append(hull_mesh)
    sensor = merge_mesh3d(*meshes_to_merge)
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
    """
    Updates an existing plotly mesh3d dictionary of an object which has a magnetic vector. The
    object gets colorized, positioned and oriented based on provided arguments
    """
    vertices = np.array([mesh_dict[k] for k in "xyz"]).T
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
            vertices=vertices,
            axis=magnetization,
        )
    if orientation is not None:
        vertices = orientation.apply(vertices)
    x, y, z = (vertices + position).T
    mesh_dict.update(
        x=x, y=y, z=z, showscale=False, name=f"""{name}{name_suffix}""", **kwargs
    )
    return {**mesh_dict, **kwargs}


def merge_mesh3d(*traces):
    """
    Merges a list of plotly mesh3d dictionaries. The `i,j,k` index parameters need to cummulate the
    indices of each object in order to point to the right vertices in the concatenated vertices.
    `x,y,z,i,j,k` are mandatory fields, the `intensity` and `facecolor` parameters also get
    concatenated if they are present in all objects. All other parameter found in the dictionary
    keys are taken from the first object, other keys from further objects are ignored.
    """
    merged_trace = {}
    L = np.array([0] + [len(b["x"]) for b in traces[:-1]]).cumsum()
    for k in "ijk":
        if k in traces[0]:
            merged_trace[k] = np.hstack([b[k] + l for b, l in zip(traces, L)])
    for k in "xyz":
        merged_trace[k] = np.concatenate([b[k] for b in traces])
    for k in ("intensity", "facecolor"):
        if k in traces[0] and traces[0][k] is not None:
            merged_trace[k] = np.hstack([b[k] for b in traces])
    for k, v in traces[0].items():
        if k not in merged_trace:
            merged_trace[k] = v
    return merged_trace


def merge_scatter3d(*traces):
    """
    Merges a list of plotly scatter3d. `x,y,z` are mandatory fields and are concatenated with a
    `None` vertex to prevent line connection between objects to be concatenated. Keys are taken from
    the first object, other keys from further objects are ignored.
    """
    merged_trace = {}
    for k in "xyz":
        merged_trace[k] = np.hstack([pts for b in traces for pts in [[None], b[k]]])
    for k, v in traces[0].items():
        if k not in merged_trace:
            merged_trace[k] = v
    return merged_trace


def merge_traces(*traces):
    """
    Merges a list of plotly 3d-traces. Supported trace types are `mesh3d` and `scatter3d`.
    All traces have be of the same type when merging. Keys are taken from the first object, other
    keys from further objects are ignored.
    """
    if traces[0]["type"] == "mesh3d":
        trace = merge_mesh3d(*traces)
    elif traces[0]["type"] == "scatter3d":
        trace = merge_scatter3d(*traces)
    return trace


def getTraces(
    input_obj,
    show_path=False,
    show_direction=True,
    size_dipoles=1,
    size_sensors=1,
    size_pixels=1,
    show_path_numbering=False,
    color_transition=None,
    north_color=None,
    middle_color=None,
    south_color=None,
    pixel_color=None,
    **kwargs,
) -> list:
    """
    This is a helper function providing the plotly traces for any object of the magpylib library. In
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
    if not show_direction:
        color_transition = -1

    mag_color_kwargs = dict(
        color_transition=color_transition,
        north_color=north_color,
        middle_color=middle_color,
        south_color=south_color,
    )

    color = kwargs.get("color", None)
    traces = []
    if isinstance(input_obj, Markers):
        x, y, z = input_obj.markers.T
        trace = go.Scatter3d(
            name="Marker" if len(x) == 1 else f"Markers ({len(x)} points)",
            x=x,
            y=y,
            z=z,
            marker_color=input_obj.color,
            marker_symbol="x",
            marker_size=2,
            mode="markers",
        )
        traces.append(trace)
    else:
        if isinstance(input_obj, Sensor):
            kwargs.update(
                dim=getattr(input_obj, "dimension", size_sensors),
                pixel=getattr(input_obj, "pixel", (0.0, 0.0, 0.0)),
                size_pixels=size_pixels,
                pixel_color=pixel_color,
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
                height=input_obj.dimension[1],
                base_vertices=base_vertices,
                **mag_color_kwargs,
            )
            make_func = make_Cylinder
        elif isinstance(input_obj, CylinderSegment):
            Nvert = min(
                50, Config.ITER_CYLINDER
            )  # no need to render more than 50 vertices
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
                show_arrows=show_direction,
            )
            make_func = make_Line
        elif isinstance(input_obj, Circular):
            kwargs.update(
                diameter=input_obj.diameter,
                current=input_obj.current,
                show_arrows=show_direction,
            )
            make_func = make_Circular
        else:
            kwargs.update(name=type(input_obj).__name__)
            make_func = make_UnsupportedObject

        haspath = input_obj.position.ndim > 1
        if haspath:
            path_len = input_obj.position.shape[0]
            if show_path is True or show_path is False:
                inds = np.array([-1])
            elif isinstance(show_path, int):
                inds = np.arange(path_len, dtype=int)[::-show_path]
            elif hasattr(show_path, "__iter__") and not isinstance(show_path, str):
                inds = np.array(show_path)
            inds = inds[inds < path_len]
            if inds.size == 0:
                inds = np.array([path_len - 1])
            path_traces = []
            for pos, orient in zip(
                input_obj.position[inds], input_obj.orientation[inds]
            ):
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
    markers=None,
    show_path=False,
    fig=None,
    renderer=None,
    animate_time=3,
    animate_fps=30,
    zoom=1,
    backtofirst=False,
    color_discrete_sequence=None,
    **kwargs,
):

    """
    Display objects and paths graphically using the plotly library.

    Parameters
    ----------
    objects: sources, collections or sensors
        Objects to be displayed.

    fig: plotly Figure, default=None
        Display graphical output in a given figure:
        - plotly.graph_objects.Figure
        - plotly.graph_objects.FigureWidget
        By default a new `Figure` is created and displayed.

    markers: array_like, None, shape (N,3), default=None
        Display position markers in the global CS. By default no marker is displayed.

    show_direction: bool, default=True
        Set True to show magnetization and current directions. The `matplotlib`
        backend uses arrows and the plotly backend displays direction with a color
        gradient for magnets/dipoles and arrows for currents.

    show_path: bool or int or array_like, default=True
        Options True, False, positive int or iterable. By default object paths are shown. If
        show_path is a positive integer, objects will be displayed at multiple path
        positions along the path, in steps of show_path. If show_path is an iterable
        of integers, objects will be displayed for the provided indices.
        If show_path='animate, the plot will be animated according to the `animate_time`
        and 'animate_fps' parameters.

    size_sensor: float, default=1
        Adjust automatic display size of sensors.

    size_direction: float, default=1
        Adjust automatic display size of direction arrows.

    size_dipoles: float, default=1
        Adjust automatic display size of dipoles.

    size_pixels: float, default=1
        A positive number. Adjusts automatic display size of sensor pixels. When set to 0,
        pixels will be hidden, when greater than 0, pixels will occupy half the ratio of the minimum
        distance between any pixel of the same sensor, equal to `size_pixel`.

    zoom: float, default = 1
        Adjust plot zoom-level. When zoom=0 all objects are just inside the 3D-axes.
    backtofirst: bool, default = False
        If True, the last frame of the animation will be the first again.

    animate_time: float, default = 3
        Sets the animation duration

    animate_fps: float, default = 50
        This sets the maximum allowed frame rate. In case of path positions needed to be displayed
        exceeds the `animate_fps` the path position will be downsampled to be lower or equal
        the `animate_fps`. This is mainly depending on the pc/browser performance and is set to
        50 by default to avoid hanging the animation process.

    title: str, default = "3D-Paths Animation"
        When zoom=0 all objects are just inside the 3D-axes.

    color_discrete_sequence: list or array_like, iterable, default=
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

    Returns
    -------
    None: NoneType
    """
    show_fig = False
    if fig is None:
        show_fig = True
        fig = go.Figure()

    title = getattr(objs[0], "name", None) if len(objs) == 1 else None

    if markers is not None and markers:
        objs = list(objs) + [Markers(*markers)]

    if color_discrete_sequence is None:
        color_discrete_sequence = Config.COLOR_DISCRETE_SEQUENCE

    with fig.batch_update():
        if (
            not any(getattr(obj, "position", np.array([])).ndim > 1 for obj in objs)
            and show_path == "animate"
        ):  # check if some path exist for any object
            show_path = True
            warnings.warn("No path to be animated detected, displaying standard plot")
        if show_path == "animate":
            title = "3D-Paths Animation" if title is None else title
            animate_path(
                fig,
                objs,
                color_discrete_sequence,
                title=title,
                animate_time=animate_time,
                animate_fps=animate_fps,
                backtofirst=backtofirst,
                **kwargs,
            )
        else:
            traces_dicts = {
                obj: getTraces(obj, show_path=show_path, color=color, **kwargs)
                for obj, color in zip(objs, cycle(color_discrete_sequence))
            }
            traces = [t for tr in traces_dicts.values() for t in tr]
            fig.add_traces(traces)
            fig.update_layout(
                title_text=title,
                **{f"scene_{k}axis_title": f"{k} [mm]" for k in "xyz"},
                scene_aspectmode="data",
            )
        apply_fig_ranges(fig, zoom)
    if show_fig:
        fig.show(renderer=renderer)


def apply_fig_ranges(fig, zoom=1):
    """This is a helper function which applies the ranges properties of the provided `fig` object
    according to a certain zoom level. All three space direction will be equal and match the
    maximum of the ranges needed to display all objects, including their paths.

    Parameters
    ----------
    zoom: float, default = 1
        When zoom=0 all objects are just inside the 3D-axes.

    Returns
    -------
    None: NoneType
    """
    ranges = {k: [] for k in ("xyz")}
    frames = fig.frames if fig.frames else [fig]
    for frame in frames:
        for t in frame.data:
            for k, v in ranges.items():
                v.extend([np.min(t[k]), np.max(t[k])])
    r = np.array([[np.min(v), np.max(v)] for v in ranges.values()])
    size = np.diff(r, axis=1)
    m = size.max() / 2
    center = r.mean(axis=1)
    ranges = np.array([center - m * (1 + zoom), center + m * (1 + zoom)]).T
    fig.update_scenes(
        **{
            f"{k}axis": dict(range=ranges[i], autorange=False, title=f"{k} [mm]")
            for i, k in enumerate("xyz")
        },
        aspectratio={k: 1 for k in "xyz"},
        aspectmode="manual",
    )


def animate_path(
    fig,
    objs,
    color_discrete_sequence=None,
    title="3D-Paths Animation",
    animate_time=3,
    animate_fps=50,
    backtofirst=False,
    **kwargs,
):
    """This is a helper function which attaches plotly frames to the provided `fig` object
    according to a certain zoom level. All three space direction will be equal and match the
    maximum of the ranges needed to display all objects, including their paths.

    Parameters
    ----------
    backtofirst: bool, default = False
        If True, the last frame of the animation will be the first again.

    animate_time: float, default = 3
        Sets the animation duration

    animate_fps: float, default = 50
        This sets the maximum allowed frame rate. In case of path positions needed to be displayed
        exceeds the `animate_fps` the path position will be downsampled to be lower or equal
        the `animate_fps`. This is mainly depending on the pc/browser performance and is set to
        50 by default to avoid hanging the animation process.

    title: str, default = "3D-Paths Animation"
        When zoom=0 all objects are just inside the 3D-axes.

    color_discrete_sequence: list or array_like, iterable, default=
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

    Returns
    -------
    None: NoneTyp
    """
    # make sure the number of frames does not exceed the max frames and max frame rate
    # downsample if necessary
    path_lengths = [
        obj.position.shape[0] if obj.position.ndim > 1 else 0 for obj in objs
    ]
    N = max(path_lengths)
    maxpos = min(animate_time * animate_fps, Config.ANIMATE_MAX_FRAMES)
    if N <= maxpos:
        path_indices = np.arange(N)
    else:
        round_step = N / (maxpos - 1)
        ar = np.linspace(0, N, N, endpoint=False)
        path_indices = np.unique(np.floor(ar / round_step) * round_step).astype(
            int
        )  # downsampled indices
        path_indices[-1] = N - 1  # make sure the last frame is the last path position

    # calculate exponent of last frame index to avoid digit shift in
    # frame number display during animation
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
            for obj, color in zip(objs, cycle(color_discrete_sequence))
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

    # update fig
    frame_duration = int(animate_time * 1000 / path_indices.shape[0])
    fig.frames = frames
    fig.add_traces(frames[0].data)
    fig.update_layout(
        height=None,
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

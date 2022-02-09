""" Display function codes"""

from typing import Tuple
import numpy as np
from scipy.spatial.transform import Rotation as RotScipy
from magpylib._src.style import Markers


class MagpyMarkers:
    """A class that stores markers 3D-coordinates"""

    _object_type = "Marker"

    def __init__(self, *markers):
        self.style = Markers()
        self.markers = np.array(markers)


def place_and_orient_model3d(
    model_dict,
    orientation=None,
    position=None,
    coordsargs=None,
    scale=1,
    return_vertices=False,
    **kwargs,
):
    """places and orients mesh3d dict"""
    if orientation is None and position is None:
        return {**model_dict, **kwargs}
    position = (0.0, 0.0, 0.0) if position is None else position
    position = np.array(position, dtype=float)
    new_model_dict = {}
    if "args" in model_dict:
        new_model_dict["args"] = list(model_dict["args"])
    vertices = []
    if coordsargs is None:
        coordsargs = {"x": "x", "y": "y", "z": "z"}
    useargs = False
    for k in "xyz":
        key = coordsargs[k]
        if key.startswith("args"):
            useargs = True
            ind = int(key[5])
            v = model_dict["args"][ind]
        else:
            if key in model_dict:
                v = model_dict[key]
            else:
                raise ValueError(
                    "Rotating/Moving of provided model failed, trace dictionary "
                    f"has no argument {k!r}, use `coordsargs` to specify the names of the "
                    "coordinates to be used"
                )
        vertices.append(v)
    vertices = np.array(vertices).T
    if orientation is not None:
        vertices = orientation.apply(vertices)
    new_vertices = (vertices * scale + position).T
    for i, k in enumerate("xyz"):
        key = coordsargs[k]
        if useargs:
            ind = int(key[5])
            new_model_dict["args"][ind] = new_vertices[i]
        else:
            new_model_dict[key] = new_vertices[i]
    new_dict = {**model_dict, **new_model_dict, **kwargs}
    if return_vertices:
        return new_dict, new_vertices
    return new_dict


def draw_arrowed_line(vec, pos, sign=1, arrow_size=1) -> Tuple:
    """
    Provides x,y,z coordinates of an arrow drawn in the x-y-plane (z=0), showing up the y-axis and
    centered in x,y,z=(0,0,0). The arrow vertices are then turned in the direction of `vec` and
    moved to position `pos`.
    """
    norm = np.linalg.norm(vec)
    nvec = np.array(vec) / norm
    yaxis = np.array([0, 1, 0])
    cross = np.cross(nvec, yaxis)
    dot = np.dot(nvec, yaxis)
    n = np.linalg.norm(cross)
    if dot == -1:
        sign *= -1
    hy = sign * 0.1 * arrow_size
    hx = 0.06 * arrow_size
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
    if n != 0:
        t = np.arccos(dot)
        R = RotScipy.from_rotvec(-t * cross / n)
        arrow = R.apply(arrow)
    x, y, z = (arrow + pos).T
    return x, y, z


def draw_arrow_from_vertices(vertices, current, arrow_size):
    """returns scatter coordinates of arrows between input vertices"""
    vectors = np.diff(vertices, axis=0)
    positions = vertices[:-1] + vectors / 2
    vertices = np.concatenate(
        [
            draw_arrowed_line(vec, pos, np.sign(current), arrow_size=arrow_size)
            for vec, pos in zip(vectors, positions)
        ],
        axis=1,
    )

    return vertices


def draw_arrowed_circle(current, diameter, arrow_size, vert):
    """draws an oriented circle with an arrow"""
    t = np.linspace(0, 2 * np.pi, vert)
    x = np.cos(t)
    y = np.sin(t)
    if arrow_size != 0:
        hy = 0.2 * np.sign(current) * arrow_size
        hx = 0.15 * arrow_size
        x = np.hstack([x, [1 + hx, 1, 1 - hx]])
        y = np.hstack([y, [-hy, 0, -hy]])
    x = x * diameter / 2
    y = y * diameter / 2
    z = np.zeros(x.shape)
    vertices = np.array([x, y, z])
    return vertices


def get_rot_pos_from_path(obj, show_path=None):
    """
    subsets orientations and positions depending on `show_path` value.
    examples:
    show_path = [1,2,8], path_len = 6 -> path_indices = [1,2,6]
    returns rots[[1,2,6]], poss[[1,2,6]]
    """
    # pylint: disable=protected-access
    # pylint: disable=invalid-unary-operand-type
    if show_path is None:
        show_path = True
    pos = getattr(obj, "_position", None)
    if pos is None:
        pos = obj.position
    pos = np.array(pos)
    orient = getattr(obj, "_orientation", None)
    if orient is None:
        orient = getattr(obj, "orientation", None)
    if orient is None:
        orient = RotScipy.from_rotvec([[0, 0, 1]])
    pos = np.array([pos]) if pos.ndim == 1 else pos
    path_len = pos.shape[0]
    if show_path is True or show_path is False or show_path==0:
        inds = np.array([-1])
    elif isinstance(show_path, int):
        inds = np.arange(path_len, dtype=int)[::-show_path]
    elif hasattr(show_path, "__iter__") and not isinstance(show_path, str):
        inds = np.array(show_path)
    inds[inds >= path_len] = path_len - 1
    inds = np.unique(inds)
    if inds.size == 0:
        inds = np.array([path_len - 1])
    rots = orient[inds]
    poss = pos[inds]
    return rots, poss


def faces_cuboid(src, show_path):
    """
    compute vertices and faces of Cuboid input for plotting
    takes Cuboid source
    returns vert, faces
    returns all faces when show_path=all
    """
    # pylint: disable=protected-access
    a, b, c = src.dimension
    vert0 = np.array(
        (
            (0, 0, 0),
            (a, 0, 0),
            (0, b, 0),
            (0, 0, c),
            (a, b, 0),
            (a, 0, c),
            (0, b, c),
            (a, b, c),
        )
    )
    vert0 = vert0 - src.dimension / 2

    rots, poss = get_rot_pos_from_path(src, show_path)

    faces = []
    for rot, pos in zip(rots, poss):
        vert = rot.apply(vert0) + pos
        faces += [
            [vert[0], vert[1], vert[4], vert[2]],
            [vert[0], vert[1], vert[5], vert[3]],
            [vert[0], vert[2], vert[6], vert[3]],
            [vert[7], vert[6], vert[2], vert[4]],
            [vert[7], vert[6], vert[3], vert[5]],
            [vert[7], vert[5], vert[1], vert[4]],
        ]
    return faces


def faces_cylinder(src, show_path):
    """
    Compute vertices and faces of Cylinder input for plotting.

    Parameters
    ----------
    - src (source object)
    - show_path (bool or int)

    Returns
    -------
    vert, faces (returns all faces when show_path=int)
    """
    # pylint: disable=protected-access
    res = 15  # surface discretization

    # generate cylinder faces
    r, h2 = src.dimension / 2
    hs = np.array([-h2, h2])
    phis = np.linspace(0, 2 * np.pi, res)
    phis2 = np.roll(np.linspace(0, 2 * np.pi, res), 1)
    faces = [
        np.array(
            [
                (r * np.cos(p1), r * np.sin(p1), h2),
                (r * np.cos(p1), r * np.sin(p1), -h2),
                (r * np.cos(p2), r * np.sin(p2), -h2),
                (r * np.cos(p2), r * np.sin(p2), h2),
            ]
        )
        for p1, p2 in zip(phis, phis2)
    ]
    faces += [
        np.array([(r * np.cos(phi), r * np.sin(phi), h) for phi in phis]) for h in hs
    ]

    # add src attributes position and orientation depending on show_path
    rots, poss = get_rot_pos_from_path(src, show_path)

    # all faces (incl. along path) adding pos and rot
    all_faces = []
    for rot, pos in zip(rots, poss):
        for face in faces:
            all_faces += [[rot.apply(f) + pos for f in face]]

    return all_faces


def faces_cylinder_segment(src, show_path):
    """
    Compute vertices and faces of CylinderSegment for plotting.

    Parameters
    ----------
    - src (source object)
    - show_path (bool or int)

    Returns
    -------
    vert, faces (returns all faces when show_path=int)
    """
    # pylint: disable=protected-access
    res = 15  # surface discretization

    # generate cylinder segment faces
    r1, r2, h, phi1, phi2 = src.dimension
    res_tile = (
        int((phi2 - phi1) / 360 * 2 * res) + 2
    )  # resolution used for tile curved surface
    phis = np.linspace(phi1, phi2, res_tile) / 180 * np.pi
    phis2 = np.roll(phis, 1)
    faces = [
        np.array(
            [  # inner curved surface
                (r1 * np.cos(p1), r1 * np.sin(p1), h / 2),
                (r1 * np.cos(p1), r1 * np.sin(p1), -h / 2),
                (r1 * np.cos(p2), r1 * np.sin(p2), -h / 2),
                (r1 * np.cos(p2), r1 * np.sin(p2), h / 2),
            ]
        )
        for p1, p2 in zip(phis[1:], phis2[1:])
    ]
    faces += [
        np.array(
            [  # outer curved surface
                (r2 * np.cos(p1), r2 * np.sin(p1), h / 2),
                (r2 * np.cos(p1), r2 * np.sin(p1), -h / 2),
                (r2 * np.cos(p2), r2 * np.sin(p2), -h / 2),
                (r2 * np.cos(p2), r2 * np.sin(p2), h / 2),
            ]
        )
        for p1, p2 in zip(phis[1:], phis2[1:])
    ]
    faces += [
        np.array(
            [  # sides
                (r1 * np.cos(p), r1 * np.sin(p), h / 2),
                (r2 * np.cos(p), r2 * np.sin(p), h / 2),
                (r2 * np.cos(p), r2 * np.sin(p), -h / 2),
                (r1 * np.cos(p), r1 * np.sin(p), -h / 2),
            ]
        )
        for p in [phis[0], phis[-1]]
    ]
    faces += [
        np.array(  # top surface
            [(r1 * np.cos(p), r1 * np.sin(p), h / 2) for p in phis]
            + [(r2 * np.cos(p), r2 * np.sin(p), h / 2) for p in phis[::-1]]
        )
    ]
    faces += [
        np.array(  # bottom surface
            [(r1 * np.cos(p), r1 * np.sin(p), -h / 2) for p in phis]
            + [(r2 * np.cos(p), r2 * np.sin(p), -h / 2) for p in phis[::-1]]
        )
    ]

    # add src attributes position and orientation depending on show_path
    rots, poss = get_rot_pos_from_path(src, show_path)

    # all faces (incl. along path) adding pos and rot
    all_faces = []
    for rot, pos in zip(rots, poss):
        for face in faces:
            all_faces += [[rot.apply(f) + pos for f in face]]

    return all_faces


def faces_sphere(src, show_path):
    """
    Compute vertices and faces of Sphere input for plotting.

    Parameters
    ----------
    - src (source object)
    - show_path (bool or int)

    Returns
    -------
    vert, faces (returns all faces when show_path=int)
    """
    # pylint: disable=protected-access
    res = 15  # surface discretization

    # generate sphere faces
    r = src.diameter / 2
    phis = np.linspace(0, 2 * np.pi, res)
    phis2 = np.roll(np.linspace(0, 2 * np.pi, res), 1)
    ths = np.linspace(0, np.pi, res)
    faces = [
        r
        * np.array(
            [
                (np.cos(p) * np.sin(t1), np.sin(p) * np.sin(t1), np.cos(t1)),
                (np.cos(p) * np.sin(t2), np.sin(p) * np.sin(t2), np.cos(t2)),
                (np.cos(p2) * np.sin(t2), np.sin(p2) * np.sin(t2), np.cos(t2)),
                (np.cos(p2) * np.sin(t1), np.sin(p2) * np.sin(t1), np.cos(t1)),
            ]
        )
        for p, p2 in zip(phis, phis2)
        for t1, t2 in zip(ths[1:-2], ths[2:-1])
    ]
    faces += [
        r
        * np.array(
            [(np.cos(p) * np.sin(th), np.sin(p) * np.sin(th), np.cos(th)) for p in phis]
        )
        for th in [ths[1], ths[-2]]
    ]

    # add src attributes position and orientation depending on show_path
    rots, poss = get_rot_pos_from_path(src, show_path)

    # all faces (incl. along path) adding pos and rot
    all_faces = []
    for rot, pos in zip(rots, poss):
        for face in faces:
            all_faces += [[rot.apply(f) + pos for f in face]]

    return all_faces


def system_size(points):
    """compute system size for display"""
    # determine min/max from all to generate aspect=1 plot
    if points:
        pts = np.vstack(points)
        xs = [np.amin(pts[:, 0]), np.amax(pts[:, 0])]
        ys = [np.amin(pts[:, 1]), np.amax(pts[:, 1])]
        zs = [np.amin(pts[:, 2]), np.amax(pts[:, 2])]

        xsize = xs[1] - xs[0]
        ysize = ys[1] - ys[0]
        zsize = zs[1] - zs[0]

        xcenter = (xs[1] + xs[0]) / 2
        ycenter = (ys[1] + ys[0]) / 2
        zcenter = (zs[1] + zs[0]) / 2

        size = max([xsize, ysize, zsize])

        limx0 = xcenter + size / 2
        limx1 = xcenter - size / 2
        limy0 = ycenter + size / 2
        limy1 = ycenter - size / 2
        limz0 = zcenter + size / 2
        limz1 = zcenter - size / 2
    else:
        limx0, limx1, limy0, limy1, limz0, limz1 = -1, 1, -1, 1, -1, 1
    return limx0, limx1, limy0, limy1, limz0, limz1

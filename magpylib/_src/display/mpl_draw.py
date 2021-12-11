""" matplotlib draw-functionalities"""

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from magpylib._src.default_classes import default_settings as Config
from magpylib._src.display.disp_utility import (
    get_rot_pos_from_path,
    draw_arrow_from_vertices,
    draw_arrowed_circle,
    place_and_orient_model3d,
)


def draw_directs_faced(faced_objects, colors, ax, show_path, size_direction):
    """draw direction of magetization of faced magnets

    Parameters
    ----------
    - faced_objects(list of src objects): with magnetization vector to be drawn
    - colors: colors of faced_objects
    - ax(Pyplot 3D axis): to draw in
    - show_path(bool or int): draw on every position where object is displayed
    """
    # pylint: disable=protected-access
    # pylint: disable=too-many-branches

    for col, obj in zip(colors, faced_objects):

        # add src attributes position and orientation depending on show_path
        rots, poss = get_rot_pos_from_path(obj, show_path)

        # vector length, color and magnetization
        if obj._object_type in ("Cuboid", "Cylinder"):
            length = 1.8 * np.amax(obj.dimension)
        elif obj._object_type == "CylinderSegment":
            length = 1.8 * np.amax(obj.dimension[:3])  # d1,d2,h
        else:
            length = 1.8 * obj.diameter  # Sphere
        mag = obj.magnetization

        # collect all draw positions and directions
        draw_pos, draw_direc = [], []
        for rot, pos in zip(rots, poss):
            if (
                obj._object_type == "CylinderSegment"
            ):  # change cylinder_tile draw_pos to geo center
                odim = obj.dimension
                r1, r2, _, phi1, phi2 = odim
                phi_mid = (phi1 + phi2) / 2 * np.pi / 180
                r_mid = (r2 + r1) / 2
                shift = r_mid * np.array([np.cos(phi_mid), np.sin(phi_mid), 0])
                shift = rot.apply(shift)
                draw_pos += [pos + shift]
            else:
                draw_pos += [pos]
            direc = mag / (np.linalg.norm(mag) + 1e-6)
            draw_direc += [rot.apply(direc)]
        draw_pos = np.array(draw_pos)
        draw_direc = np.array(draw_direc)

        # use quiver() separately for each object to easier control
        # color and vector length
        ax.quiver(
            draw_pos[:, 0],
            draw_pos[:, 1],
            draw_pos[:, 2],
            draw_direc[:, 0],
            draw_direc[:, 1],
            draw_direc[:, 2],
            length=length * size_direction,
            color=col,
        )


def draw_markers(markers, ax, color, symbol, size):
    """name = programm"""
    ax.plot(
        markers[:, 0],
        markers[:, 1],
        markers[:, 2],
        color=color,
        ls="",
        marker=symbol,
        ms=size,
    )


def draw_path(
    obj, col, marker_symbol, marker_size, marker_color, line_style, line_width, ax
):
    """draw path in given color and return list of path-points"""
    # pylint: disable=protected-access
    path = obj._position
    if len(path) > 1:
        ax.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            ls=line_style,
            lw=line_width,
            color=col,
            marker=marker_symbol,
            mfc=marker_color,
            mec=marker_color,
            ms=marker_size,
        )
        ax.plot(
            [path[0, 0]], [path[0, 1]], [path[0, 2]], marker="o", ms=4, mfc=col, mec="k"
        )
    return list(path)


def draw_faces(faces, col, lw, alpha, ax):
    """draw faces in respective color and return list of vertex-points"""
    cuboidf = Poly3DCollection(
        faces,
        facecolors=col,
        linewidths=lw,
        edgecolors="k",
        alpha=alpha,
    )
    ax.add_collection3d(cuboidf)
    return faces


def draw_pixel(sensors, ax, col, pixel_col, pixel_size, pixel_symb, show_path):
    """draw pixels and return a list of pixel-points in gloabl CS"""
    # pylint: disable=protected-access

    # collect sensor and pixel positions in global CS
    pos_sens, pos_pixel = [], []
    for sens in sensors:
        rots, poss = get_rot_pos_from_path(sens, show_path)

        pos_pixel_flat = np.reshape(sens.pixel, (-1, 3))

        for rot, pos in zip(rots, poss):
            pos_sens += [pos]

            for pix in pos_pixel_flat:
                pos_pixel += [pos + rot.apply(pix)]

    pos_all = pos_sens + pos_pixel
    pos_pixel = np.array(pos_pixel)

    # display pixel positions
    ax.plot(
        pos_pixel[:, 0],
        pos_pixel[:, 1],
        pos_pixel[:, 2],
        marker=pixel_symb,
        mfc=pixel_col,
        mew=pixel_size,
        mec=col,
        ms=pixel_size * 4,
        ls="",
    )

    # return all positions for system size evaluation
    return list(pos_all)


def draw_sensors(sensors, ax, sys_size, show_path, size):
    """draw sensor cross"""
    # pylint: disable=protected-access

    # collect plot data
    possis, exs, eys, ezs = [], [], [], []
    for sens in sensors:
        rots, poss = get_rot_pos_from_path(sens, show_path)

        for rot, pos in zip(rots, poss):
            possis += [pos]
            exs += [rot.apply((1, 0, 0))]
            eys += [rot.apply((0, 1, 0))]
            ezs += [rot.apply((0, 0, 1))]

    possis = np.array(possis)
    exs = np.array(exs)
    eys = np.array(eys)
    ezs = np.array(ezs)

    # quiver plot of basis vectors
    arrowlength = sys_size * size / Config.display.autosizefactor
    for col, es in zip(["r", "g", "b"], [exs, eys, ezs]):
        ax.quiver(
            possis[:, 0],
            possis[:, 1],
            possis[:, 2],
            es[:, 0],
            es[:, 1],
            es[:, 2],
            color=col,
            length=arrowlength,
        )


def draw_dipoles(dipoles, ax, sys_size, show_path, size, color, pivot):
    """draw dipoles"""
    # pylint: disable=protected-access

    # collect plot data
    possis, moms = [], []
    for dip in dipoles:
        rots, poss = get_rot_pos_from_path(dip, show_path)

        mom = dip.moment / np.linalg.norm(dip.moment)

        for rot, pos in zip(rots, poss):
            possis += [pos]
            moms += [rot.apply(mom)]

    possis = np.array(possis)
    moms = np.array(moms)

    # quiver plot of basis vectors
    arrowlength = sys_size * size / Config.display.autosizefactor
    ax.quiver(
        possis[:, 0],
        possis[:, 1],
        possis[:, 2],
        moms[:, 0],
        moms[:, 1],
        moms[:, 2],
        color=color,
        length=arrowlength,
        pivot=pivot,  # {'tail', 'middle', 'tip'},
    )


def draw_circular(circulars, show_path, col, size, width, ax):
    """draw circulars and return a list of positions"""
    # pylint: disable=protected-access

    # graphical settings
    discret = 72 + 1
    lw = width

    draw_pos = []  # line positions
    for circ in circulars:

        # add src attributes position and orientation depending on show_path
        rots, poss = get_rot_pos_from_path(circ, show_path)

        # init orientation line positions
        vertices = draw_arrowed_circle(circ.current, circ.diameter, size, discret).T
        # apply pos and rot, draw, store line positions
        for rot, pos in zip(rots, poss):
            possis1 = rot.apply(vertices) + pos
            ax.plot(possis1[:, 0], possis1[:, 1], possis1[:, 2], color=col, lw=lw)
            draw_pos += list(possis1)

    return draw_pos


def draw_line(lines, show_path, col, size, width, ax) -> list:
    """draw lines and return a list of positions"""
    # pylint: disable=protected-access

    # graphical settings
    lw = width

    draw_pos = []  # line positions
    for line in lines:

        # add src attributes position and orientation depending on show_path
        rots, poss = get_rot_pos_from_path(line, show_path)

        # init orientation line positions
        if size != 0:
            vertices = draw_arrow_from_vertices(line.vertices, line.current, size)
        else:
            vertices = np.array(line.vertices).T
        # apply pos and rot, draw, store line positions
        for rot, pos in zip(rots, poss):
            possis1 = rot.apply(vertices.T) + pos
            ax.plot(possis1[:, 0], possis1[:, 1], possis1[:, 2], color=col, lw=lw)
            draw_pos += list(possis1)

    return draw_pos


def draw_model3d_extra(obj, style, show_path, ax, color):
    """positions, orients and draws extra 3d model including path positions"""
    extra_model3d_traces = (
        style.model3d.extra if style.model3d.extra is not None else []
    )
    extra_model3d_traces = [
        t for t in extra_model3d_traces if t.backend == "matplotlib"
    ]
    path_traces_extra = {}
    for orient, pos in zip(*get_rot_pos_from_path(obj, show_path)):
        for extr in extra_model3d_traces:
            if extr.show:
                trace3d = place_and_orient_model3d(
                    extr.trace,
                    orientation=orient,
                    position=pos,
                )
                ttype = extr.trace["type"]
                if ttype not in path_traces_extra:
                    path_traces_extra[ttype] = []
                path_traces_extra[ttype].append(trace3d)

    for traces_extra in path_traces_extra.values():
        for tr in traces_extra:
            kwargs = {"color": color}
            kwargs.update({k: v for k, v in tr.items() if k != "type"})
            getattr(ax, tr["type"])(**kwargs)

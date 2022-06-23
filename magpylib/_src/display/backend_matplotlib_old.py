""" matplotlib draw-functionalities"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from magpylib._src.defaults.defaults_classes import default_settings as Config
from magpylib._src.display.traces_utility import draw_arrow_from_vertices
from magpylib._src.display.traces_utility import draw_arrowed_circle
from magpylib._src.display.traces_utility import get_flatten_objects_properties
from magpylib._src.display.traces_utility import get_rot_pos_from_path
from magpylib._src.display.traces_utility import MagpyMarkers
from magpylib._src.display.traces_utility import place_and_orient_model3d
from magpylib._src.input_checks import check_excitations
from magpylib._src.style import get_style


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

    rots, poss, _ = get_rot_pos_from_path(src, show_path)

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
    rots, poss, _ = get_rot_pos_from_path(src, show_path)

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
    rots, poss, _ = get_rot_pos_from_path(src, show_path)

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
    rots, poss, _ = get_rot_pos_from_path(src, show_path)

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

        # bring (n,m,3) point dimensions (e.g. from plot_surface body)
        #    to correct (n,3) shape
        for i, p in enumerate(points):
            if p.ndim == 3:
                points[i] = np.reshape(p, (-1, 3))

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


def draw_directs_faced(faced_objects, colors, ax, show_path, size_direction):
    """draw direction of magnetization of faced magnets

    Parameters
    ----------
    - faced_objects(list of src objects): with magnetization vector to be drawn
    - colors: colors of faced_objects
    - ax(Pyplot 3D axis): to draw in
    - show_path(bool or int): draw on every position where object is displayed
    """
    # pylint: disable=protected-access
    # pylint: disable=too-many-branches
    points = []
    for col, obj in zip(colors, faced_objects):

        # add src attributes position and orientation depending on show_path
        rots, poss, inds = get_rot_pos_from_path(obj, show_path)

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
        for rot, pos, ind in zip(rots, poss, inds):
            if obj._object_type == "CylinderSegment":
                # change cylinder_tile draw_pos to barycenter
                pos = obj._barycenter[ind]
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
        arrow_tip_pos = ((draw_direc * length * size_direction) + draw_pos)[0]
        points.append(arrow_tip_pos)
    return points


def draw_markers(markers, ax, color, symbol, size):
    """draws magpylib markers"""
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
    cuboid_faces = Poly3DCollection(
        faces,
        facecolors=col,
        linewidths=lw,
        edgecolors="k",
        alpha=alpha,
    )
    ax.add_collection3d(cuboid_faces)
    return faces


def draw_pixel(sensors, ax, col, pixel_col, pixel_size, pixel_symb, show_path):
    """draw pixels and return a list of pixel-points in global CS"""
    # pylint: disable=protected-access

    # collect sensor and pixel positions in global CS
    pos_sens, pos_pixel = [], []
    for sens in sensors:
        rots, poss, _ = get_rot_pos_from_path(sens, show_path)

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


def draw_sensors(sensors, ax, sys_size, show_path, size, arrows_style):
    """draw sensor cross"""
    # pylint: disable=protected-access
    arrowlength = sys_size * size / Config.display.autosizefactor

    # collect plot data
    possis, exs, eys, ezs = [], [], [], []
    for sens in sensors:
        rots, poss, _ = get_rot_pos_from_path(sens, show_path)

        for rot, pos in zip(rots, poss):
            possis += [pos]
            exs += [rot.apply((1, 0, 0))]
            eys += [rot.apply((0, 1, 0))]
            ezs += [rot.apply((0, 0, 1))]

        possis = np.array(possis)
        coords = np.array([exs, eys, ezs])

        # quiver plot of basis vectors
        arrow_colors = (
            arrows_style.x.color,
            arrows_style.y.color,
            arrows_style.z.color,
        )
        arrow_show = (arrows_style.x.show, arrows_style.y.show, arrows_style.z.show)
        for acol, ashow, es in zip(arrow_colors, arrow_show, coords):
            if ashow:
                ax.quiver(
                    possis[:, 0],
                    possis[:, 1],
                    possis[:, 2],
                    es[:, 0],
                    es[:, 1],
                    es[:, 2],
                    color=acol,
                    length=arrowlength,
                )


def draw_dipoles(dipoles, ax, sys_size, show_path, size, color, pivot):
    """draw dipoles"""
    # pylint: disable=protected-access

    # collect plot data
    possis, moms = [], []
    for dip in dipoles:
        rots, poss, _ = get_rot_pos_from_path(dip, show_path)

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
        rots, poss, _ = get_rot_pos_from_path(circ, show_path)

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
        rots, poss, _ = get_rot_pos_from_path(line, show_path)

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
    """positions, orients and draws extra 3d model including path positions
    returns True if at least one the traces is now new default"""
    extra_model3d_traces = style.model3d.data if style.model3d.data is not None else []
    points = []
    rots, poss, _ = get_rot_pos_from_path(obj, show_path)
    for orient, pos in zip(rots, poss):
        for extr in extra_model3d_traces:
            if extr.show:
                extr.update(extr.updatefunc())
                if extr.backend == "matplotlib":
                    kwargs = extr.kwargs() if callable(extr.kwargs) else extr.kwargs
                    args = extr.args() if callable(extr.args) else extr.args
                    kwargs, args, vertices = place_and_orient_model3d(
                        model_kwargs=kwargs,
                        model_args=args,
                        orientation=orient,
                        position=pos,
                        coordsargs=extr.coordsargs,
                        scale=extr.scale,
                        return_vertices=True,
                        return_model_args=True,
                    )
                    points.append(vertices.T)
                    if "color" not in kwargs or kwargs["color"] is None:
                        kwargs.update(color=color)
                    getattr(ax, extr.constructor)(*args, **kwargs)
    return points


def display_matplotlib_old(
    *obj_list_semi_flat,
    canvas=None,
    markers=None,
    zoom=0,
    colorsequence=None,
    animation=False,
    **kwargs,
):
    """Display objects and paths graphically with the matplotlib backend."""
    # pylint: disable=protected-access
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements

    # apply config default values if None
    # create or set plotting axis

    if animation is not False:
        msg = "The matplotlib backend does not support animation at the moment.\n"
        msg += "Use `backend=plotly` instead."
        warnings.warn(msg)
        # animation = False

    axis = canvas
    if axis is None:
        fig = plt.figure(dpi=80, figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect((1, 1, 1))
        generate_output = True
    else:
        ax = axis
        generate_output = False

    # draw objects and evaluate system size --------------------------------------

    # draw faced objects and store vertices
    points = []
    dipoles = []
    sensors = []
    markers_list = [o for o in obj_list_semi_flat if isinstance(o, MagpyMarkers)]
    obj_list_semi_flat = [o for o in obj_list_semi_flat if o not in markers_list]
    flat_objs_props = get_flatten_objects_properties(
        *obj_list_semi_flat, colorsequence=colorsequence
    )
    for obj, props in flat_objs_props.items():
        color = props["color"]
        style = get_style(obj, Config, **kwargs)
        path_frames = style.path.frames
        if path_frames is None:
            path_frames = True
        obj_color = style.color if style.color is not None else color
        lw = 0.25
        faces = None
        if obj.style.model3d.data:
            pts = draw_model3d_extra(obj, style, path_frames, ax, obj_color)
            points += pts
        if obj.style.model3d.showdefault:
            if obj._object_type == "Cuboid":
                lw = 0.5
                faces = faces_cuboid(obj, path_frames)
            elif obj._object_type == "Cylinder":
                faces = faces_cylinder(obj, path_frames)
            elif obj._object_type == "CylinderSegment":
                faces = faces_cylinder_segment(obj, path_frames)
            elif obj._object_type == "Sphere":
                faces = faces_sphere(obj, path_frames)
            elif obj._object_type == "Line":
                if style.arrow.show:
                    check_excitations([obj])
                arrow_size = style.arrow.size if style.arrow.show else 0
                arrow_width = style.arrow.width
                points += draw_line(
                    [obj], path_frames, obj_color, arrow_size, arrow_width, ax
                )
            elif obj._object_type == "Loop":
                if style.arrow.show:
                    check_excitations([obj])
                arrow_width = style.arrow.width
                arrow_size = style.arrow.size if style.arrow.show else 0
                points += draw_circular(
                    [obj], path_frames, obj_color, arrow_size, arrow_width, ax
                )
            elif obj._object_type == "Sensor":
                sensors.append((obj, obj_color))
                points += draw_pixel(
                    [obj],
                    ax,
                    obj_color,
                    style.pixel.color,
                    style.pixel.size,
                    style.pixel.symbol,
                    path_frames,
                )
            elif obj._object_type == "Dipole":
                dipoles.append((obj, obj_color))
                points += [obj.position]
            elif obj._object_type == "CustomSource":
                draw_markers(
                    np.array([obj.position]), ax, obj_color, symbol="*", size=10
                )
                label = (
                    obj.style.label
                    if obj.style.label is not None
                    else str(type(obj).__name__)
                )
                ax.text(*obj.position, label, horizontalalignment="center")
                points += [obj.position]
            if faces is not None:
                alpha = style.opacity
                pts = draw_faces(faces, obj_color, lw, alpha, ax)
                points += [np.vstack(pts).reshape(-1, 3)]
                if style.magnetization.show:
                    check_excitations([obj])
                    pts = draw_directs_faced(
                        [obj],
                        [obj_color],
                        ax,
                        path_frames,
                        style.magnetization.size,
                    )
                    points += pts
        if style.path.show:
            marker, line = style.path.marker, style.path.line
            points += draw_path(
                obj,
                obj_color,
                marker.symbol,
                marker.size,
                marker.color,
                line.style,
                line.width,
                ax,
            )

    # markers -------------------------------------------------------
    if markers_list:
        markers_instance = markers_list[0]
        style = get_style(markers_instance, Config, **kwargs)
        markers = np.array(markers_instance.markers)
        s = style.marker
        draw_markers(markers, ax, s.color, s.symbol, s.size)
        points += [markers]

    # draw direction arrows (based on src size) -------------------------
    # objects with faces

    # determine system size -----------------------------------------
    limx1, limx0, limy1, limy0, limz1, limz0 = system_size(points)

    # make sure ranges are not null
    limits = np.array([[limx0, limx1], [limy0, limy1], [limz0, limz1]])
    limits[np.squeeze(np.diff(limits)) == 0] += np.array([-1, 1])
    sys_size = np.max(np.diff(limits))
    c = limits.mean(axis=1)
    m = sys_size.max() / 2
    ranges = np.array([c - m * (1 + zoom), c + m * (1 + zoom)]).T

    # draw all system sized based quantities -------------------------

    # not optimal for loop if many sensors/dipoles
    for sens in sensors:
        sensor, color = sens
        style = get_style(sensor, Config, **kwargs)
        draw_sensors([sensor], ax, sys_size, path_frames, style.size, style.arrows)
    for dip in dipoles:
        dipole, color = dip
        style = get_style(dipole, Config, **kwargs)
        draw_dipoles(
            [dipole], ax, sys_size, path_frames, style.size, color, style.pivot
        )

    # plot styling --------------------------------------------------
    ax.set(
        **{f"{k}label": f"{k} [mm]" for k in "xyz"},
        **{f"{k}lim": r for k, r in zip("xyz", ranges)},
    )

    # generate output ------------------------------------------------
    if generate_output:
        plt.show()
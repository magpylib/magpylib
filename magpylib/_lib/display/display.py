""" Display function codes"""

import numpy as np
import matplotlib.pyplot as plt
from magpylib._lib.utility import format_obj_input, test_path_format
from magpylib._lib.display.mpl_draw import (
    draw_directs_faced,
    draw_faces,
    draw_markers,
    draw_path,
    draw_pixel,
    draw_sensors,
    draw_dipoles,
    draw_circular,
    draw_line,
)
from magpylib._lib.display.disp_utility import (
    faces_cuboid,
    faces_cylinder,
    system_size,
    faces_sphere,
    faces_cylinder_section,
)
from magpylib._lib.input_checks import check_excitations, check_dimensions
from magpylib._lib.config import Config


# ON INTERFACE
def display(
    *objects,
    markers=None,
    canvas=None,
    show_direction=True,
    show_path=True,
    size_sensors=1,
    size_direction=1,
    size_dipoles=1,
    size_pixels=1,
    zoom=1,
    plotting_backend=None,
    **kwargs,
):
    """
    Display objects and paths graphically.

    Parameters
    ----------
    objects: sources, collections or sensors
        Objects to be displayed.

    canvas: pyplot Axis or plotly Figure, default=None
        Display graphical output in a given canvas:
        - with matplotlib: pyplot axis (must be 3D).
        - with plotly: plotly.graph_objects.Figure
            or plotly.graph_objects.FigureWidget
        By default a new canvas is created and displayed.

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

    size_sensors: float, default=1
        Adjust automatic display size of sensors.

    size_direction: float, default=1
        Adjust automatic display size of direction arrows.

    size_dipoles: float, default=1
        Adjust automatic display size of dipoles.

    size_pixels: float, default=1
        A positive number. Adjusts automatic display size of sensor pixels. When set to 0,
        pixels will be hidden, when greater than 0, pixels will occupy half the ratio of the minimum
        distance between any pixel of the same sensor, equal to `size_pixel`.

    plotting_backend: default=None
        One of 'matplotlib', 'plolty'. If not set, parameter will default to
        Config.PLOTTING_BACKEND

    zoom: float, default = 1
        Adjust plot zoom-level. When zoom=0 all objects are just inside the 3D-axes.

    Returns
    -------
    None: NoneType

    Examples
    --------

    Display multiple objects, object paths, markers in 3D using Matplotlib:

    >>> import magpylib as magpy
    >>> col = magpy.Collection(
        [magpy.magnet.Sphere(magnetization=(0,0,1), diameter=1) for _ in range(3)])
    >>> for displ,src in zip([(.1414,0,0),(-.1,-.1,0),(-.1,.1,0)], col):
    >>>     src.move([displ]*50, increment=True)
    >>>     src.rotate_from_angax(angle=[10]*50, axis='z', anchor=0, start=0, increment=True)
    >>> ts = [-.6,-.4,-.2,0,.2,.4,.6]
    >>> sens = magpy.Sensor(position=(0,0,2), pixel=[(x,y,0) for x in ts for y in ts])
    >>> magpy.display(col, sens)
    --> graphic output

    Display figure on your own 3D Matplotlib axis:

    >>> import matplotlib.pyplot as plt
    >>> import magpylib as magpy
    >>> my_axis = plt.axes(projection='3d')
    >>> magnet = magpy.magnet.Cuboid(magnetization=(0,0,1), dimension=(1,2,3))
    >>> sens = magpy.Sensor(position=(0,0,3))
    >>> magpy.display(magnet, sens, axis=my_axis)
    >>> plt.show()
    --> graphic output

    """

    # flatten input
    obj_list = format_obj_input(objects)

    # test if all source dimensions and excitations (if sho_direc=True) have been initialized
    check_dimensions(obj_list)
    if show_direction:
        check_excitations(obj_list)

    # test if every individual obj_path is good
    test_path_format(obj_list)
    check_show_path = (
        isinstance(show_path, (int, bool))
        or show_path == "animate"
        or (hasattr(show_path, "__iter__") and not isinstance(show_path, str))
    )
    assert check_show_path, (
        f"`show_path` argument of type {type(show_path)} is invalid, \n"
        "it must be one of (True, False, 'animate'), a positive path index "
        "or an Iterable of path indices."
    )

    if plotting_backend is None:
        plotting_backend = Config.PLOTTING_BACKEND

    if plotting_backend == "matplotlib":
        assert (
            show_path != "animate"
        ), "the matplotlib backend does not support animation"
        display_matplotlib(
            *obj_list,
            axis=canvas,
            markers=markers,
            show_path=show_path,
            show_direction=show_direction,
            size_direction=size_direction,
            size_sensors=size_sensors,
            size_dipoles=size_dipoles,
            zoom=zoom,
        )
    elif plotting_backend == "plotly":
        # pylint: disable=import-outside-toplevel
        from magpylib._lib.display.plotly_draw import display_plotly

        display_plotly(
            *obj_list,
            fig=canvas,
            markers=markers,
            show_path=show_path,
            show_direction=show_direction,
            size_sensors=size_sensors,
            size_dipoles=size_dipoles,
            size_pixels=size_pixels,
            zoom=zoom,
            **kwargs,
        )


def display_matplotlib(
    *obj_list,
    axis=None,
    markers=None,
    show_path=True,
    show_direction=True,
    size_sensors=1,
    size_direction=1,
    size_dipoles=1,
    zoom=1,
):
    """
    Display objects and paths graphically with the matplotlib backend.

    Parameters
    ----------
    objects: sources, collections or sensors
        Objects to be displayed.

    axis: pyplot.axis, default=None
        Display graphical output in a given pyplot axis (must be 3D). By default a new
        pyplot figure is created and displayed.

    markers: array_like, shape (N,3), default=[(0,0,0)]
        Display position markers in the global CS. By default no marker is displayed.

    show_direction: bool, default=True
        Set True to show magnetization and current directions with quiver arrows

    show_path: bool or int or array_like, default=True
        Options True, False, positive int or iterable. By default object paths are shown. If
        show_path is a positive integer, objects will be displayed at multiple path
        positions along the path, in steps of show_path. If show_path is an iterable
        of integers, objects will be displayed for the provided indices.

    size_sensors: float, default=1
        Adjust automatic display size of sensors.

    size_direction: float, default=1
        Adjust automatic display size of direction arrows.

    size_dipoles: float, default=1
        Adjust automatic display size of dipoles.

    zoom: float, default=1
        Adjust plot zoom-level. When zoom=0 all objects are just inside the 3D-axes.

    Returns
    -------
    None: NoneType

    Examples
    --------

    Display multiple objects, object paths, markers in 3D using Matplotlib:

    >>> import magpylib as magpy
    >>> col = magpy.Collection(
        [magpy.magnet.Sphere(magnetization=(0,0,1), diameter=1) for _ in range(3)])
    >>> for displ,src in zip([(.1414,0,0),(-.1,-.1,0),(-.1,.1,0)], col):
    >>>     src.move([displ]*50, increment=True)
    >>>     src.rotate_from_angax(angle=[10]*50, axis='z', anchor=0, start=0, increment=True)
    >>> ts = [-.6,-.4,-.2,0,.2,.4,.6]
    >>> sens = magpy.Sensor(position=(0,0,2), pixel=[(x,y,0) for x in ts for y in ts])
    >>> magpy.display(col, sens)
    --> graphic output

    Display figure on your own 3D Matplotlib axis:

    >>> import matplotlib.pyplot as plt
    >>> import magpylib as magpy
    >>> my_axis = plt.axes(projection='3d')
    >>> magnet = magpy.magnet.Cuboid(magnetization=(0,0,1), dimension=(1,2,3))
    >>> sens = magpy.Sensor(position=(0,0,3))
    >>> magpy.display(magnet, sens, axis=my_axis)
    >>> plt.show()
    --> graphic output

    """
    # pylint: disable=protected-access
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements

    # objects with faces
    faced_objects = [
        obj
        for obj in obj_list
        if obj._object_type in ("Cuboid", "Cylinder", "CylinderSegment", "Sphere")
    ]

    # sensors
    sensors = [obj for obj in obj_list if obj._object_type == "Sensor"]

    # dipoles
    dipoles = [obj for obj in obj_list if obj._object_type == "Dipole"]

    # currents
    circulars = [obj for obj in obj_list if obj._object_type == "Circular"]
    lines = [obj for obj in obj_list if obj._object_type == "Line"]

    # create or set plotting axis
    if axis is None:
        fig = plt.figure(dpi=80, figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_box_aspect((1, 1, 1))
        generate_output = True
    else:
        ax = axis
        generate_output = False

    # load color map
    cmap = plt.cm.get_cmap("hsv")

    # draw objects and evaluate system size --------------------------------------

    # draw faced objects and store vertices
    face_points = []
    for i, obj in enumerate(faced_objects):
        col = cmap(i / len(faced_objects))

        if obj._object_type == "Cuboid":
            faces = faces_cuboid(obj, show_path)
            lw = 0.5
            face_points += draw_faces(faces, col, lw, ax)

        elif obj._object_type == "Cylinder":
            faces = faces_cylinder(obj, show_path)
            lw = 0.25
            face_points += draw_faces(faces, col, lw, ax)

        elif obj._object_type == "CylinderSegment":
            faces = faces_cylinder_section(obj, show_path)
            lw = 0.25
            face_points += draw_faces(faces, col, lw, ax)

        elif obj._object_type == "Sphere":
            faces = faces_sphere(obj, show_path)
            lw = 0.25
            face_points += draw_faces(faces, col, lw, ax)

    # draw sensor pixel and get positions
    sensor_points = draw_pixel(sensors, ax, show_path)

    # get dipole positions
    dipole_points = [dip.position for dip in dipoles]

    # draw circulars and get line positions
    current_points = draw_circular(circulars, show_path, ax)
    current_points += draw_line(lines, show_path, ax)

    # draw paths and get path points
    path_points = []
    if show_path:  # True or int>0
        for i, obj in enumerate(faced_objects):
            col = cmap(i / len(faced_objects))
            path_points += draw_path(obj, col, ax)

        for sens in sensors:
            path_points += draw_path(sens, ".6", ax)

        for dip in dipoles:
            path_points += draw_path(dip, ".6", ax)

        for circ in circulars:
            path_points += draw_path(circ, ".6", ax)

        for line in lines:
            path_points += draw_path(line, ".6", ax)

    # markers -------------------------------------------------------
    if markers is not None and markers:
        markers = np.array(markers)
        draw_markers(markers, ax)
    else:
        markers = []

    # draw direction arrows (based on src size) -------------------------
    if show_direction:
        draw_directs_faced(faced_objects, cmap, ax, show_path, size_direction)

    # determine system size -----------------------------------------
    limx1, limx0, limy1, limy0, limz1, limz0 = system_size(
        face_points, sensor_points, dipole_points, markers, path_points, current_points
    )
    # make sure ranges are not null
    ranges = np.array([[limx0, limx1], [limy0, limy1], [limz0, limz1]])
    ranges[np.squeeze(np.diff(ranges)) == 0] += np.array([-1, 1])
    sys_size = np.max(np.diff(ranges))

    # draw all system sized based quantities -------------------------
    draw_sensors(sensors, ax, sys_size, show_path, size_sensors)
    draw_dipoles(dipoles, ax, sys_size, show_path, size_dipoles)

    # plot styling --------------------------------------------------
    ax.set(
        **{f"{k}label": f"{k} [mm]" for k in "xyz"},
        **{
            f"{k}lim": (r[0] - abs(r[0]) * zoom, r[1] + abs(r[1]) * zoom)
            for k, r in zip("xyz", ranges)
        },
    )

    # generate output ------------------------------------------------
    if generate_output:
        plt.show()

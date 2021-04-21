""" Display function codes"""

import numpy as np
import matplotlib.pyplot as plt
from magpylib._lib.utility import format_obj_input, test_path_format
from magpylib._lib.display.mpl_draw import (draw_directs_faced, draw_faces, draw_markers, draw_path,
    draw_pixel, draw_sensors, draw_dipoles)
from magpylib._lib.display.disp_utility import (faces_box, faces_cylinder, system_size,
    faces_sphere)
from magpylib import _lib


# ON INTERFACE
def display(
        *objects,
        markers=[(0,0,0)],
        axis=None,
        direc=False,
        show_path=True,
        size_sensors=1,
        size_direc=1,
        size_dipoles=1):
    """
    Display objects and paths graphically using matplotlib 3D.

    Parameters
    ----------
    objects: sources, collections or sensors
        Show a 3D reprensation of given objects in matplotlib.

    markers: array_like, shape (N,3), default=[(0,0,0)]
        Display position markers in the global CS. By default a marker is in the origin.

    axis: pyplot.axis, default=None
        Display graphical output in a given pyplot axis (must be 3D). By default a new
        pyplot figure is created and displayed.

    direc: bool, default=False
        Set True to show magnetization and current directions.

    show_path: bool or int, default=True
        Options True, False, positive int. By default object paths are shown. If show_path is
        a positive integer, objects will be displayed at each path position in steps of show_path.

    size_sensor: float, default=1
        Adjust automatic display size of sensors.

    size_direc: float, default=1
        Adjust automatic display size of direction arrows

    Returns
    -------
    None
    """
    # pylint: disable=protected-access
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    # pylint: disable=dangerous-default-value

    # avoid circular imports
    Box = _lib.obj_classes.Box
    Cylinder = _lib.obj_classes.Cylinder
    Sensor = _lib.obj_classes.Sensor
    Sphere = _lib.obj_classes.Sphere
    Dipole = _lib.obj_classes.Dipole

    # create or set plotting axis
    if axis is None:
        fig = plt.figure(dpi=80, figsize=(8,8))
        ax = fig.gca(projection='3d')
        ax.set_box_aspect((1, 1, 1))
        generate_output = True
    else:
        ax = axis
        generate_output = False

    # load color map
    cmap = plt.cm.get_cmap('hsv')

    # flatten input
    obj_list = format_obj_input(objects)

    # test if every individual obj_path is good
    test_path_format(obj_list)

    # sort input objects --------------------------------------------------------

    # objects with faces
    faced_objects = [obj for obj in obj_list if isinstance(obj, (
        Box,
        Cylinder,
        Sphere
        ))]

    # sensors
    sensors = [obj for obj in obj_list if isinstance(obj, Sensor)]

    # dipoles
    dipoles = [obj for obj in obj_list if isinstance(obj, Dipole)]

    # draw objects and evaluate system size --------------------------------------

    # draw faced objects and store vertices
    face_points = []
    for i, obj in enumerate(faced_objects):
        col = cmap(i/len(faced_objects))

        if isinstance(obj, Box):
            faces = faces_box(obj,show_path)
            lw = 0.5
            face_points += draw_faces(faces, col, lw, ax)

        elif isinstance(obj, Cylinder):
            faces = faces_cylinder(obj,show_path)
            lw = 0.25
            face_points += draw_faces(faces, col, lw, ax)

        elif isinstance(obj, Sphere):
            faces = faces_sphere(obj,show_path)
            lw = 0.25
            face_points += draw_faces(faces, col, lw, ax)

    # draw sensor pixel
    sensor_points = draw_pixel(sensors, ax, show_path)

    # get dipole positions
    dipole_points = [dip.pos for dip in dipoles]

    # draw paths and get path points
    path_points = []
    if show_path:  # True or int>0
        for i, obj in enumerate(faced_objects):
            col = cmap(i/len(faced_objects))
            path_points += draw_path(obj, col, ax)

        for sens in sensors:
            path_points += draw_path(sens, '.6', ax)

        for dip in dipoles:
            path_points += draw_path(dip, '.6', ax)

    # markers -------------------------------------------------------
    if markers:
        markers = np.array(markers)
        draw_markers(markers, ax)

    # draw direc arrows (based on src size) -------------------------
    if direc:
        draw_directs_faced(faced_objects, cmap, ax, show_path, size_direc)

    # determine system size -----------------------------------------
    limx1, limx0, limy1, limy0, limz1, limz0 = system_size(
        face_points, sensor_points, dipole_points, markers, path_points)

    sys_size = max([limx1-limx0, limy1-limy0, limz1-limz0])

    # draw all system sized based quantities -------------------------
    draw_sensors(sensors, ax, sys_size, show_path, size_sensors)
    draw_dipoles(dipoles, ax, sys_size, show_path, size_dipoles)

    # plot styling --------------------------------------------------
    ax.set(
        xlabel = 'x [mm]',
        ylabel = 'y [mm]',
        zlabel = 'z [mm]',
        xlim=(limx0, limx1),
        ylim=(limy0, limy1),
        zlim=(limz0, limz1)
        )

    # generate output ------------------------------------------------
    if generate_output:
        plt.show()

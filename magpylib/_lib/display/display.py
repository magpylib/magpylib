""" Display function codes"""

import numpy as np
import matplotlib.pyplot as plt
from magpylib._lib.utility import format_obj_input, test_path_format
from magpylib._lib.display.mpl_draw import (draw_directs_faced, draw_faces, draw_markers, draw_path,
    draw_pixel, draw_sensors, draw_dipoles, draw_circular, draw_line)
from magpylib._lib.display.disp_utility import (faces_cuboid, faces_cylinder, system_size,
    faces_sphere)
from magpylib import _lib


# ON INTERFACE
def display(
        *objects,
        markers=[(0,0,0)],
        axis=None,
        show_direction=False,
        show_path=True,
        size_sensors=1,
        size_direction=1,
        size_dipoles=1,
        zoom=0.5):
    """
    Display objects and paths graphically using matplotlib 3D plotting.

    Parameters
    ----------
    objects: sources, collections or sensors
        Objects to be displayed.

    markers: array_like, shape (N,3), default=[(0,0,0)]
        Display position markers in the global CS. By default a marker is placed
        in the origin.

    axis: pyplot.axis, default=None
        Display graphical output in a given pyplot axis (must be 3D). By default a new
        pyplot figure is created and displayed.

    show_direction: bool, default=False
        Set True to show magnetization and current directions.

    show_path: bool or int, default=True
        Options True, False, positive int. By default object paths are shown. If
        show_path is a positive integer, objects will be displayed at multiple path
        positions along the path, in steps of show_path.

    size_sensor: float, default=1
        Adjust automatic display size of sensors.

    size_direction: float, default=1
        Adjust automatic display size of direction arrows.

    size_dipoles: float, default=1
        Adjust automatic display size of dipoles.

    Returns
    -------
    None: NoneType

    Examples
    --------

    Display multiple objects, object paths, markers in 3D using Matplotlib:

    >>> import magpylib as mag3
    >>> col = mag3.Collection(
        [mag3.magnet.Sphere(magnetization=(0,0,1), diameter=1) for _ in range(3)])
    >>> for displ,src in zip([(.1414,0,0),(-.1,-.1,0),(-.1,.1,0)], col):
    >>>     src.move([displ]*50, increment=True)
    >>>     src.rotate_from_angax(angle=[10]*50, axis='z', anchor=0, start=0, increment=True)
    >>> ts = [-.6,-.4,-.2,0,.2,.4,.6]
    >>> sens = mag3.Sensor(position=(0,0,2), pixel=[(x,y,0) for x in ts for y in ts])
    >>> mag3.display(col, sens)
    --> graphic output

    Display figure on your own 3D Matplotlib axis:

    >>> import matplotlib.pyplot as plt
    >>> import magpylib as mag3
    >>> my_axis = plt.axes(projection='3d')
    >>> magnet = mag3.magnet.Cuboid(magnetization=(0,0,1), dimension=(1,2,3))
    >>> sens = mag3.Sensor(position=(0,0,3))
    >>> mag3.display(magnet, sens, axis=my_axis)
    >>> plt.show()
    --> graphic output

    """
    # pylint: disable=protected-access
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    # pylint: disable=dangerous-default-value

    # avoid circular imports
    Cuboid = _lib.obj_classes.Cuboid
    Cylinder = _lib.obj_classes.Cylinder
    Sensor = _lib.obj_classes.Sensor
    Sphere = _lib.obj_classes.Sphere
    Dipole = _lib.obj_classes.Dipole
    Circular = _lib.obj_classes.Circular
    Line = _lib.obj_classes.Line

    # create or set plotting axis
    if axis is None:
        fig = plt.figure(dpi=80, figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
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
        Cuboid,
        Cylinder,
        Sphere
        ))]

    # sensors
    sensors = [obj for obj in obj_list if isinstance(obj, Sensor)]

    # dipoles
    dipoles = [obj for obj in obj_list if isinstance(obj, Dipole)]

    # currents
    circulars = [obj for obj in obj_list if isinstance(obj, Circular)]
    lines = [obj for obj in obj_list if isinstance(obj, Line)]

    # draw objects and evaluate system size --------------------------------------

    # draw faced objects and store vertices
    face_points = []
    for i, obj in enumerate(faced_objects):
        col = cmap(i/len(faced_objects))

        if isinstance(obj, Cuboid):
            faces = faces_cuboid(obj,show_path)
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
            col = cmap(i/len(faced_objects))
            path_points += draw_path(obj, col, ax)

        for sens in sensors:
            path_points += draw_path(sens, '.6', ax)

        for dip in dipoles:
            path_points += draw_path(dip, '.6', ax)

        for circ in circulars:
            path_points += draw_path(circ, '.6', ax)

        for line in lines:
            path_points += draw_path(line, '.6', ax)


    # markers -------------------------------------------------------
    if markers:
        markers = np.array(markers)
        draw_markers(markers, ax)

    # draw direction arrows (based on src size) -------------------------
    if show_direction:
        draw_directs_faced(faced_objects, cmap, ax, show_path, size_direction)

    # determine system size -----------------------------------------
    limx1, limx0, limy1, limy0, limz1, limz0 = system_size(
        face_points, sensor_points, dipole_points, markers, path_points, current_points)

    sys_size = max([limx1-limx0, limy1-limy0, limz1-limz0])

    # draw all system sized based quantities -------------------------
    draw_sensors(sensors, ax, sys_size, show_path, size_sensors)
    draw_dipoles(dipoles, ax, sys_size, show_path, size_dipoles)

    # plot styling --------------------------------------------------
    ax.set(
        xlabel = 'x [mm]',
        ylabel = 'y [mm]',
        zlabel = 'z [mm]',
        xlim=(limx0-abs(limx0)*zoom, limx1+abs(limx1)*zoom),
        ylim=(limy0-abs(limy0)*zoom, limy1+abs(limy1)*zoom),
        zlim=(limz0-abs(limz0)*zoom, limz1+abs(limz1)*zoom)
        )

    # generate output ------------------------------------------------
    if generate_output:
        plt.show()

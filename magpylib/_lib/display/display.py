""" Display function codes"""

import numpy as np
import matplotlib.pyplot as plt
import magpylib as mag3
from magpylib._lib.math_utility import format_obj_input, test_path_format
from magpylib._lib.display.mpl_draw import (draw_directs, draw_faces, draw_markers, draw_path,
    draw_sensors)
from magpylib._lib.display.disp_utility import faces_box, faces_cylinder, system_size

# pylint: disable=useless-return

def display(
        *objects,
        markers=[(0,0,0)],
        axis=None,
        direc=False,
        show_path=True):
    """ Display objects and paths graphically using matplotlib.

    Parameters
    ----------
    objects: sources, collections or sensors
        Show a 3D reprensation of given objects in matplotlib.

    markers: array_like, shape (N,3), default=[(0,0,0)]
        Mark positions in graphic output. Default value puts a marker
        in the origin.

    axis: pyplot.axis, default=None
        Display graphical output in a given pyplot axis (must be 3D).

    direc: bool, default=False
        Set True to plot magnetization and current directions

    show_path: bool/string, default=True
        Set True to plot object paths. Set to 'all' to plot an object
        represenation at each path position.

    Returns
    -------
    no return
    """
    # pylint: disable=protected-access
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    # pylint: disable=dangerous-default-value

    # create or set plotting axis
    if axis is None:
        fig = plt.figure(dpi=80, figsize=(8,8))
        ax = fig.gca(projection='3d')
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

    # draw objects --------------------------------------------------
    faced_objects = [obj for obj in obj_list if isinstance(obj, (
        mag3.magnet.Box,
        mag3.magnet.Cylinder
        ))]

    for i, obj in enumerate(faced_objects):
        col = cmap(i/len(faced_objects))

        if isinstance(obj, mag3.magnet.Box):
            faces = faces_box(obj)
            lw = 0.5
            draw_faces(faces, col, lw, ax)

        elif isinstance(obj, mag3.magnet.Cylinder):
            faces = faces_cylinder(obj)
            lw = 0.25
            draw_faces(faces, col, lw, ax)

    sensors = [obj for obj in obj_list if isinstance(obj, mag3.Sensor)]
    draw_sensors(sensors, ax)

    # path ------------------------------------------------------
    if show_path is True:
        for i, obj in enumerate(faced_objects):
            col = cmap(i/len(faced_objects))
            draw_path(obj, col, ax)

        for sens in sensors:
            draw_path(sens, '.6', ax)

    # markers -------------------------------------------------------
    markers = np.array(markers)
    draw_markers(markers, ax)

    # directs -------------------------------------------------------
    if direc:
        draw_directs(faced_objects, cmap, ax)

    # determine system size
    sys_size = system_size(faced_objects, sensors, markers)

    # plot styling --------------------------------------------------
    ax.set(
        xlabel = 'x [mm]',
        ylabel = 'y [mm]',
        zlabel = 'z [mm]',
        xlim=(-sys_size, sys_size),
        ylim=(-sys_size, sys_size),
        zlim=(-sys_size, sys_size)
        )

    # generate output ------------------------------------------------
    if generate_output:
        plt.show()

    return None

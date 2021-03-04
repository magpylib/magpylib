""" Display function codes"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import magpylib3 as mag3
from magpylib3._lib.math_utility import format_src_input, good_path_format
from magpylib3._lib.display.mpl_draw import (draw_directs, draw_faces, draw_markers, draw_path,
    draw_sensors)
from magpylib3._lib.display.disp_utility import faces_box, faces_cylinder, system_size


def display(
        *objects,
        markers=[(0,0,0)],
        axis=None,
        direc=False,
        show_path=True):
    """ Display sources/sensors graphically

    Parameters
    ----------
    objects: sources, collections or sensors
        Display a 3D reprensation of given objects using matplotlib

    markers: array_like, shape (N,3), default=[(0,0,0)]
        Mark positions in graphic output. Puts a marker in the origin.
        by default.

    axis: pyplot.axis, default=None
        Display graphical output in a given pyplot axis (must be 3D).

    direc: bool, default=False
        Set True to plot magnetization and current directions

    show_path: bool/string, default=False
        Set True to plot object paths. Set 'all' to plot an object
        represenation at each path position.

    Returns
    -------
    None
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
    obj_list = format_src_input(objects)

    # test if every individual obj_path is good
    for obj in obj_list:
        if not good_path_format([obj]):
            sys.exit('ERROR: display() - bad path format (different pos/rot length)')


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

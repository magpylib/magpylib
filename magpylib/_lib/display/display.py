""" Display function codes"""

import warnings
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from magpylib._lib.utility import format_obj_input, test_path_format
from magpylib._lib.display.style import get_style
from magpylib._lib.display.disp_utility import Markers

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
    path=True,  # bool, int, index list, 'animate'
    zoom=1,
    animate_time=5,
    markers=None,
    backend=None,
    canvas=None,
    **kwargs,
):
    """
    Display objects and paths graphically.

    Parameters
    ----------
    objects: sources, collections or sensors
        Objects to be displayed.

    path: bool or int or array_like, default=True
        Options True, False, positive int or iterable. By default object paths are shown. If
        path is a positive integer, objects will be displayed at multiple path
        positions along the path, in steps of path. If path is an iterable
        of integers, objects will be displayed for the provided indices.
        If path='animate, the plot will be animated according to the `animate_time`
        and 'animate_fps' parameters.

    zoom: float, default = 1
        Adjust plot zoom-level. When zoom=0 all objects are just inside the 3D-axes.

    animate_time: float, default = 3
        Sets the animation duration

    markers: array_like, None, shape (N,3), default=None
        Display position markers in the global CS. By default no marker is displayed.

    backend: default=None
        One of 'matplotlib', 'plolty'. If not set, parameter will default to
        Config.BACKEND

    canvas: pyplot Axis or plotly Figure, default=None
        Display graphical output in a given canvas:
        - with matplotlib: pyplot axis (must be 3D).
        - with plotly: plotly.graph_objects.Figure
            or plotly.graph_objects.FigureWidget
        By default a new canvas is created and displayed.

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

    # test if every individual obj_path is good
    test_path_format(obj_list)
    check_show_path = (
        isinstance(path, (int, bool))
        or path == "animate"
        or (hasattr(path, "__iter__") and not isinstance(path, str))
    )
    assert check_show_path, (
        f"`path` argument of type {type(path)} is invalid, \n"
        "it must be one of (True, False, 'animate'), a positive path index "
        "or an Iterable of path indices."
    )

    if backend is None:
        backend = Config.BACKEND

    if backend == "matplotlib":
        if path == "animate":
            warnings.warn(
                "The matplotlib backend does not support animation, falling back to path=True"
            )
            path = True
        display_matplotlib(
            *obj_list,
            markers=markers,
            show_path=path,
            zoom=zoom,
            axis=canvas,
            **kwargs,
        )
    elif backend == "plotly":
        # pylint: disable=import-outside-toplevel
        from magpylib._lib.display.plotly_draw import display_plotly

        display_plotly(
            *obj_list,
            markers=markers,
            show_path=path,
            zoom=zoom,
            fig=canvas,
            animate_time=animate_time,
            **kwargs,
        )


def display_matplotlib(
    *obj_list,
    axis=None,
    markers=None,
    show_path=True,
    zoom=1,
    color_sequence=None,
    **kwargs,
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

    markers: array_like, shape (N,3), default=None
        Display position markers in the global CS. By default no marker is displayed.

    show_path: bool or int or array_like, default=True
        Options True, False, positive int or iterable. By default object paths are shown. If
        show_path is a positive integer, objects will be displayed at multiple path
        positions along the path, in steps of show_path. If show_path is an iterable
        of integers, objects will be displayed for the provided indices.

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

    # apply config default values if None
    if color_sequence is None:
        color_sequence = Config.COLOR_SEQUENCE

    # create or set plotting axis
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
    dipoles_color = []
    sensors_color = []
    faced_objects_color = []

    for obj, color in zip(obj_list, cycle(color_sequence)):
        style = get_style(obj, **kwargs)
        color = style.color if style.color is not None else color
        lw = 0.25
        faces = None
        if obj._object_type == "Cuboid":
            lw = 0.5
            faces = faces_cuboid(obj, show_path)
        elif obj._object_type == "Cylinder":
            faces = faces_cylinder(obj, show_path)
        elif obj._object_type == "CylinderSegment":
            faces = faces_cylinder_section(obj, show_path)
        elif obj._object_type == "Sphere":
            faces = faces_sphere(obj, show_path)
        elif obj._object_type == "Line":
            size = style.current.size if style.current.show else 0
            points += draw_line([obj], show_path, color, size, ax)
        elif obj._object_type == "Circular":
            size = style.current.size if style.current.show else 0
            points += draw_circular([obj], show_path, color, size, ax)
        elif obj._object_type == "Sensor":
            sensors_color += [color]
            points += draw_pixel(
                [obj], ax, color, style.pixel.color, style.pixel.size * 4, show_path
            )
        elif obj._object_type == "Dipole":
            dipoles_color += [color]
            points += [obj.position]
        if faces is not None:
            faced_objects_color += [color]
            pts = draw_faces(faces, color, lw, ax)
            points += [np.vstack(pts).reshape(-1, 3)]
            if style.magnetization.show:
                check_excitations([obj])
                draw_directs_faced(
                    [obj], [color], ax, show_path, style.magnetization.size
                )
        if show_path:
            marker,line = style.path.marker, style.path.line
            points += draw_path(obj, color, marker.symbol, marker.size, marker.color, line.style, line.width, ax)

    # markers -------------------------------------------------------
    if markers is not None and markers:
        m = Markers()
        style = get_style(m, **kwargs)
        markers = np.array(markers)
        draw_markers(markers, ax, style=style.marker)
        points += [markers]

    # draw direction arrows (based on src size) -------------------------
    # objects with faces

    # determine system size -----------------------------------------
    limx1, limx0, limy1, limy0, limz1, limz0 = system_size(points)

    # make sure ranges are not null
    ranges = np.array([[limx0, limx1], [limy0, limy1], [limz0, limz1]])
    ranges[np.squeeze(np.diff(ranges)) == 0] += np.array([-1, 1])
    sys_size = np.max(np.diff(ranges))

    # draw all system sized based quantities -------------------------
    # sensors
    sensors = [obj for obj in obj_list if obj._object_type == "Sensor"]

    # dipoles
    dipoles = [obj for obj in obj_list if obj._object_type == "Dipole"]

    # not optimal for loop if many sensors/dipoles
    for sensor in sensors:
        style = get_style(sensor, **kwargs)
        draw_sensors([sensor], ax, sys_size, show_path, style.size)
    for dipole, color in zip(dipoles, dipoles_color):
        style = get_style(dipole, **kwargs)
        draw_dipoles([dipole], ax, sys_size, show_path, style.size, color)

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

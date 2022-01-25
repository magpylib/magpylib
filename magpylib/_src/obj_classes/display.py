""" Display function codes"""

import warnings
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from magpylib._src.utility import format_obj_input, test_path_format
from magpylib._src.style import get_style
from magpylib._src.display.display_utility import MagpyMarkers

from magpylib._src.display.display_matplotlib import (
    draw_directs_faced,
    draw_faces,
    draw_markers,
    draw_path,
    draw_pixel,
    draw_sensors,
    draw_dipoles,
    draw_circular,
    draw_line,
    draw_model3d_extra,
)
from magpylib._src.display.display_utility import (
    faces_cuboid,
    faces_cylinder,
    system_size,
    faces_sphere,
    faces_cylinder_section,
)
from magpylib._src.input_checks import check_excitations, check_dimensions
from magpylib._src.defaults.defaults_classes import default_settings as Config


# ON INTERFACE
def display(
    *objects,
    path=True,  # bool, int, index list, 'animate'
    zoom=0,
    animate_time=3,
    markers=None,
    backend=None,
    canvas=None,
    **kwargs,
):
    """
    Display objects and paths graphically. Style input can be a style-dict or
    style-underscore_magic.

    Parameters
    ----------
    objects: sources, collections or sensors
        Objects to be displayed.

    path: bool or int i or array_like shape (n,) or `'animate'`, default = True
        Option False shows objects at final path position and hides paths.
        Option True shows objects at final path position and shows object paths.
        Option int i displays the objects at every i'th path position.
        Option array_like shape (n,) discribes certain path indices. The objects
        displays are displayed at every given path index.
        Option `'animate'` (Plotly backend only) shows an animation of objectes moving
        along their paths.

    zoom: float, default = 0
        Adjust plot zoom-level. When zoom=0 3D-figure boundaries are tight.

    animate_time: float, default = 3
        Sets the animation duration.

    markers: array_like, shape (N,3), default=None
        Display position markers in the global CS. By default no marker is displayed.

    backend: string, default=None
        One of `'matplotlib'` or `'plotly'`. If not set, parameter will default to
        magpylib.defaults.display.backend which comes as 'matplotlib' with installation.

    canvas: pyplot Axis or plotly Figure, default=None
        Display graphical output in a given canvas:
        - with matplotlib: pyplot axis (must be 3D).
        - with plotly: plotly.graph_objects.Figure or plotly.graph_objects.FigureWidget
        By default a new canvas is created and displayed.

    Returns
    -------
    None: NoneType

    Examples
    --------

    Display multiple objects, object paths, markers in 3D using Matplotlib or Plotly:

    >>> import magpylib as magpy
    >>> src = magpy.magnet.Sphere(magnetization=(0,0,1), diameter=1)
    >>> src.move([(.1,0,0)]*50, increment=True)
    >>> src.rotate_from_angax(angle=[10]*50, axis='z', anchor=0, start=0, increment=True)
    >>> ts = [-.4,0,.4]
    >>> sens = magpy.Sensor(position=(0,0,2), pixel=[(x,y,0) for x in ts for y in ts])
    >>> magpy.display(src, sens)
    >>> magpy.display(src, sens, backend='plotly')
    --> graphic output

    Display figure on your own canvas (here Matplotlib 3D axis):

    >>> import matplotlib.pyplot as plt
    >>> import magpylib as magpy
    >>> my_axis = plt.axes(projection='3d')
    >>> magnet = magpy.magnet.Cuboid(magnetization=(1,1,1), dimension=(1,2,3))
    >>> sens = magpy.Sensor(position=(0,0,3))
    >>> magpy.display(magnet, sens, canvas=my_axis, zoom=1)
    >>> plt.show()
    --> graphic output

    Use sophisticated figure styling options accessible from defaults, as individual object styles
    or as global style arguments in display.

    >>> import magpylib as magpy
    >>> src1 = magpy.magnet.Sphere((1,1,1), 1)
    >>> src2 = magpy.magnet.Sphere((1,1,1), 1, (1,0,0))
    >>> magpy.defaults.display.style.magnet.magnetization.size = 2
    >>> src1.style.magnetization.size = 1
    >>> magpy.display(src1, src2, style_color='r', zoom=3)
    --> graphic output
    """

    # flatten input
    obj_list_flat = format_obj_input(objects, allow='sources+sensors')
    obj_list_semi_flat = format_obj_input(objects, allow='sources+sensors+collections')

    # test if all source dimensions and excitations (if sho_direc=True) have been initialized
    check_dimensions(obj_list_flat)

    # test if every individual obj_path is good
    test_path_format(obj_list_flat)
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
        backend = Config.display.backend

    if backend == "matplotlib":
        if path == "animate":
            warnings.warn(
                "The matplotlib backend does not support animation, falling back to path=True"
            )
            path = True
        display_matplotlib(
            *obj_list_semi_flat,
            markers=markers,
            path=path,
            zoom=zoom,
            axis=canvas,
            **kwargs,
        )
    elif backend == "plotly":
        # pylint: disable=import-outside-toplevel
        from magpylib._src.display.plotly.plotly_display import display_plotly

        display_plotly(
            *obj_list_semi_flat,
            markers=markers,
            show_path=path,
            zoom=zoom,
            fig=canvas,
            animate_time=animate_time,
            **kwargs,
        )


def display_matplotlib(
    *obj_list_semi_flat,
    axis=None,
    markers=None,
    path=True,
    zoom=0,
    color_sequence=None,
    **kwargs,
):
    """
    Display objects and paths graphically with the matplotlib backend.

    - axis: matplotlib axis3d object
    - markers: list of marker positions
    - path: bool / int / list of ints
    - zoom: zoom level, 0=tight boundaries
    - color_sequence: list of colors for object coloring
    """
    # pylint: disable=protected-access
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements

    # apply config default values if None
    if color_sequence is None:
        color_sequence = Config.display.colorsequence
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

    for semi_flat_obj, color in zip(obj_list_semi_flat, cycle(color_sequence)):
        flat_objs = [semi_flat_obj]
        if getattr(semi_flat_obj, "children", None) is not None:
            flat_objs = semi_flat_obj.children
            if getattr(semi_flat_obj, "position", None) is not None:
                flat_objs += [semi_flat_obj]
                color = color if semi_flat_obj.style.color is None else semi_flat_obj.style.color
        for obj in flat_objs:
            style = get_style(obj, Config, **kwargs)
            color = style.color if style.color is not None else color
            lw = 0.25
            faces = None
            if obj.style.model3d.extra:
                draw_model3d_extra(obj, style, path, ax, color)
            if obj.style.model3d.show:
                if obj._object_type == "Cuboid":
                    lw = 0.5
                    faces = faces_cuboid(obj, path)
                elif obj._object_type == "Cylinder":
                    faces = faces_cylinder(obj, path)
                elif obj._object_type == "CylinderSegment":
                    faces = faces_cylinder_section(obj, path)
                elif obj._object_type == "Sphere":
                    faces = faces_sphere(obj, path)
                elif obj._object_type == "Line":
                    if style.arrow.show:
                        check_excitations([obj])
                    arrow_size = style.arrow.size if style.arrow.show else 0
                    arrow_width = style.arrow.width
                    points += draw_line(
                        [obj], path, color, arrow_size, arrow_width, ax
                    )
                elif obj._object_type == "Loop":
                    if style.arrow.show:
                        check_excitations([obj])
                    arrow_width = style.arrow.width
                    arrow_size = style.arrow.size if style.arrow.show else 0
                    points += draw_circular(
                        [obj], path, color, arrow_size, arrow_width, ax
                    )
                elif obj._object_type == "Sensor":
                    sensors_color += [color]
                    points += draw_pixel(
                        [obj],
                        ax,
                        color,
                        style.pixel.color,
                        style.pixel.size,
                        style.pixel.symbol,
                        path,
                    )
                elif obj._object_type == "Dipole":
                    dipoles_color += [color]
                    points += [obj.position]
                elif obj._object_type == "CustomSource":
                    draw_markers(np.array([obj.position]), ax, color, symbol="*", size=10)
                    name = (
                        obj.style.name
                        if obj.style.name is not None
                        else str(type(obj).__name__)
                    )
                    ax.text(*obj.position, name, horizontalalignment="center")
                    points += [obj.position]
                if faces is not None:
                    faced_objects_color += [color]
                    alpha = style.opacity
                    pts = draw_faces(faces, color, lw, alpha, ax)
                    points += [np.vstack(pts).reshape(-1, 3)]
                    if style.magnetization.show:
                        check_excitations([obj])
                        pts = draw_directs_faced(
                            [obj], [color], ax, path, style.magnetization.size
                        )
                        points += pts
            if path:
                marker, line = style.path.marker, style.path.line
                if style.path.show:
                    points += draw_path(
                        obj,
                        color,
                        marker.symbol,
                        marker.size,
                        marker.color,
                        line.style,
                        line.width,
                        ax,
                    )

    # markers -------------------------------------------------------
    if markers is not None and markers:
        m = MagpyMarkers()
        style = get_style(m, Config, **kwargs)
        markers = np.array(markers)
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
    # sensors
    sensors = [obj for obj in obj_list_semi_flat if obj._object_type == "Sensor"]

    # dipoles
    dipoles = [obj for obj in obj_list_semi_flat if obj._object_type == "Dipole"]

    # not optimal for loop if many sensors/dipoles
    for sensor in sensors:
        style = get_style(sensor, Config, **kwargs)
        draw_sensors([sensor], ax, sys_size, path, style.size)
    for dipole, color in zip(dipoles, dipoles_color):
        style = get_style(dipole, Config, **kwargs)
        draw_dipoles([dipole], ax, sys_size, path, style.size, color, style.pivot)

    # plot styling --------------------------------------------------
    ax.set(
        **{f"{k}label": f"{k} [mm]" for k in "xyz"},
        **{f"{k}lim": r for k, r in zip("xyz", ranges)},
    )

    # generate output ------------------------------------------------
    if generate_output:
        plt.show()

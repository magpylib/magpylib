""" Display function codes"""

import warnings
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from magpylib._lib.utility import format_obj_input, test_path_format
from magpylib._lib.style import get_style
from magpylib._lib.display.disp_utility import MagpyMarkers

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
from magpylib._lib.config import default_settings as Config


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
    Display objects and paths graphically.

    Parameters
    ----------
    objects: sources, collections or sensors
        Objects to be displayed.

    path: bool or int i or array_like shape (n,) or 'animate', default = True
        Option False shows objects at final path position and hides paths.
        Option True shows objects at final path position and shows object paths.
        Option int i displays the objects at every i'th path position.
        Option array_like shape (n,) discribes certain path indices. The objects
        displays are displayed at every given path index.
        Option 'animate' (Plotly backend only) shows an animation of objectes moving
        along their paths.

    zoom: float, default = 0
        Adjust plot zoom-level. When zoom=0 3D-figure boundaries are tight.

    animate_time: float, default = 3
        Sets the animation duration.

    markers: array_like, shape (N,3), default=None
        Display position markers in the global CS. By default no marker is displayed.

    backend: string, default=None
        One of 'matplotlib' or 'plotly'. If not set, parameter will default to
        magpylib.defaults.display.backend which comes as 'matplotlib' with installation.

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

    Use sophisticated figure styling options accessable from defaults, as individual object styles
    or as global style arguments in display.

    >>> import magpylib as magpy
    >>> src1 = magpy.magnet.Sphere((1,1,1), 1)
    >>> src2 = magpy.magnet.Sphere((1,1,1), 1, (1,0,0))
    >>> magpy.defaults.display.styles.magnets.magnetization.size = 2
    >>> src1.style.magnetization.size = 1
    >>> magpy.display(src1, src2, style_color='r', zoom=3)
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
        backend = Config.display.backend

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
    zoom=0,
    color_sequence=None,
    **kwargs,
):
    """
    Display objects and paths graphically with the matplotlib backend.

    - axis: matplotlib axis3d object
    - markers: list of marker positions
    - show_path: bool / int / list of ints
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

    for obj, color in zip(obj_list, cycle(color_sequence)):
        style = get_style(obj, Config, **kwargs)
        color = style.color if style.color is not None else color
        lw = 0.25
        faces = None
        if style.mesh3d.data is not None and (
            style.mesh3d.show is True or style.mesh3d.show == "inplace"
        ):
            text = (
                f"{obj} has a mesh3d attached, which cannot be plotted with the matplotlib "
                "backend at the moment"
            )
            warnings.warn(text)
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
            if style.current.show:
                check_excitations([obj])
            size = style.current.size if style.current.show else 0
            width = style.current.width
            points += draw_line([obj], show_path, color, size, width, ax)
        elif obj._object_type == "Circular":
            if style.current.show:
                check_excitations([obj])
            width = style.current.width
            size = style.current.size if style.current.show else 0
            points += draw_circular([obj], show_path, color, size, width, ax)
        elif obj._object_type == "Sensor":
            sensors_color += [color]
            points += draw_pixel(
                [obj],
                ax,
                color,
                style.pixel.color,
                style.pixel.size,
                style.pixel.symbol,
                show_path,
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
            marker, line = style.path.marker, style.path.line
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
        draw_markers(markers, ax, style=style.marker)
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
    sensors = [obj for obj in obj_list if obj._object_type == "Sensor"]

    # dipoles
    dipoles = [obj for obj in obj_list if obj._object_type == "Dipole"]

    # not optimal for loop if many sensors/dipoles
    for sensor in sensors:
        style = get_style(sensor, Config, **kwargs)
        draw_sensors([sensor], ax, sys_size, show_path, style.size)
    for dipole, color in zip(dipoles, dipoles_color):
        style = get_style(dipole, Config, **kwargs)
        draw_dipoles([dipole], ax, sys_size, show_path, style.size, color, style.pivot)

    # plot styling --------------------------------------------------
    ax.set(
        **{f"{k}label": f"{k} [mm]" for k in "xyz"},
        **{f"{k}lim": r for k, r in zip("xyz", ranges)},
    )

    # generate output ------------------------------------------------
    if generate_output:
        plt.show()

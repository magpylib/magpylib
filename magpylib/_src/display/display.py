""" Display function codes"""

import warnings
from magpylib._src.utility import format_obj_input, test_path_format
from magpylib._src.display.display_matplotlib import display_matplotlib
from magpylib._src.input_checks import check_dimensions
from magpylib._src.defaults.defaults_classes import default_settings as Config


# ON INTERFACE
def show(
    *objects,
    zoom=0,
    animation=False,
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

    zoom: float, default = 0
        Adjust plot zoom-level. When zoom=0 3D-figure boundaries are tight.

    animation: bool, default=False
        If True and at least one object has a path, animated paths are rendered.
        If a positive float, the animation duration target is set to the given value.
        This feature is only available for the `plotly` backend at the moment.

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
    >>> magpy.show(src, sens)
    >>> magpy.show(src, sens, backend='plotly')
    --> graphic output

    Display figure on your own canvas (here Matplotlib 3D axis):

    >>> import matplotlib.pyplot as plt
    >>> import magpylib as magpy
    >>> my_axis = plt.axes(projection='3d')
    >>> magnet = magpy.magnet.Cuboid(magnetization=(1,1,1), dimension=(1,2,3))
    >>> sens = magpy.Sensor(position=(0,0,3))
    >>> magpy.show(magnet, sens, canvas=my_axis, zoom=1)
    >>> plt.show()
    --> graphic output

    Use sophisticated figure styling options accessible from defaults, as individual object styles
    or as global style arguments in display.

    >>> import magpylib as magpy
    >>> src1 = magpy.magnet.Sphere((1,1,1), 1)
    >>> src2 = magpy.magnet.Sphere((1,1,1), 1, (1,0,0))
    >>> magpy.defaults.display.style.magnet.magnetization.size = 2
    >>> src1.style.magnetization.size = 1
    >>> magpy.show(src1, src2, style_color='r', zoom=3)
    --> graphic output
    """

    # flatten input
    obj_list_flat = format_obj_input(objects, allow="sources+sensors")
    obj_list_semi_flat = format_obj_input(objects, allow="sources+sensors+collections")

    # test if all source dimensions and excitations have been initialized
    check_dimensions(obj_list_flat)

    # test if every individual obj_path is good
    test_path_format(obj_list_flat)

    if backend is None:
        backend = Config.display.backend

    if backend == "matplotlib":
        if animation is not False:
            msg = "The matplotlib backend does not support animation at the moment. "
            msg+= "Use plotly backend instead."
            warnings.warn(msg)
            # animation = False
        display_matplotlib(
            *obj_list_semi_flat,
            markers=markers,
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
            zoom=zoom,
            fig=canvas,
            animation=animation,
            **kwargs,
        )

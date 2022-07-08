""" Display function codes"""
import warnings

from magpylib._src.display.display_matplotlib import display_matplotlib
from magpylib._src.input_checks import check_dimensions
from magpylib._src.input_checks import check_excitations
from magpylib._src.input_checks import check_format_input_backend
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.input_checks import check_input_animation
from magpylib._src.input_checks import check_input_zoom
from magpylib._src.utility import format_obj_input
from magpylib._src.utility import test_path_format


def show(
    *objects,
    zoom=0,
    animation=False,
    markers=None,
    backend=None,
    canvas=None,
    **kwargs,
):
    """Display objects and paths graphically.

    Global graphic styles can be set with kwargs as style dictionary or using
    style underscore magic.

    Parameters
    ----------
    objects: Magpylib objects (sources, collections, sensors)
        Objects to be displayed.

    zoom: float, default=`0`
        Adjust plot zoom-level. When zoom=0 3D-figure boundaries are tight.

    animation: bool or float, default=`False`
        If `True` and at least one object has a path, the paths are rendered.
        If input is a positive float, the animation time is set to the given value.
        This feature is only available for the plotly backend.

    markers: array_like, shape (n,3), default=`None`
        Display position markers in the global coordinate system.

    backend: string, default=`None`
        Define plotting backend. Must be one of `'matplotlib'` or `'plotly'`. If not
        set, parameter will default to `magpylib.defaults.display.backend` which is
        `'matplotlib'` by installation default.

    canvas: matplotlib.pyplot `AxesSubplot` or plotly `Figure` object, default=`None`
        Display graphical output on a given canvas:
        - with matplotlib: `matplotlib.axes._subplots.AxesSubplot` with `projection=3d.
        - with plotly: `plotly.graph_objects.Figure` or `plotly.graph_objects.FigureWidget`.
        By default a new canvas is created and immediately displayed.

    Returns
    -------
    `None`: NoneType

    Examples
    --------

    Display multiple objects, object paths, markers in 3D using Matplotlib or Plotly:

    >>> import magpylib as magpy
    >>> src = magpy.magnet.Sphere(magnetization=(0,0,1), diameter=1)
    >>> src.move([(0.1*x,0,0) for x in range(50)])
    Sphere...
    >>> src.rotate_from_angax(angle=[*range(0,400,10)], axis='z', anchor=0, start=11)
    Sphere...
    >>> ts = [-.4,0,.4]
    >>> sens = magpy.Sensor(position=(0,0,2), pixel=[(x,y,0) for x in ts for y in ts])
    >>> magpy.show(src, sens) # doctest: +SKIP
    >>> magpy.show(src, sens, backend='plotly') # doctest: +SKIP
    >>> # graphic output

    Display output on your own canvas (here a Matplotlib 3d-axes):

    >>> import matplotlib.pyplot as plt
    >>> import magpylib as magpy
    >>> my_axis = plt.axes(projection='3d')
    >>> magnet = magpy.magnet.Cuboid(magnetization=(1,1,1), dimension=(1,2,3))
    >>> sens = magpy.Sensor(position=(0,0,3))
    >>> magpy.show(magnet, sens, canvas=my_axis, zoom=1)
    >>> plt.show() # doctest: +SKIP
    >>> # graphic output

    Use sophisticated figure styling options accessible from defaults, as individual object styles
    or as global style arguments in display.

    >>> import magpylib as magpy
    >>> src1 = magpy.magnet.Sphere((1,1,1), 1, [(0,0,0), (0,0,3)])
    >>> src2 = magpy.magnet.Sphere((1,1,1), 1, [(1,0,0), (1,0,3)], style_path_show=False)
    >>> magpy.defaults.display.style.magnet.magnetization.size = 2
    >>> src1.style.magnetization.size = 1
    >>> magpy.show(src1, src2, style_color='r') # doctest: +SKIP
    >>> # graphic output
    """

    # flatten input
    obj_list_flat = format_obj_input(objects, allow="sources+sensors")
    obj_list_semi_flat = format_obj_input(objects, allow="sources+sensors+collections")

    # test if all source dimensions and excitations have been initialized
    check_dimensions(obj_list_flat)
    check_excitations(obj_list_flat)

    # test if every individual obj_path is good
    test_path_format(obj_list_flat)

    # input checks
    backend = check_format_input_backend(backend)
    check_input_zoom(zoom)
    check_input_animation(animation)
    check_format_input_vector(
        markers,
        dims=(2,),
        shape_m1=3,
        sig_name="markers",
        sig_type="array_like of shape (n,3)",
        allow_None=True,
    )

    check_input_zoom(zoom)
    check_input_animation(animation)
    check_format_input_vector(
        markers,
        dims=(2,),
        shape_m1=3,
        sig_name="markers",
        sig_type="array_like of shape (n,3)",
        allow_None=True,
    )

    if backend == "matplotlib":
        if animation is not False:
            msg = "The matplotlib backend does not support animation at the moment.\n"
            msg += "Use `backend=plotly` instead."
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

""" Display function codes"""
from contextlib import contextmanager
from importlib import import_module

from magpylib._src.display.traces_generic import MagpyMarkers
from magpylib._src.display.traces_utility import process_show_input_objs
from magpylib._src.input_checks import check_dimensions
from magpylib._src.input_checks import check_excitations
from magpylib._src.input_checks import check_format_input_backend
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.input_checks import check_input_animation
from magpylib._src.input_checks import check_input_zoom
from magpylib._src.utility import test_path_format


class DisplayContext:
    """Display context class"""

    def __init__(self, isrunning=False):
        self.isrunning = isrunning
        self.objects = ()
        self.objects_from_ctx = ()
        self.kwargs = {}
        self.canvas = None

    def reset(self):
        """Reset display context"""
        self.isrunning = False
        self.objects = ()
        self.objects_from_ctx = ()
        self.kwargs = {}


ctx = DisplayContext()


def _show(
    *objects,
    zoom=0,
    animation=False,
    markers=None,
    backend=None,
    canvas=None,
    return_fig=False,
    row=None,
    col=None,
    output="model3d",
    sumup=True,
    **kwargs,
):
    """Display objects and paths graphically.

    See `show` function for docstring details.
    """

    # process input objs
    objects, obj_list_flat, max_rows, max_cols, subplot_specs = process_show_input_objs(
        objects,
        row,
        col,
        output,
        sumup,
    )
    kwargs["max_rows"], kwargs["max_cols"] = max_rows, max_cols
    kwargs["subplot_specs"] = subplot_specs

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

    # pylint: disable=import-outside-toplevel
    display_func = getattr(
        import_module(f"magpylib._src.display.backend_{backend}"), f"display_{backend}"
    )

    if markers:
        objects = list(objects) + [
            {
                "objects": [MagpyMarkers(*markers)],
                "row": 1,
                "col": 1,
                "output": "model3d",
            }
        ]

    return display_func(
        *objects,
        zoom=zoom,
        canvas=canvas,
        animation=animation,
        return_fig=return_fig,
        **kwargs,
    )


def show(*objects, row=None, col=None, output=None, sumup=None, **kwargs):
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
        Define plotting backend. Must be one of `'matplotlib'`, `'plotly'`. If not
        set, parameter will default to `magpylib.defaults.display.backend` which is
        `'matplotlib'` by installation default.

    canvas: matplotlib.pyplot `AxesSubplot` or plotly `Figure` object, default=`None`
        Display graphical output on a given canvas:
        - with matplotlib: `matplotlib.axes._subplots.AxesSubplot` with `projection=3d.
        - with plotly: `plotly.graph_objects.Figure` or `plotly.graph_objects.FigureWidget`.
        By default a new canvas is created and immediately displayed.

    return_fig: bool, default=False
        If True, the function call returns the figure object.
        - with matplotlib: `matplotlib.figure.Figure`.
        - with plotly: `plotly.graph_objects.Figure` or `plotly.graph_objects.FigureWidget`.
        - with pyvista: `pyvista.Plotter`.

    Returns
    -------
    `None` or figure object

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

    # allows kwargs to override within `with show_context`
    # Example:
    # with magpy.show_context(canvas=fig, zoom=1):
    #   src1.show(row=1, col=1)
    #   magpy.show(src2, row=1, col=2)
    #   magpy.show(src1, src2, row=1, col=3, zoom=10)
    # # -> zoom=10 should override zoom=1 from context

    rco = {"row": row, "col": col, "output": output, "sumup": sumup}
    if ctx.isrunning:
        rco = {k: v for k, v in rco.items() if v is not None}
        ctx.kwargs.update(kwargs)
        ctx_objects = tuple({**o, **rco} for o in ctx.objects_from_ctx)
        objects, *_ = process_show_input_objs(objects + ctx_objects, **rco)
        ctx.objects += tuple(objects)
        return None
    return _show(*objects, **rco, **kwargs)


@contextmanager
def show_context(*objects, row=None, col=None, output=None, sumup=None, **kwargs):
    """Context manager to temporarily set display settings in the `with` statement context.

    You need to invoke as ``show_context(pattern1=value1, pattern2=value2)``.
    """
    # pylint: disable=protected-access
    try:
        ctx.isrunning = True
        objects, *_ = process_show_input_objs(objects, row, col, output, sumup)
        ctx.objects_from_ctx += tuple(objects)
        ctx.kwargs.update(**kwargs)
        yield ctx
        _show(*ctx.objects, **ctx.kwargs)
    finally:
        ctx.reset()

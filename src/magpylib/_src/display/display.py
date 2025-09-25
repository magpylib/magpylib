"""Display function codes"""

import warnings
from contextlib import contextmanager
from importlib import import_module
from typing import ClassVar

from matplotlib.axes import Axes as mplAxes
from matplotlib.figure import Figure as mplFig

from magpylib._src.defaults.defaults_utility import _DefaultValue, get_defaults_dict
from magpylib._src.display.traces_generic import MagpyMarkers, get_frames
from magpylib._src.display.traces_utility import (
    DEFAULT_ROW_COL_PARAMS,
    process_show_input_objs,
)
from magpylib._src.input_checks import (
    check_format_input_backend,
    check_format_input_vector,
    check_input_animation,
    check_input_canvas_update,
)
from magpylib._src.utility import check_path_format

disp_args = set(get_defaults_dict("display"))


class RegisteredBackend:
    """Base class for display backends"""

    backends: ClassVar[dict[str, "RegisteredBackend"]] = {}

    def __init__(
        self,
        *,
        name,
        show_func,
        supports_animation,
        supports_subplots,
        supports_colorgradient,
        supports_animation_output,
    ):
        self.name = name
        self.show_func = show_func
        self.supports = {
            "animation": supports_animation,
            "subplots": supports_subplots,
            "colorgradient": supports_colorgradient,
            "animation_output": supports_animation_output,
        }
        self._register_backend(name)

    def _register_backend(self, name):
        self.backends[name] = self

    @classmethod
    def show(
        cls,
        *objs,
        backend,
        title=None,
        max_rows=None,
        max_cols=None,
        subplot_specs=None,
        **kwargs,
    ):
        """Display function of the current backend"""
        self = cls.backends[backend]
        fallback = {
            "animation": {"animation": False},
            "subplots": {"row": None, "col": None},
            "animation_output": {"animation_output": None},
        }
        for name, params in fallback.items():
            condition = not all(kwargs.get(k, v) == v for k, v in params.items())
            if condition and not self.supports[name]:
                supported = [k for k, v in self.backends.items() if v.supports[name]]
                supported_str = (
                    f"one of {supported!r}"
                    if len(supported) > 1
                    else f"{supported[0]!r}"
                )
                warnings.warn(
                    "Unsupported feature for selected backend: "
                    f"the {backend} backend does not support {name!r}. "
                    f"Use {supported_str} instead. "
                    f"Falling back to: {params}",
                    stacklevel=2,
                )
                kwargs.update(params)
        display_kwargs = {
            k: v
            for k, v in kwargs.items()
            if any(k.startswith(arg) for arg in disp_args)
        }
        kwargs = {k: v for k, v in kwargs.items() if k not in display_kwargs}
        backend_kwargs = {
            k[len(backend) + 1 :]: v
            for k, v in kwargs.items()
            if k.startswith(f"{backend.lower()}_")
        }
        backend_kwargs = {**kwargs.pop(backend, {}), **backend_kwargs}
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith(backend)}
        fig_kwargs = {
            **kwargs.pop("fig", {}),
            **{k[4:]: v for k, v in kwargs.items() if k.startswith("fig_")},
            **backend_kwargs.pop("fig", {}),
            **{k[4:]: v for k, v in backend_kwargs.items() if k.startswith("fig_")},
        }
        show_kwargs = {
            **kwargs.pop("show", {}),
            **{k[5:]: v for k, v in kwargs.items() if k.startswith("show_")},
            **backend_kwargs.pop("show", {}),
            **{k[5:]: v for k, v in backend_kwargs.items() if k.startswith("show_")},
        }
        kwargs = {
            k: v for k, v in kwargs.items() if not (k.startswith(("fig", "show")))
        }
        data = get_frames(
            objs,
            supports_colorgradient=self.supports["colorgradient"],
            backend=backend,
            title=title,
            **display_kwargs,
        )
        return self.show_func(
            data,
            max_rows=max_rows,
            max_cols=max_cols,
            subplot_specs=subplot_specs,
            fig_kwargs=fig_kwargs,
            show_kwargs=show_kwargs,
            **kwargs,
        )


def get_show_func(backend):
    """Return the backend show function"""
    # defer import to show call. Importerror should only fail if unavalaible backend is called
    return lambda *args, backend=backend, **kwargs: getattr(
        import_module(f"magpylib._src.display.backend_{backend}"), f"display_{backend}"
    )(*args, **kwargs)


def infer_backend(canvas):
    """Infers the plotting backend from canvas and environment"""
    # pylint: disable=import-outside-toplevel
    backend = "matplotlib"
    in_notebook = False
    plotly_available = False
    try:
        # pylint: disable=unused-import
        import plotly  # noqa: I001, PLC0415
        from magpylib._src.utility import is_notebook  # noqa: PLC0415

        plotly_available = True
        in_notebook = is_notebook()
        if in_notebook:
            backend = "plotly"
    except ImportError:  # pragma: no cover
        pass
    if isinstance(canvas, mplAxes | mplFig):
        backend = "matplotlib"
    elif plotly_available and isinstance(
        canvas, plotly.graph_objects.Figure | plotly.graph_objects.FigureWidget
    ):
        backend = "plotly"
    else:
        try:
            # pylint: disable=unused-import
            import pyvista  # noqa: PLC0415

            if isinstance(canvas, pyvista.Plotter):
                backend = "pyvista"
        except ImportError:  # pragma: no cover
            pass
    return backend


def _show(
    *objects,
    animation=False,
    markers=None,
    canvas=None,
    canvas_update=None,
    backend=None,
    **kwargs,
):
    """Display objects and paths graphically.

    See `show()` function for docstring details.
    """

    # process input objs
    objects, obj_list_flat, max_rows, max_cols, subplot_specs = process_show_input_objs(
        objects,
        **{k: v for k, v in kwargs.items() if k in DEFAULT_ROW_COL_PARAMS},
    )
    kwargs = {k: v for k, v in kwargs.items() if k not in DEFAULT_ROW_COL_PARAMS}
    canvas_update = check_input_canvas_update(canvas_update, canvas)
    # test if every individual obj_path is good
    check_path_format(obj_list_flat)

    # input checks
    backend = check_format_input_backend(backend)
    check_input_animation(animation)
    check_format_input_vector(
        markers,
        dims=(2,),
        shape_m1=3,
        sig_name="markers",
        sig_type="array-like of shape (n, 3)",
        allow_None=True,
    )

    if markers:
        objects.append({"objects": [MagpyMarkers(*markers)], **DEFAULT_ROW_COL_PARAMS})

    if backend == "auto":
        backend = infer_backend(canvas)

    return RegisteredBackend.show(
        *objects,
        backend=backend,
        animation=animation,
        canvas=canvas,
        canvas_update=canvas_update,
        subplot_specs=subplot_specs,
        max_rows=max_rows,
        max_cols=max_cols,
        **kwargs,
    )


def show(
    *objects,
    # pylint: disable=unused-argument
    backend=_DefaultValue,
    canvas=_DefaultValue,
    animation=_DefaultValue,
    zoom=_DefaultValue,
    markers=_DefaultValue,
    return_fig=_DefaultValue,
    canvas_update=_DefaultValue,
    row=_DefaultValue,
    col=_DefaultValue,
    output=_DefaultValue,
    sumup=_DefaultValue,
    pixel_agg=_DefaultValue,
    style=_DefaultValue,
    **kwargs,
):
    """Display objects and paths graphically.

    Global graphic styles can be set with kwargs as style dictionary or using
    style underscore magic.

    Parameters
    ----------
    *objects : Source | Sensor | Collection
        One or multiple Magpylib objects to be displayed.
    backend : {'auto', 'matplotlib', 'plotly', 'pyvista'}, default 'auto'
        With ``'auto'`` the backend becomes ``'plotly'`` inside
        a notebook when Plotly is installed, otherwise ``'matplotlib'``. If
        ``canvas`` is provided, its type determines the backend.
    canvas : None | matplotlib.Figure | plotly.Figure | pyvista.Plotter, default None
        Existing canvas to draw on. If ``None``, a new canvas is created and
        displayed.
    animation : bool | float, default False
        If ``True`` and at least one object has a path, the path is animated.
        A positive float sets the total animation duration in seconds (Plotly only).
    zoom : float, default 0.0
        3D plot zoom level. 0 means tight bounds.
    markers : array-like, shape (n, 3) | None, default None
        Global position markers shown as points.
    return_fig : bool, default False
        If ``True``, return the underlying figure object (Figure / FigureWidget / Plotter).
    canvas_update : str | bool, default 'auto'
        Layout update behaviour when using a provided canvas. With ``'auto'``
        applies internal layout only for newly created canvases. ``True`` forces
        update, ``False`` suppresses it.
    row : int | None, default None
        Subplot row index.
    col : int | None, default None
        Subplot column index.
    output : str | tuple[str, ...], default 'model3d'
        Plot output type. ``'model3d'`` shows 3D geometry. Field plots are defined via
        component strings like ``'Bx'``, ``'Bxy'``, ``'Hyz'``. Multiple axes in a string
        imply vector norm combination (e.g., ``'Bxy'`` => ``sqrt(Bx**2 + By**2)``).
    sumup : bool, default True
        Sum field contributions of sources.
    pixel_agg : str, default 'mean'
        NumPy reducer applied across sensor pixels (e.g., ``'min'``, ``'max'``, ``'std'``).
    style : dict | None, default None
        Global style overrides, e.g. ``{'color': 'red'}`` or via underscore magic
        (``style_color='red'``). Applied to matching objects.

    Returns
    -------
    None | matplotlib.Figure | plotly.Figure | pyvista.Plotter
        The created/updated figure object if ``return_fig=True``; otherwise ``None``.

    Examples
    --------

    Display multiple objects, object paths, markers in 3D using Matplotlib or Plotly:

    >>> import magpylib as magpy
    >>> src = magpy.magnet.Sphere(polarization=(0, 0, 1), diameter=1)
    >>> src.move([(0.1*x, 0, 0) for x in range(50)])
    Sphere...
    >>> src.rotate_from_angax(angle=[*range(0, 400, 10)], axis='z', anchor=0, start=11)
    Sphere...
    >>> ts = [-.4, 0, .4]
    >>> sens = magpy.Sensor(position=(0, 0, 2), pixel=[(x, y, 0) for x in ts for y in ts])
    >>> magpy.show(src, sens) # doctest: +SKIP
    >>> magpy.show(src, sens, backend='plotly') # doctest: +SKIP
    >>> # graphic output

    Display output on your own canvas (here a Matplotlib 3d-axes):

    >>> import matplotlib.pyplot as plt
    >>> import magpylib as magpy
    >>> my_axis = plt.axes(projection='3d')
    >>> magnet = magpy.magnet.Cuboid(polarization=(1, 1, 1), dimension=(1, 2, 3))
    >>> sens = magpy.Sensor(position=(0, 0, 3))
    >>> magpy.show(magnet, sens, canvas=my_axis, zoom=1)
    >>> plt.show() # doctest: +SKIP
    >>> # graphic output

    Use sophisticated figure styling options accessible from defaults, as individual object styles
    or as global style arguments in display.

    >>> import magpylib as magpy
    >>> src1 = magpy.magnet.Sphere(position=[(0, 0, 0), (0, 0, 3)], diameter=1, polarization=(1, 1, 1))
    >>> src2 = magpy.magnet.Sphere(
    ...     position=[(1, 0, 0), (1, 0, 3)],
    ...     diameter=1,
    ...     polarization=(1, 1, 1),
    ...     style_path_show=False
    ... )
    >>> magpy.defaults.display.style.magnet.magnetization.size = 2
    >>> src1.style.magnetization.size = 1
    >>> magpy.show(src1, src2, style_color='r') # doctest: +SKIP
    >>> # graphic output

    Use a context manager to jointly animate 3d and 2d subplots

    >>> import magpylib as magpy
    >>> import numpy as np
    >>> import plotly.graph_objects as go
    >>> path_len = 40
    >>> sensor = magpy.Sensor()
    >>> cyl1 = magpy.magnet.Cylinder(
    ...    polarization=(.1, 0, 0),
    ...    dimension=(1, 2),
    ...    position=(4, 0, 0),
    ...    style_label="Cylinder1",
    ... )
    >>> sensor.move(np.linspace((0, 0, -3), (0, 0, 3), path_len), start=0)
    Sensor(id=...)
    >>> cyl1.rotate_from_angax(angle=np.linspace(0, 300, path_len), start=0, axis="z", anchor=0)
    Cylinder(id=...)
    >>> cyl2 = cyl1.copy().move((0, 0, 5))
    >>> fig = go.Figure()
    >>> with magpy.show_context(cyl1, cyl2, sensor, canvas=fig, backend="plotly", animation=True):
    ...    magpy.show(col=1, output="model3d")
    ...    magpy.show(col=2, output="Bxy", sumup=True)
    ...    magpy.show(col=3, output="Bz", sumup=False)
    >>> fig.show() # doctest: +SKIP
    >>> # graphic output
    """
    kwargs.update(
        {
            k: v
            for k, v in locals().items()
            if v is not _DefaultValue and k not in ("objects", "kwargs")
        }
    )
    if ctx.isrunning:
        rco = {k: v for k, v in kwargs.items() if k in DEFAULT_ROW_COL_PARAMS}
        ctx.kwargs.update(
            {k: v for k, v in kwargs.items() if k not in DEFAULT_ROW_COL_PARAMS}
        )
        ctx_objects = tuple({**o, **rco} for o in ctx.objects_from_ctx)
        objects, *_ = process_show_input_objs(ctx_objects + objects, **rco)
        ctx.objects += tuple(objects)
        return None
    return _show(*objects, **kwargs)


@contextmanager
def show_context(
    *objects,
    # pylint: disable=unused-argument
    backend=_DefaultValue,
    canvas=_DefaultValue,
    animation=_DefaultValue,
    zoom=_DefaultValue,
    markers=_DefaultValue,
    return_fig=_DefaultValue,
    canvas_update=_DefaultValue,
    row=_DefaultValue,
    col=_DefaultValue,
    output=_DefaultValue,
    sumup=_DefaultValue,
    pixel_agg=_DefaultValue,
    style=_DefaultValue,
    **kwargs,
):
    """Context manager for grouping multiple ``show()`` calls with shared settings.

    Use this to apply common display options across several successive ``show()``
    calls and have them rendered together on a single canvas and/or subplot
    layout. All supplied options are remembered during the context and applied
    to the final combined render when the context exits.

    Parameters
    ----------
    objects : Source | Sensor | Collection
        One or more Magpylib objects to register up-front for display.
    backend : str, default magpylib.defaults.display.backend
        Plotting backend: ``'auto'``, ``'matplotlib'``, ``'plotly'``, or
        ``'pyvista'``.
    canvas : None | matplotlib.Figure | plotly.Figure | pyvista.Plotter, default None
        Existing canvas to draw on. If ``None``, a new canvas is created.
    animation : bool | float, default False
        If ``True`` (and at least one object has a path) the path is animated.
        A positive float sets total animation duration in seconds (Plotly only).
    zoom : float, default 0.0
        3D plot zoom level 0.0 means tight bounds.
    markers : array-like, shape (n, 3) | None, default None
        Global position markers shown as points.
    return_fig : bool, default False
        If ``True``, return the underlying figure from the final render.
    canvas_update : str | bool, default 'auto'
        Layout update behaviour when using a provided canvas: ``'auto'``,
        ``True``, or ``False``.
    row : int | None, default None
        Subplot row index for all enclosed ``show()`` calls that omit ``row``.
    col : int | None, default None
        Subplot column index for all enclosed ``show()`` calls that omit ``col``.
    output : str | tuple[str, ...], default 'model3d'
        Plot output type (e.g., ``'model3d'``, ``'Bx'``, ``'Bxy'``, ``'Hyz'``).
    sumup : bool, default True
        Sum source contributions when ``output != 'model3d'``.
    pixel_agg : str, default 'mean'
        NumPy reducer applied across sensor pixels for non-``'model3d'`` outputs
        (e.g., ``'min'``, ``'max'``, ``'std'``).
    style : dict | None, default None
        Global style overrides (e.g., ``{'color': 'red'}``) or via underscore
        magic (e.g., ``style_color='red'``).
    **kwargs
        Additional backend, figure (``fig_*``), and show (``show_*``) options.

    Yields
    ------
    DisplayContext
        The active display context. After the ``with`` block, the combined
        render result is available as ``ctx.show_return_value``.

    Returns
    -------
    None
        The context manager does not return a value.

    Notes
    -----
    - All ``show()`` calls inside the context inherit unspecified options from
      the context manager arguments.
    - Objects passed to the context are combined with objects passed to each
      inner ``show()`` call.
    - On exit, a single ``show()`` is executed with the aggregated objects and
      options.

    Examples
    --------
    Create a 1x3 Plotly layout and render 3 outputs in one animation:

    >>> import magpylib as magpy
    >>> import plotly.graph_objects as go
    >>> fig = go.Figure()
    >>> with magpy.show_context(src1, src2, sensor, canvas=fig, backend='plotly', animation=True) as ctx:
    ...     magpy.show(col=1, output='model3d')
    ...     magpy.show(col=2, output='Bxy', sumup=True)
    ...     magpy.show(col=3, output='Bz', sumup=False)
    >>> fig  # doctest: +SKIP
    """
    # pylint: disable=protected-access
    kwargs.update(
        {
            k: v
            for k, v in locals().items()
            if v is not _DefaultValue and k not in ("objects", "kwargs")
        }
    )
    try:
        ctx.isrunning = True
        rco = {k: v for k, v in kwargs.items() if k in DEFAULT_ROW_COL_PARAMS}
        objects, *_ = process_show_input_objs(objects, **rco)
        ctx.objects_from_ctx += tuple(objects)
        ctx.kwargs.update(
            {k: v for k, v in kwargs.items() if k not in DEFAULT_ROW_COL_PARAMS}
        )
        yield ctx
        ctx.show_return_value = _show(*ctx.objects, **ctx.kwargs)
    finally:
        ctx.reset(reset_show_return_value=False)


class DisplayContext:
    """Display context class"""

    show = staticmethod(show)

    def __init__(self, isrunning=False):
        self.isrunning = isrunning
        self.objects = ()
        self.objects_from_ctx = ()
        self.kwargs = {}
        self.show_return_value = None

    def reset(self, reset_show_return_value=True):
        """Reset display context"""
        self.isrunning = False
        self.objects = ()
        self.objects_from_ctx = ()
        self.kwargs = {}
        if reset_show_return_value:
            self.show_return_value = None


ctx = DisplayContext()


RegisteredBackend(
    name="matplotlib",
    show_func=get_show_func("matplotlib"),
    supports_animation=True,
    supports_subplots=True,
    supports_colorgradient=False,
    supports_animation_output=False,
)


RegisteredBackend(
    name="plotly",
    show_func=get_show_func("plotly"),
    supports_animation=True,
    supports_subplots=True,
    supports_colorgradient=True,
    supports_animation_output=False,
)

RegisteredBackend(
    name="pyvista",
    show_func=get_show_func("pyvista"),
    supports_animation=True,
    supports_subplots=True,
    supports_colorgradient=True,
    supports_animation_output=True,
)

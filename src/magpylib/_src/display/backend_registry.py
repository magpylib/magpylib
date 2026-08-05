"""Display backend registry.

This module deliberately imports nothing from magpylib at module level. Both
`input_checks` and the defaults tree have to consult the registry to validate a
backend name, and both are imported long before the display layer; anything
this module pulled in eagerly would close that loop. What `RegisteredBackend.show`
needs from the display layer is therefore imported inside the call, by which
time everything is loaded.
"""

# pylint: disable=import-outside-toplevel

import warnings
from dataclasses import replace
from functools import cache
from importlib import import_module

from magpylib._src.display.api import API_VERSION, DisplayBackend, Scene


@cache
def _display_arg_names():
    """Names of the `magpy.defaults.display` settings, resolved lazily."""
    from magpylib._src.defaults.defaults_utility import (  # noqa: PLC0415
        get_defaults_dict,
    )

    return set(get_defaults_dict("display"))


def register_backend(
    name,
    show_func,
    *,
    supports_animation=False,
    supports_subplots=False,
    supports_colorgradient=False,
    supports_animation_output=False,
    supports_native_traces=True,
    merge_traces=True,
    handles_traces=None,
    accepts_options=None,
):
    """Register a display backend from a plain function.

    The imperative counterpart to subclassing
    `magpylib.graphics.backend.DisplayBackend`: use this for a backend defined
    in a script or notebook, and the entry-point group ``magpylib.backends``
    for one shipped in a package, where ``pip install`` should be enough.

    Once registered the name is accepted everywhere a built-in name is:
    ``show(backend=name)``, ``magpy.defaults.display.backend`` and
    ``style.model3d.data[].backend``.

    Parameters
    ----------
    name : str
        Name the backend is selected by. Re-registering a name replaces it.
    show_func : callable
        Called as ``show_func(scene)`` with a
        `magpylib.graphics.backend.Scene`; returns the figure object.
    supports_animation, supports_subplots, supports_colorgradient, supports_animation_output : bool, default False
        Capabilities. `show` warns and falls back rather than handing the
        backend something it has not declared it can draw. They default to
        False so a capability added in a later magpylib release never changes
        an existing backend's behaviour.
    supports_native_traces : bool, default True
        Whether models attached via ``style.model3d.data`` naming this backend
        are rendered. False skips them with a warning.
    merge_traces : bool, default True
        Preference, not a capability: whether traces of different objects may
        be merged into fewer, larger ones.
    handles_traces : set of str, optional
        Trace ``type`` values this backend draws. ``None`` assumes all.
        Declaring it lets magpylib warn about a type the backend never handles
        rather than silently omitting it.
    accepts_options : set of str, optional
        Extra keyword arguments this backend accepts, forwarded through
        `Scene.options`. ``None`` accepts anything, which also means a
        misspelled argument passes unnoticed; declaring the set lets magpylib
        warn about it.

    Returns
    -------
    type
        The generated `DisplayBackend` subclass.

    Examples
    --------
    >>> import magpylib as magpy
    >>> def show_types(scene):
    ...     return sorted({t["type"] for f in scene.frames for t in f.traces})
    >>> _ = magpy.register_backend("typelist", show_types)
    >>> src = magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))
    >>> magpy.show(src, backend="typelist")
    ['mesh3d', 'scatter3d']
    """
    return _make_backend(
        name=name,
        show_func=show_func,
        supports_animation=supports_animation,
        supports_subplots=supports_subplots,
        supports_colorgradient=supports_colorgradient,
        supports_animation_output=supports_animation_output,
        supports_native_traces=supports_native_traces,
        merge_traces=merge_traces,
        handles_traces=handles_traces,
        accepts_options=accepts_options,
    )


def _make_backend(
    *,
    name,
    show_func,
    supports_animation,
    supports_subplots,
    supports_colorgradient,
    supports_animation_output,
    supports_native_traces=True,
    merge_traces=True,
    handles_traces=None,
    accepts_options=None,
):
    """Build and register a DisplayBackend subclass delegating to show_func."""
    return type(
        f"{name.title()}Backend",
        (DisplayBackend,),
        {
            "name": name,
            "supports_animation": supports_animation,
            "supports_subplots": supports_subplots,
            "supports_colorgradient": supports_colorgradient,
            "supports_animation_output": supports_animation_output,
            "supports_native_traces": supports_native_traces,
            "merge_traces": merge_traces,
            "handles_traces": handles_traces,
            "accepts_options": accepts_options,
            "show": staticmethod(show_func),
        },
    )


#: Internal alias for the adapter under its previous name.
RegisteredBackend = _make_backend


class ShowDispatcher:
    """Resolves a backend by name, applies fallbacks, and hands over the Scene."""

    @classmethod
    def _warn_unsupported(cls, backend, feature, resolution):
        """Warn that `backend` cannot do `feature`, naming one that can."""
        supported = [
            k for k, v in DisplayBackend.backends.items() if v.supports[feature]
        ]
        supported_str = (
            f"one of {supported!r}" if len(supported) > 1 else f"{supported[0]!r}"
        )
        warnings.warn(
            "Unsupported feature for selected backend: "
            f"the {backend} backend does not support {feature!r}. "
            f"Use {supported_str} instead. "
            f"{resolution}",
            stacklevel=3,
        )

    @classmethod
    def _collapse_subplots(cls, objs, backend):
        """Drop a subplot grid onto a single plot, for backends without subplots.

        Only well defined for a homogeneous grid: a panel is either a 3D scene
        or a 2D field plot, and a grid mixing the two has no single-plot
        equivalent. Mixed grids are therefore passed through with a warning
        rather than silently flattened into something wrong.
        """
        from magpylib._src.display.traces_utility import (  # noqa: PLC0415
            process_show_input_objs,
        )

        outputs = {obj.get("output", "model3d") for obj in objs}
        if len(outputs) > 1:
            cls._warn_unsupported(
                backend,
                "subplots",
                "The grid mixes 3D and 2D panels and cannot be collapsed onto a "
                "single plot, so it is passed through unchanged; the figure is "
                "likely to be wrong.",
            )
            return objs

        cls._warn_unsupported(
            backend, "subplots", "Falling back to a single combined plot."
        )
        objs = [{**obj, "row": 1, "col": 1} for obj in objs]
        objs, _, _ = process_show_input_objs(objs)
        return objs

    @classmethod
    def show(
        cls,
        *objs,
        backend,
        title=None,
        max_rows=None,
        max_cols=None,
        **kwargs,
    ):
        """Display function of the current backend"""
        from magpylib._src.display.traces_generic import get_frames  # noqa: PLC0415

        disp_args = _display_arg_names()
        self = DisplayBackend.backends[backend]
        fallback = {
            "animation": {"animation": False},
            "animation_output": {"animation_output": None},
        }
        for name, params in fallback.items():
            condition = not all(kwargs.get(k, v) == v for k, v in params.items())
            if condition and not self.supports[name]:
                cls._warn_unsupported(backend, name, f"Falling back to: {params}")
                kwargs.update(params)

        # subplots are not in the table above: the grid is not carried in
        # `kwargs` -- `row`/`col` are consumed by `process_show_input_objs`
        # before dispatch -- so it is detected from the resolved grid instead.
        if (max_rows, max_cols) != (None, None) and not self.supports["subplots"]:
            objs = cls._collapse_subplots(objs, backend)
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
        scene = get_frames(
            objs,
            supports_colorgradient=self.supports["colorgradient"],
            backend=backend,
            title=title,
            merge_traces=self.merge_traces,
            **display_kwargs,
        )
        # complete the envelope: get_frames knows about geometry and frames,
        # the dispatch point knows about the canvas and the user's options.
        scene = replace(
            scene,
            canvas=kwargs.pop("canvas", None),
            canvas_update=kwargs.pop("canvas_update", True),
            return_fig=kwargs.pop("return_fig", False),
            legend_maxitems=kwargs.pop("legend_maxitems", Scene.legend_maxitems),
            animation=replace(scene.animation, repeat=kwargs.pop("repeat", False)),
            fig_kwargs=fig_kwargs,
            show_kwargs=show_kwargs,
            # anything magpylib does not interpret is the backend's own
            options=kwargs,
        )
        if self.api_version != API_VERSION:
            warnings.warn(
                f"The {backend!r} backend declares api_version "
                f"{self.api_version}, but this magpylib emits version "
                f"{API_VERSION}. The figure may be wrong or incomplete.",
                stacklevel=2,
            )
        unaccepted = self.unaccepted_options(scene)
        if unaccepted:
            warnings.warn(
                f"show() got unexpected keyword argument(s) "
                f"{sorted(unaccepted)!r} for the {backend} backend; they are "
                "ignored. Check for a typo.",
                stacklevel=2,
            )
        unhandled = self.unhandled_trace_types(scene)
        if unhandled:
            warnings.warn(
                f"The {backend} backend does not declare support for trace "
                f"type(s) {sorted(unhandled)!r}; they may not be drawn.",
                stacklevel=2,
            )
        return self.show(scene)


def get_show_func(backend):
    """Return the backend show function"""
    # defer import to show call. Importerror should only fail if unavalaible backend is called
    return lambda *args, backend=backend, **kwargs: getattr(
        import_module(f"magpylib._src.display.backend_{backend}"), f"display_{backend}"
    )(*args, **kwargs)


RegisteredBackend(
    name="matplotlib",
    show_func=get_show_func("matplotlib"),
    handles_traces=frozenset({"mesh3d", "scatter3d", "scatter"}),
    accepts_options=frozenset({"antialiased", "return_animation"}),
    supports_animation=True,
    supports_subplots=True,
    supports_colorgradient=False,
    supports_animation_output=False,
)


RegisteredBackend(
    name="plotly",
    show_func=get_show_func("plotly"),
    handles_traces=frozenset({"mesh3d", "scatter3d", "scatter"}),
    accepts_options=frozenset({"renderer"}),
    supports_animation=True,
    supports_subplots=True,
    supports_colorgradient=True,
    supports_animation_output=False,
)

RegisteredBackend(
    name="pyvista",
    show_func=get_show_func("pyvista"),
    handles_traces=frozenset({"mesh3d", "scatter3d", "scatter"}),
    accepts_options=frozenset({"jupyter_backend", "mp4_quality"}),
    supports_animation=True,
    supports_subplots=True,
    supports_colorgradient=True,
    supports_animation_output=True,
    # pyvista has never consumed these: its constructors take points, centers
    # and radii rather than the named x/y/z arrays place_and_orient_model3d
    # transforms, so the generic placement contract does not fit them.
    supports_native_traces=False,
)

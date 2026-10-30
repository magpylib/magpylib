"""Display backend registry.

This module deliberately imports nothing from magpylib at module level. Both
`input_checks` and the defaults tree have to consult the registry to validate a
backend name, and both are imported long before the display layer; anything
this module pulled in eagerly would close that loop. What the dispatcher
needs from the display layer is therefore imported inside the call, by which
time everything is loaded.
"""

# pylint: disable=import-outside-toplevel

import inspect
import sys
import warnings
from dataclasses import replace
from functools import cache
from importlib import import_module

from magpylib._src.display.api import API_VERSION, DisplayBackend, user_stacklevel


@cache
def _display_arg_names():
    """Names of the `magpy.defaults.display` settings, resolved lazily."""
    from magpylib._src.defaults.defaults_utility import (  # noqa: PLC0415
        get_defaults_dict,
    )

    return set(get_defaults_dict("display"))


@cache
def _consumed_arg_names():
    """Names `show()` consumes before dispatch, which no backend can claim:
    its own parameters, the per-panel ones, and what the dispatcher reads."""
    from magpylib._src.display.display import show  # noqa: PLC0415
    from magpylib._src.display.traces_utility import (  # noqa: PLC0415
        DEFAULT_ROW_COL_PARAMS,
    )

    own = {
        p for p in inspect.signature(show).parameters if p not in ("objects", "kwargs")
    }
    return frozenset(
        own | set(DEFAULT_ROW_COL_PARAMS) | {"title", "repeat", "legend_maxitems"}
    )


def register_backend(name, show_func, **capabilities):
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
        Name the backend is selected by. Re-registering a name replaces it,
        with a warning when the backend replaced was registered from another
        module, such as a built-in.
    show_func : callable
        Called as ``show_func(scene)`` with a
        `magpylib.graphics.backend.Scene`; returns the figure object.
    supports_animation, supports_subplots, supports_colorgradient, supports_animation_output : bool, default False
        Capabilities. `show` warns and falls back rather than handing the
        backend something it has not declared it can draw, except that
        without `supports_colorgradient` the payload changes without a
        warning, and that a grid mixing 3D and 2D panels arrives unchanged,
        with a warning. They default to False so a capability added in a
        later magpylib release never changes an existing backend's behavior.
    supports_native_traces : bool, default False
        Whether models attached via ``style.model3d.data`` naming this backend
        are rendered. False skips them with a warning.
    merge_traces : bool, default True
        Preference, not a capability: whether traces of different objects may
        be merged into fewer, larger ones.
    draws_pixels : bool, default False
        Whether the backend draws sensor pixels itself, from one ``pixels``
        trace per sensor, rather than as part of the sensor's mesh.
    handles_traces : set of str, optional
        Trace ``type`` values this backend draws. ``None`` assumes all.
        Declaring it lets magpylib warn about a type the backend never handles
        rather than silently omitting it.
    accepts_options : set of str, optional
        Extra keyword arguments this backend accepts, forwarded through
        `Scene.options`. ``None`` accepts anything, which also means a
        misspelled argument passes unnoticed; declaring the set lets magpylib
        warn about it. A trailing ``*`` accepts a namespace, e.g. ``"fig_*"``.
    api_version : int, optional
        The payload contract the backend was written against; pin it. Unset,
        it counts as 1.
    description, url : str, optional
        Shown where magpylib lists the backends, such as the error for an
        unknown backend name.

    Any other keyword is rejected; `DisplayBackend` lists the same names as
    class attributes.

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
    >>> magpy.show(src, backend="typelist", return_fig=True)
    ['mesh3d', 'scatter3d']
    """
    if not isinstance(name, str):
        msg = f"register_backend() needs a name string, not {name!r}"
        raise TypeError(msg)
    if not name:
        msg = "register_backend() needs a non-empty name"
        raise ValueError(msg)
    unknown = set(capabilities) - DisplayBackend._declarable
    if unknown:
        msg = (
            f"register_backend() got unknown backend capability "
            f"{sorted(unknown)!r}; see DisplayBackend for the accepted names."
        )
        raise TypeError(msg)
    return type(
        f"{name.title()}Backend",
        (DisplayBackend,),
        {
            "name": name,
            "show": staticmethod(show_func),
            # the caller's module, so that replacing a backend registered
            # elsewhere -- a built-in, say -- warns as a subclass would
            "__module__": sys._getframe(1).f_globals.get("__name__", __name__),
            **capabilities,
        },
    )


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
            stacklevel=user_stacklevel(),
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
        consumed = _consumed_arg_names()
        # the backend's own, spelled ``<backend>={...}`` or ``<backend>_<name>``.
        # Its prefix is taken first, so that a backend called "units_probe"
        # keeps "units_probe_x" -- unless the name *is* a show() argument:
        # ``style_color`` stays magpylib's even for a backend called "style"
        head = f"{backend}_"
        owns_prefix = not (backend in disp_args or backend in consumed)
        typed = {
            k[len(head) :]: k for k in kwargs if owns_prefix and k.startswith(head)
        }
        prefixed = {stripped: kwargs[k] for stripped, k in typed.items()}
        rest = {k: v for k, v in kwargs.items() if k not in typed.values()}
        # what magpylib interprets: the defaults tree's names, bare or as a
        # ``<name>_`` prefix. The underscore matters: ``stylesheet`` is a
        # backend's option, ``style_color`` is magpylib's.
        display_kwargs = {k: v for k, v in rest.items() if _reserved(k, disp_args)}
        options = {k: v for k, v in rest.items() if k not in display_kwargs}
        # what the dispatcher itself reads, from the bare spelling only:
        # ``plotly_return_fig`` is the backend's own option
        canvas = options.pop("canvas", None)
        canvas_update = options.pop("canvas_update", True)
        return_fig = options.pop("return_fig", False)
        repeat = options.pop("repeat", False)
        own_dict = options.pop(backend, {})
        if not isinstance(own_dict, dict):
            msg = (
                f"show() got {backend}={own_dict!r}; the backend-named argument "
                f"takes a dict of that backend's options, e.g. "
                f"{backend}={{'renderer': 'png'}}."
            )
            raise TypeError(msg)
        # precedence, lowest first: ``fig={...}``, ``fig_a=``, the backend's
        # dict, the backend's prefix -- each flattened one level before the
        # merge, so ``probe={"fig": {"a": 1, "b": 1}}`` with ``probe_fig={"a": 2}``
        # gives fig_a=2 and keeps fig_b=1
        options = {
            **_flat_namespaces(options),
            **_flat_namespaces(own_dict),
            **_flat_namespaces(prefixed),
        }
        if not self._reserved_checked:
            claimed = sorted(
                n
                for n in self.accepts_options or ()
                if _reserved(n, disp_args) or n in consumed
            )
            if claimed:
                warnings.warn(
                    f"The {backend} backend declares option(s) {claimed} that "
                    f"magpylib reserves and never passes on: show()'s own "
                    f"arguments, and any name equal to, or starting with, one "
                    f"of {sorted(disp_args)} followed by an underscore.",
                    stacklevel=user_stacklevel(),
                )
            own_prefix = sorted(
                n for n in self.accepts_options or () if n.startswith(head)
            )
            if own_prefix:
                warnings.warn(
                    f"The {backend} backend declares option(s) {own_prefix} with "
                    f"its own name as prefix; the prefix is stripped before the "
                    "backend sees an option, so declare the bare name.",
                    stacklevel=user_stacklevel(),
                )
            if not owns_prefix:
                warnings.warn(
                    f"The backend name {backend!r} is also a show() argument, so "
                    f"neither {backend}={{...}} nor {backend}_<option> can reach "
                    "it; rename the backend.",
                    stacklevel=user_stacklevel(),
                )
            self._reserved_checked = True
        scene = get_frames(
            objs,
            supports_colorgradient=self.supports["colorgradient"],
            backend=backend,
            title=title,
            merge_traces=self.merge_traces,
            draws_pixels=self.draws_pixels,
            supports_native_traces=self.supports["native_traces"],
            **display_kwargs,
        )
        # complete the envelope: get_frames knows about geometry and frames,
        # the dispatch point knows about the canvas and the user's options.
        scene = replace(
            scene,
            canvas=canvas,
            canvas_update=canvas_update,
            return_fig=return_fig,
            animation=replace(scene.animation, repeat=repeat),
            # anything magpylib does not interpret is the backend's own
            options=options,
        )
        # unset counts as version 1, the first public one
        declared = 1 if self.api_version is None else self.api_version
        if declared != API_VERSION:
            direction = "older" if declared < API_VERSION else "newer"
            advice = (
                "The backend may misread what it is handed; look for an update of it."
                if direction == "older"
                else "It may expect what this magpylib does not send; update magpylib."
            )
            said = (
                "does not declare api_version, so counts as 1,"
                if self.api_version is None
                else f"declares api_version {declared},"
            )
            warnings.warn(
                f"The {backend!r} backend {said} {direction} than the version "
                f"{API_VERSION} this magpylib emits. {advice}",
                stacklevel=user_stacklevel(),
            )
        unaccepted = self.unaccepted_options(scene)
        if unaccepted:
            warnings.warn(
                f"show() got unexpected keyword argument(s) "
                f"{sorted(typed.get(n, n) for n in unaccepted)!r} for the "
                f"{backend} backend, which does not declare them; they are "
                "passed on in scene.options anyway. Check for a typo.",
                stacklevel=user_stacklevel(),
            )
        unhandled = self.unhandled_trace_types(scene)
        if unhandled:
            warnings.warn(
                f"The {backend} backend does not declare support for trace "
                f"type(s) {sorted(unhandled)!r}; they may not be drawn.",
                stacklevel=user_stacklevel(),
            )
        figure = self.show(scene)
        # show() documents that it returns the figure only when return_fig is
        # set. All three built-in backends return None otherwise, so the promise
        # held by convention rather than by construction -- a backend that
        # returns its figure unconditionally leaked it into the caller.
        return figure if scene.return_fig else None


def _reserved(name, display_args):
    """Whether `name` is one magpylib interprets: a display default's name,
    bare or as the ``<name>_`` prefix of a nested one."""
    return any(name == arg or name.startswith(f"{arg}_") for arg in display_args)


def _flat_namespaces(options):
    """``fig={...}`` and ``show={...}`` as ``fig_*``/``show_*`` keys.

    One level, values verbatim: ``fig={"layout": {"width": 400}}`` is
    ``fig_layout={"width": 400}``, exactly what the underscore spelling
    passes. The dict spelling is the lower-precedence one: ``fig_a=2`` beats
    ``fig={"a": 1}``, as it always has.
    """
    flat = {}
    for namespace in ("fig", "show"):
        nested = options.get(namespace)
        if isinstance(nested, dict):
            flat.update({f"{namespace}_{k}": v for k, v in nested.items()})
    flat.update(
        {
            k: v
            for k, v in options.items()
            if not (k in ("fig", "show") and isinstance(v, dict))
        }
    )
    return flat


def get_show_func(backend):
    """Return the backend show function"""
    # defer import to show call. Importerror should only fail if unavalaible backend is called
    return lambda *args, backend=backend, **kwargs: getattr(
        import_module(f"magpylib._src.display.backend_{backend}"), f"display_{backend}"
    )(*args, **kwargs)


register_backend(
    "matplotlib",
    get_show_func("matplotlib"),
    supports_native_traces=True,
    handles_traces=frozenset({"mesh3d", "scatter3d", "scatter"}),
    accepts_options=frozenset({"antialiased", "return_animation", "fig_*", "show_*"}),
    supports_animation=True,
    supports_subplots=True,
    supports_colorgradient=False,
    supports_animation_output=False,
)


register_backend(
    "plotly",
    get_show_func("plotly"),
    supports_native_traces=True,
    handles_traces=frozenset({"mesh3d", "scatter3d", "scatter"}),
    accepts_options=frozenset({"renderer", "fig_*", "show_*"}),
    supports_animation=True,
    supports_subplots=True,
    supports_colorgradient=True,
    supports_animation_output=False,
)

register_backend(
    "pyvista",
    get_show_func("pyvista"),
    handles_traces=frozenset({"mesh3d", "scatter3d", "scatter"}),
    accepts_options=frozenset({"jupyter_backend", "mp4_quality", "fig_*", "show_*"}),
    supports_animation=True,
    supports_subplots=True,
    supports_colorgradient=True,
    supports_animation_output=True,
    # pyvista has never consumed these: its constructors take points, centers
    # and radii rather than the named x/y/z arrays place_and_orient_model3d
    # transforms, so the generic placement contract does not fit them.
    supports_native_traces=False,
)

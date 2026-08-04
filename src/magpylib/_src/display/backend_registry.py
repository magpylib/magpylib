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
from functools import cache
from importlib import import_module
from typing import ClassVar


@cache
def _display_arg_names():
    """Names of the `magpy.defaults.display` settings, resolved lazily."""
    from magpylib._src.defaults.defaults_utility import (  # noqa: PLC0415
        get_defaults_dict,
    )

    return set(get_defaults_dict("display"))


class RegisteredBackend:
    """A plotting backend `show` can dispatch to.

    Instantiating registers the backend under `name`; the registry is what
    `check_format_input_backend` and the `backend` style/defaults fields
    validate against, so registering is sufficient to make the name usable.

    This is internal for now. The public entry point for third-party backends
    is still being designed -- see `DISPLAY_BACKEND_API.md`. The notes below
    describe the current hand-off, and are the reason that design exists.

    Parameters
    ----------
    name: str
        Name the backend is selected by. Re-registering a name replaces it.
    show_func: callable
        Called as ``show_func(data, max_rows=, max_cols=, subplot_specs=,
        fig_kwargs=, show_kwargs=, canvas=, canvas_update=)``.
    supports_animation: bool
        Whether a path can be rendered as an animation. If False, `show` warns
        and falls back to a static figure.
    supports_subplots: bool
        Whether traces can be placed on a `row`/`col` grid. If False, `show`
        warns and collapses the grid onto a single plot -- except a grid mixing
        3D and 2D panels, which has no single-plot equivalent and is passed
        through with a warning instead.
    supports_colorgradient: bool
        Whether vertex colors are interpolated across a mesh. If False, magnet
        meshes are geometrically sliced per color band instead.
    supports_animation_output: bool
        Whether the animation can be written to a file (`.mp4`/`.gif`). If
        False, `show` warns and falls back to displaying it.

    Notes
    -----
    `show_func` receives a dict with keys ``frames``, ``ranges``, ``labels``
    and ``input_kwargs``. Each frame is a dict with ``name``, ``data``,
    ``extra_backend_traces`` and ``layout``; ``data`` holds `mesh3d` and
    `scatter3d` traces as plain dicts of numpy arrays.

    Three things are easy to miss when writing a backend:

    - **`frame["extra_backend_traces"]` must be consumed.** When a user
      attaches a native model via ``style.model3d.data`` naming this backend,
      the trace is routed into this list rather than into ``frame["data"]``,
      already positioned and oriented. A backend that ignores the list silently
      drops the user's models -- no warning, no error.
    - **2D traces.** With ``output="Bx"`` (etc.) rather than ``"model3d"``,
      frames also carry plain `scatter` traces, not `scatter3d`. A pure-3D
      backend has no answer for these.
    - **The trace dicts use plotly's vocabulary** (``colorscale``,
      ``showscale``, ``legendgroup``, ``type``), and ``subplot_specs`` uses
      plotly's ``{"type": "scene"|"xy"}``. Neutralizing that is the main reason
      the public API is not simply this class.
    """

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
    def _warn_unsupported(cls, backend, feature, resolution):
        """Warn that `backend` cannot do `feature`, naming one that can."""
        supported = [k for k, v in cls.backends.items() if v.supports[feature]]
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

        Only well defined for a homogeneous grid: `subplot_specs` marks each
        cell as a 3D scene or a 2D field plot, and a grid mixing the two has no
        single-plot equivalent. Mixed grids are therefore passed through with a
        warning rather than silently flattened into something wrong.
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
            return objs, None, None, None

        cls._warn_unsupported(
            backend, "subplots", "Falling back to a single combined plot."
        )
        objs = [{**obj, "row": 1, "col": 1} for obj in objs]
        objs, max_rows, max_cols, subplot_specs = process_show_input_objs(objs)
        return objs, max_rows, max_cols, subplot_specs

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
        from magpylib._src.display.traces_generic import get_frames  # noqa: PLC0415

        disp_args = _display_arg_names()
        self = cls.backends[backend]
        fallback = {
            "animation": {"animation": False},
            "animation_output": {"animation_output": None},
        }
        for name, params in fallback.items():
            condition = not all(kwargs.get(k, v) == v for k, v in params.items())
            if condition and not self.supports[name]:
                self._warn_unsupported(backend, name, f"Falling back to: {params}")
                kwargs.update(params)

        # subplots are not in the table above: the grid is not carried in
        # `kwargs` -- `row`/`col` are consumed by `process_show_input_objs`
        # before dispatch -- so it is detected from the resolved grid instead.
        if (max_rows, max_cols) != (None, None) and not self.supports["subplots"]:
            objs, max_rows, max_cols, subplot_specs = self._collapse_subplots(
                objs, backend
            )
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

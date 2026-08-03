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
        from magpylib._src.display.traces_generic import get_frames  # noqa: PLC0415

        disp_args = _display_arg_names()
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


def register_backend(
    name,
    show_func,
    *,
    supports_animation,
    supports_subplots,
    supports_colorgradient,
    supports_animation_output,
):
    """Register a third-party plotting backend for use with `show`.

    Once registered, the backend can be selected anywhere a built-in backend
    can: ``show(..., backend=name)``, ``magpy.defaults.display.backend``, and
    ``style.model3d.data[].backend``.

    Parameters
    ----------
    name: str
        Name the backend is selected by. Registering an existing name replaces
        it.
    show_func: callable
        Called as ``show_func(data, max_rows=, max_cols=, subplot_specs=,
        fig_kwargs=, show_kwargs=, **kwargs)``, where `data` is the generic
        frame structure described below.
    supports_animation: bool
        Whether the backend can render a path as an animation. If False, `show`
        warns and falls back to a static figure.
    supports_subplots: bool
        Whether the backend can place traces on a `row`/`col` grid. If False,
        `show` warns and falls back to a single plot.
    supports_colorgradient: bool
        Whether the backend interpolates vertex colors across a mesh. If False,
        magnet meshes are geometrically sliced per color band instead.
    supports_animation_output: bool
        Whether the backend can write the animation to a file (`.mp4`/`.gif`).
        If False, `show` warns and falls back to displaying it.

    Returns
    -------
    RegisteredBackend
        The registration, mostly useful for unregistering in tests.

    Notes
    -----
    `show_func` receives a backend-neutral dict with keys ``frames``,
    ``ranges``, ``labels`` and ``input_kwargs``. Each frame is a dict with
    ``name``, ``data``, ``extra_backend_traces`` and ``layout``; ``data`` holds
    `mesh3d` and `scatter3d` traces as plain dicts of numpy arrays.

    Three things are easy to miss when writing a backend:

    - **`frame["extra_backend_traces"]` must be consumed.** When a user
      attaches a native model via ``style.model3d.data`` with
      ``backend="<yourname>"``, the trace is routed into this list rather than
      into ``frame["data"]``, already positioned and oriented. A backend that
      ignores the list silently drops the user's models -- no warning, no error.
    - **2D traces.** With ``output="Bx"`` (etc.) rather than ``"model3d"``,
      frames also carry plain `scatter` traces, not `scatter3d`. A pure-3D
      backend should either render them or document that it cannot.
    - **Subplots.** Every trace carries `row` and `col`. Declaring
      ``supports_subplots=False`` is fine; `show` warns and falls back.

    Examples
    --------
    >>> import magpylib as magpy
    >>> def show_noop(data, **kwargs):
    ...     return data
    >>> _ = magpy.register_backend(
    ...     "noop",
    ...     show_noop,
    ...     supports_animation=False,
    ...     supports_subplots=False,
    ...     supports_colorgradient=False,
    ...     supports_animation_output=False,
    ... )
    >>> src = magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=(0, 0, 1))
    >>> sorted(magpy.show(src, backend="noop"))
    ['frames', 'input_kwargs', 'labels', 'ranges']
    """
    return RegisteredBackend(
        name=name,
        show_func=show_func,
        supports_animation=supports_animation,
        supports_subplots=supports_subplots,
        supports_colorgradient=supports_colorgradient,
        supports_animation_output=supports_animation_output,
    )


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

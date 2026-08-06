"""Cross-backend tests for user-supplied 3D models (`style.model3d`).

This feature is advertised for every backend but was only ever exercised for
Matplotlib, through `make_Cuboid("matplotlib")` -- which supplies both `args`
and `kwargs`, so the args-only form the style docs recommend went untested and
crashed. Pyvista never consumed backend-specific models at all. These tests
pin the behaviour of each backend against each kind of model.
"""

import numpy as np
import pytest

import magpylib as magpy
from magpylib.graphics.model3d import make_Cuboid

BACKENDS = ("matplotlib", "plotly", "pyvista")

# backends that render a model addressed to them by name; pyvista does not,
# see test_display_pyvista.test_pyvista_warns_on_backend_specific_model3d
NATIVE_BACKENDS = ("matplotlib", "plotly")


def source():
    return magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))


def drawn_count(fig, backend):
    """Number of rendered items, however the backend counts them."""
    if backend == "plotly":
        return len(fig.data)
    if backend == "pyvista":
        return len(list(fig.renderer.actors))
    ax = fig.axes[0]
    return len(ax.lines) + len(ax.collections)


def render(obj, backend):
    return drawn_count(magpy.show(obj, backend=backend, return_fig=True), backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_generic_model3d_renders_on_every_backend(backend):
    """A 'generic' model is translated for every backend, pyvista included."""
    obj = source()
    obj.style.model3d.add_trace(
        {
            "backend": "generic",
            "constructor": "scatter3d",
            "kwargs": {"x": (0, 2), "y": (0, 2), "z": (0, 2), "mode": "lines"},
        }
    )
    assert render(obj, backend) > render(source(), backend)


@pytest.mark.parametrize("backend", NATIVE_BACKENDS)
def test_backend_specific_model3d_renders(backend):
    """A model naming the rendering backend is drawn by it."""
    obj = source()
    obj.style.model3d.add_trace(**make_Cuboid(backend, position=(3, 0, 0)))
    assert render(obj, backend) > render(source(), backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_model3d_for_another_backend_is_ignored(backend):
    """A model addressed elsewhere is skipped quietly -- it is not an error."""
    other = "plotly" if backend != "plotly" else "matplotlib"
    obj = source()
    obj.style.model3d.add_trace(**make_Cuboid(other, position=(3, 0, 0)))

    assert render(obj, backend) == render(source(), backend)


@pytest.mark.parametrize("backend", NATIVE_BACKENDS)
def test_model3d_show_false_suppresses_the_model(backend):
    """`show=False` on a trace keeps it out of the figure."""
    obj = source()
    trace = make_Cuboid(backend, position=(3, 0, 0))
    trace["show"] = False
    obj.style.model3d.add_trace(**trace)

    assert render(obj, backend) == render(source(), backend)


def test_matplotlib_model3d_args_only_and_kwargs_only_agree():
    """The two documented Matplotlib spellings must both work.

    `make_Cuboid("matplotlib")` passes args *and* kwargs, which is why the
    args-only form -- the one the style docs describe, with coordsargs
    defaulting to args[0..2] -- went unnoticed when it raised.
    """
    xyz = (np.array([0.0, 2.0]),) * 3

    args_only = source()
    args_only.style.model3d.add_trace(
        backend="matplotlib", constructor="plot", args=xyz
    )
    kwargs_only = source()
    kwargs_only.style.model3d.add_trace(
        backend="matplotlib",
        constructor="plot",
        kwargs={"xs": xyz[0], "ys": xyz[1], "zs": xyz[2]},
        coordsargs={"x": "xs", "y": "ys", "z": "zs"},
    )

    assert render(args_only, "matplotlib") == render(kwargs_only, "matplotlib")
    assert render(args_only, "matplotlib") > render(source(), "matplotlib")


@pytest.mark.parametrize("backend", NATIVE_BACKENDS)
def test_model3d_updatefunc_is_called_at_show_time(backend):
    """`updatefunc` is evaluated per show, not once at attach time."""
    calls = []

    def updatefunc():
        calls.append(1)
        return make_Cuboid(backend, position=(3, 0, 0))

    obj = source()
    obj.style.model3d.add_trace(backend=backend, updatefunc=updatefunc)

    magpy.show(obj, backend=backend, return_fig=True)
    assert calls, "updatefunc was never called"
    before = len(calls)
    magpy.show(obj, backend=backend, return_fig=True)
    assert len(calls) > before, "updatefunc not re-evaluated on the second show"

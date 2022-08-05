import pytest
import pyvista as pv

import magpylib as magpy


def test_Cuboid_display():
    "test simple display with path"
    src = magpy.magnet.Cuboid((0, 0, 1000), (1, 1, 1))
    src.move([[i, 0, 0] for i in range(2)], start=0)
    fig = src.show(return_fig=True, backend="pyvista")
    assert isinstance(fig, pv.Plotter)


def test_animation():
    "animation not supported, should warn and display static"
    pl = pv.Plotter()
    src = magpy.magnet.Cuboid((0, 0, 1000), (1, 1, 1))
    with pytest.warns(UserWarning):
        src.show(canvas=pl, animation=True, backend="pyvista")


def test_ipygany_jupyter_backend():
    """ipygany backend does not support custom colorscales"""
    src = magpy.magnet.Cuboid((0, 0, 1000), (1, 1, 1))
    src.show(return_fig=True, backend="pyvista", jupyter_backend="ipygany")


def test_extra_model3d():
    """test extra model 3d"""
    trace_mesh3d = {
        "constructor": "Mesh3d",
        "kwargs": {
            "x": (1, 0, -1, 0),
            "y": (-0.5, 1.2, -0.5, 0),
            "z": (-0.5, -0.5, -0.5, 1),
            "i": (0, 0, 0, 1),
            "j": (1, 1, 2, 2),
            "k": (2, 3, 3, 3),
            "opacity": 0.5,
            "facecolor": ["blue"] * 2 + ["red"] * 2,
        },
    }
    coll = magpy.Collection(position=(0, -3, 0), style_label="'Mesh3d' trace")
    coll.style.model3d.add_trace(trace_mesh3d)

    magpy.show(coll, return_fig=True, backend="pyvista")

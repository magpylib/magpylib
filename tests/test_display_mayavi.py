import mayavi
from mayavi import mlab

import magpylib as magpy


def test_Cuboid_display():
    "test simple display with path"
    src = magpy.magnet.Cuboid((0, 0, 1000), (1, 1, 1))
    src.move([[i, 0, 0] for i in range(2)], start=0)
    fig = src.show(return_fig=True, style_path_numbering=True, backend="mayavi")
    mlab.options.offscreen = True
    assert isinstance(fig, mayavi.core.scene.Scene)


def test_extra_generic_trace():
    "test simple display with path"
    src = magpy.magnet.Cuboid((0, 0, 1000), (1, 1, 1))
    src.style.model3d.add_trace(
        {
            "constructor": "mesh3d",
            "kwargs": {
                "i": [7, 0, 0, 0, 4, 4, 2, 6, 4, 0, 3, 7],
                "j": [0, 7, 1, 2, 6, 7, 1, 2, 5, 5, 2, 2],
                "k": [3, 4, 2, 3, 5, 6, 5, 5, 0, 1, 7, 6],
                "x": [-1, -1, 1, 1, -1, -1, 1, 1],
                "y": [-1, 1, 1, -1, -1, 1, 1, -1],
                "z": [-1, -1, -1, -1, 1, 1, 1, 1],
                "facecolor": ["red"] * 12,
            },
            "show": True,
        }
    )
    mlab.options.offscreen = True
    fig = mlab.figure()
    src.show(canvas=fig, style_path_numbering=True, backend="mayavi")


def test_animation():
    "test simple display with path"
    mlab.options.offscreen = True
    src = magpy.magnet.Cuboid((0, 0, 1000), (1, 1, 1))
    src.move([[i, 0, 0] for i in range(2)], start=0)
    src.show(animation=True, style_path_numbering=True, backend="mayavi")

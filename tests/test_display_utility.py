import sys
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import plotly
import pytest
import pyvista

import magpylib as magpy
from magpylib._src.display.traces_utility import (
    draw_arrow_from_vertices,
    get_orientation_from_vec,
    merge_scatter3d,
)
from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib._src.utility import is_notebook, open_animation


def test_draw_arrow_from_vertices():
    """tests also the edge case when a vertex is in -y direction"""
    vertices = np.array(
        [
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
        ]
    )
    result = draw_arrow_from_vertices(vertices, sign=1, arrow_size=1)
    expected = np.array(
        [
            [-1.0, 1.0, 1.0],
            [-1.0, 0.0, 1.0],
            [-0.88, 0.2, 1.0],
            [-1.0, 0.0, 1.0],
            [-1.12, 0.2, 1.0],
            [-1.0, 0.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, 0.0],
            [-1.12, -1.0, 0.2],
            [-1.0, -1.0, 0.0],
            [-0.88, -1.0, 0.2],
            [-1.0, -1.0, 0.0],
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 0.0, -1.0],
            [-1.12, -0.2, -1.0],
            [-1.0, 0.0, -1.0],
            [-0.88, -0.2, -1.0],
            [-1.0, 0.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 0.0],
            [-1.12, 1.0, -0.2],
            [-1.0, 1.0, 0.0],
            [-0.88, 1.0, -0.2],
            [-1.0, 1.0, 0.0],
            [-1.0, 1.0, 1.0],
        ]
    )

    np.testing.assert_allclose(
        result, expected, err_msg="draw arrow from vertices failed"
    )


def test_bad_backend():
    """test bad plotting input name"""
    c = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    with pytest.raises(MagpylibBadUserInput):
        c.show(backend="asdf")


@pytest.mark.parametrize(
    ("canvas", "is_notebook_result", "backend"),
    [
        (None, True, "plotly"),
        (None, False, "matplotlib"),
        (plt.subplot(projection="3d"), True, "matplotlib"),
        (plt.subplot(projection="3d"), False, "matplotlib"),
        (plotly.graph_objects.Figure(), True, "plotly"),
        (plotly.graph_objects.Figure(), False, "plotly"),
        (plotly.graph_objects.FigureWidget(), True, "plotly"),
        (plotly.graph_objects.FigureWidget(), False, "plotly"),
        (pyvista.Plotter(), True, "pyvista"),
        (pyvista.Plotter(), False, "pyvista"),
    ],
)
def test_infer_backend(canvas, is_notebook_result, backend):
    """test inferring auto backend"""
    with patch("magpylib._src.utility.is_notebook", return_value=is_notebook_result):
        # pylint: disable=import-outside-toplevel
        from magpylib._src.display.display import infer_backend  # noqa: PLC0415

        assert infer_backend(canvas) == backend


def test_is_notebook_without_ipython():
    """IPython is optional: without it, a script is no notebook"""
    with patch.dict(sys.modules, {"IPython": None}):
        assert not is_notebook()


def test_open_animation_without_viewer(tmp_path):
    """When no viewer opens the animation, the user learns where it is"""
    with (
        patch("magpylib._src.utility.is_notebook", return_value=False),
        patch("webbrowser.open", return_value=False),
        pytest.warns(UserWarning, match="saved at"),
    ):
        assert open_animation(tmp_path / "anim.gif")


def test_merge_scatter3d():
    """test_merge_scatter3d"""

    def get_traces(n):
        return [{"type": "scatter3d", "x": [i], "y": [i], "z": [i]} for i in range(n)]

    merge_scatter3d(*get_traces(1))
    merge_scatter3d(*get_traces(3))


def test_get_orientation_from_vec():
    """test get_orientation_from_vec"""
    # should return ref axis non null vectors and invalid inputs
    # antiparallel to the ref_axis should return valid 180 rotation
    ref_axis = (0, 0, 1)
    vec = np.array(
        [
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [np.nan, 0, 0],
            [0, 0, 0],
            [0, 0, -1],
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        vec = (vec.T / np.linalg.norm(vec, axis=1)).T
    hasnan = np.isnan(vec).any(axis=1)
    orient = get_orientation_from_vec(vec, ref_axis=ref_axis)
    vec[hasnan] = ref_axis
    np.testing.assert_allclose(vec, orient.apply(ref_axis), atol=1e-8)


class BoxedSource(magpy.misc.CustomSource):
    """A source drawn from the scene's box, recording what get_trace is given."""

    _scene_box = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    def get_trace(self, scene_box=None, path_index=-1, **kwargs):
        self.calls.append((scene_box, path_index))
        trace = {"type": "scatter3d", "x": [0], "y": [0], "z": [0]}
        return {**trace, "mode": "markers", **kwargs}


@pytest.mark.parametrize("path", [False, True])
def test_scene_box_leaves_out_the_object_drawn_from_it(path):
    """An object with `_scene_box` gets the box the other objects fill, its own
    position left out of it and of the dipoles' size, at each pose it is drawn."""
    cube = magpy.magnet.Cuboid(dimension=(2, 4, 6), polarization=(0, 0, 1))
    dipole = magpy.misc.Dipole(moment=(0, 0, 1), position=(1, 2, 3))
    boxed = BoxedSource(position=(50, 0, 0))
    if path:
        boxed.position = [(50, 0, 0), (60, 0, 0)]
        boxed.style.path.frames = [0, 1]
    alone = magpy.show(cube, dipole, backend="plotly", return_fig=True)
    fig = magpy.show(cube, dipole, boxed, backend="plotly", return_fig=True)

    expected = [[-1, 1], [-2, 2], [-3, 3]]
    assert [c[1] for c in boxed.calls] == ([0, 1] if path else [-1])
    for box, _ in boxed.calls:
        np.testing.assert_allclose(box, expected)
    # the dipole's arrow is as long as without the boxed source
    arrows = [
        [d.z for d in f.data if d.name.startswith("Dipole")] for f in (alone, fig)
    ]
    np.testing.assert_allclose(np.ptp(arrows[0][0]), np.ptp(arrows[1][0]))

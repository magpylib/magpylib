import numpy as np
import pytest

import magpylib as magpy
from magpylib._src.display.display_utility import draw_arrow_from_vertices
from magpylib._src.exceptions import MagpylibBadUserInput


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
    result = draw_arrow_from_vertices(vertices, current=1, arrow_size=1)
    expected = np.array(
        [
            [
                -1.0,
                -1.0,
                -1.12,
                -1.0,
                -0.88,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.12,
                -1.0,
                -0.88,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.12,
                -1.0,
                -0.88,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.12,
                -1.0,
                -0.88,
                -1.0,
                -1.0,
            ],
            [
                -1.0,
                0.0,
                0.2,
                0.0,
                0.2,
                0.0,
                1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                0.0,
                -0.2,
                0.0,
                -0.2,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.2,
                0.0,
                0.2,
                0.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                0.0,
                -0.2,
                0.0,
                -0.2,
                0.0,
                1.0,
            ],
        ]
    )
    assert np.allclose(result, expected), "draw arrow from vertices failed"


def test_bad_backend():
    """test bad plotting input name"""
    with pytest.raises(MagpylibBadUserInput):
        c = magpy.magnet.Cuboid((0, 0, 1), (1, 1, 1))
        c.show(backend="asdf")

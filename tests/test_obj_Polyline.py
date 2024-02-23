import numpy as np
import pytest

import magpylib as magpy
from magpylib._src.exceptions import MagpylibDeprecationWarning


def test_Polyline_basic1():
    """Basic Polyline class test"""
    src = magpy.current.Polyline(current=100, vertices=[(1, 1, -1), (1, 1, 1)])
    sens = magpy.Sensor()
    B = src.getB(sens)

    x = 5.77350269 * 1e-6
    Btest = np.array([x, -x, 0])

    np.testing.assert_allclose(B, Btest)


def test_Polyline_basic2():
    """Basic Polyline class test 2"""
    src = magpy.current.Polyline(current=-100, vertices=[(1, 1, -1), (1, 1, 1)])
    sens = magpy.Sensor()
    H = src.getH(sens)

    x = 5.77350269 / 4 / np.pi * 10
    Htest = np.array([-x, x, 0])

    np.testing.assert_allclose(H, Htest)


def test_Polyline_basic3():
    """Basic Polyline class test 3"""
    line1 = magpy.current.Polyline(current=100, vertices=[(1, 1, -1), (1, 1, 1)])
    line2 = magpy.current.Polyline(
        current=100, vertices=[(1, 1, -1), (1, 1, 1), (1, 1, -1), (1, 1, 1)]
    )
    sens = magpy.Sensor()
    B = magpy.getB([line1, line2], sens)

    x = 5.77350269 * 1e-6
    Btest = np.array([(x, -x, 0)] * 2)

    np.testing.assert_allclose(B, Btest)


def test_Polyline_repr():
    """Polyline repr test"""
    line = magpy.current.Polyline(current=100, vertices=[(1, 1, -1), (1, 1, 1)])
    assert repr(line)[:8] == "Polyline", "Polyline repr failed"


def test_Polyline_specials():
    """Polyline specials tests"""
    line = magpy.current.Polyline(current=100, vertices=[(0, 0, 0), (1, 1, 1)])
    b = line.getB([0, 0, 0])
    np.testing.assert_allclose(b, np.zeros(3))

    line = magpy.current.Polyline(current=100, vertices=[(0, 0, 0), (0, 0, 0)])
    b = line.getB([1, 2, 3])
    np.testing.assert_allclose(b, np.zeros(3))

    line = magpy.current.Polyline(current=0, vertices=[(1, 2, 3), (3, 2, 1)])
    b = line.getB([0, 0, 0])
    np.testing.assert_allclose(b, np.zeros(3))


def test_line_position_bug():
    """line positions were not properly computed in collections"""
    verts1 = np.array([(1, 0, 0), (0, 1, 0), (-1, 0, 0)])
    verts2 = np.array([(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)])

    poso = [[(0, 0.99, 0.5), (0, 0.99, -0.5)]] * 3

    s1 = magpy.current.Polyline(1, verts1 + np.array((0, 0, 0.5)))
    s2 = magpy.current.Polyline(1, verts2 - np.array((0, 0, 0.5)))
    col = s1 + s2
    # B1 = col.getB([(0,.99,.5), (0,.99,-.5)])
    B1 = col.getB(poso)

    s1 = magpy.current.Polyline(1, verts1, position=(0, 0, 0.5))
    s2 = magpy.current.Polyline(1, verts2, position=(0, 0, -0.5))
    col = s1 + s2
    # B2 = col.getB([(0,.99,.5), (0,.99,-.5)])
    B2 = col.getB(poso)

    np.testing.assert_allclose(B1, B2)


def test_discontinous_line():
    """test discontinuous line"""

    line_1 = magpy.current.Polyline(
        current=1,
        vertices=[
            [0, 0, 0],
            [0, 0, 1],
        ],
    )
    line_2 = magpy.current.Polyline(
        current=1,
        vertices=[
            [1, 0, 0],
            [1, 0, 1],
        ],
    )
    line_12 = magpy.current.Polyline(
        current=1,
        vertices=[
            [None, None, None],
            *line_1.vertices.tolist(),
            [None, None, None],
            [None, None, None],
            *line_2.vertices.tolist(),
            [None, None, None],
        ],
    )

    B1 = magpy.getB((line_1, line_2), (0, 0, 0), sumup=True)
    B2 = line_12.getB((0, 0, 0))

    np.testing.assert_allclose(B1, B2)


def test_old_Line_deprecation_warning():
    """test old calss deprecation warning"""
    with pytest.warns(MagpylibDeprecationWarning):
        old_class = magpy.current.Line()

    new_class = magpy.current.Polyline()
    assert isinstance(old_class, magpy.current.Polyline)
    assert isinstance(new_class, magpy.current.Polyline)

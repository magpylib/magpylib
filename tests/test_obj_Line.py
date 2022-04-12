import numpy as np

import magpylib as magpy


def test_Line_basic1():
    """Basic Line class test"""
    src = magpy.current.Line(current=100, vertices=[(1, 1, -1), (1, 1, 1)])
    sens = magpy.Sensor()
    B = src.getB(sens)

    x = 5.77350269
    Btest = np.array([x, -x, 0])

    assert np.allclose(B, Btest)


def test_Line_basic2():
    """Basic Line class test 2"""
    src = magpy.current.Line(current=-100, vertices=[(1, 1, -1), (1, 1, 1)])
    sens = magpy.Sensor()
    H = src.getH(sens)

    x = 5.77350269 / 4 / np.pi * 10
    Htest = np.array([-x, x, 0])

    assert np.allclose(H, Htest)


def test_Line_basic3():
    """Basic Line class test 3"""
    line1 = magpy.current.Line(current=100, vertices=[(1, 1, -1), (1, 1, 1)])
    line2 = magpy.current.Line(
        current=100, vertices=[(1, 1, -1), (1, 1, 1), (1, 1, -1), (1, 1, 1)]
    )
    sens = magpy.Sensor()
    B = magpy.getB([line1, line2], sens)

    x = 5.77350269
    Btest = np.array([(x, -x, 0)] * 2)

    assert np.allclose(B, Btest)


def test_Line_repr():
    """Line repr test"""
    line = magpy.current.Line(current=100, vertices=[(1, 1, -1), (1, 1, 1)])
    assert line.__repr__()[:4] == "Line", "Line repr failed"


def test_Line_specials():
    """Line specials tests"""
    line = magpy.current.Line(current=100, vertices=[(0, 0, 0), (1, 1, 1)])
    b = line.getB([0, 0, 0])
    assert np.allclose(b, np.zeros(3))

    line = magpy.current.Line(current=100, vertices=[(0, 0, 0), (0, 0, 0)])
    b = line.getB([1, 2, 3])
    assert np.allclose(b, np.zeros(3))

    line = magpy.current.Line(current=0, vertices=[(1, 2, 3), (3, 2, 1)])
    b = line.getB([0, 0, 0])
    assert np.allclose(b, np.zeros(3))


def test_line_position_bug():
    """line positions were not properly computed in collections"""
    verts1 = np.array([(1, 0, 0), (0, 1, 0), (-1, 0, 0)])
    verts2 = np.array([(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)])

    poso = [[(0, 0.99, 0.5), (0, 0.99, -0.5)]] * 3

    s1 = magpy.current.Line(1, verts1 + np.array((0, 0, 0.5)))
    s2 = magpy.current.Line(1, verts2 - np.array((0, 0, 0.5)))
    col = s1 + s2
    # B1 = col.getB([(0,.99,.5), (0,.99,-.5)])
    B1 = col.getB(poso)

    s1 = magpy.current.Line(1, verts1, position=(0, 0, 0.5))
    s2 = magpy.current.Line(1, verts2, position=(0, 0, -0.5))
    col = s1 + s2
    # B2 = col.getB([(0,.99,.5), (0,.99,-.5)])
    B2 = col.getB(poso)

    assert np.allclose(B1, B2)

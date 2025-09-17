from magpylib._src.fields.field_BHfunc import _getBH_func
from magpylib._src.fields.field_BH_circle import _BHJM_circle
import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R
import magpylib as magpy
from magpylib.func import (
    current_circle_field,
    current_polyline_field
)

def test_getBHfunc_bad_inputs1():
    """test bad orientation input"""
    dic = {
        'position': (0,0,0),
        'orientation': 1,
        'observers': (0,0,0),
        'diameter': 3.123,
        'current': 1.123,
    }
    with pytest.raises(TypeError):
        B = _getBH_func(_BHJM_circle, 'B', dic, True)

def test_getBHfunc_bad_inputs2():
    """test bad orientation input"""
    dic = {
        'position': (0,0,0),
        'orientation': None,
        'observers': (0,0,0),
        'diameter': 'woot',
        'current': 1.123,
    }
    with pytest.raises(ValueError):
        B = _getBH_func(_BHJM_circle, 'B', dic, True)

def test_getBHfunc_bad_inputs3():
    """test bad orientation input"""
    dic = {
        'position': (0,0,0),
        'orientation': None,
        'observers': (0,0,0),
        'diameter': [(1,2,3), (1,2,3)],
        'current': 1.123,
    }
    with pytest.raises(ValueError):
        B = _getBH_func(_BHJM_circle, 'B', dic, True)

def test_getBHfunc_bad_inputs4():
    """test bad orientation input"""
    dic = {
        'position': (0,0,0),
        'orientation': None,
        'observers': (0,0,0,0),
        'diameter': (1,2,3),
        'current': 1.123,
    }
    with pytest.raises(ValueError):
        B = _getBH_func(_BHJM_circle, 'B', dic, True)

def test_getBHfunc_bad_inputs5():
    """test bad orientation input"""
    dic = {
        'position': (0,0,0),
        'orientation': None,
        'observers': (0,0),
        'diameter': (1,2,3),
        'current': 1.123,
    }
    with pytest.raises(ValueError):
        B = _getBH_func(_BHJM_circle, 'B', dic, True)


def test_func_circle():
    """test if Circle implementation gives correct output"""
    B = current_circle_field('B', (0,0,0), 2, 1)
    Btest = np.array([0, 0, 0.6283185307179586 * 1e-6])
    np.testing.assert_allclose(B, Btest)

    H = current_circle_field('H', (0,0,0), 2, 1)
    Htest = np.array([0, 0, 0.6283185307179586 * 10 / 4 / np.pi])
    np.testing.assert_allclose(H, Htest)


def test_func_squeeze():
    """test if squeeze works"""
    B1 = current_circle_field('B', (0, 0, 0), 2, 1)
    B2 = current_circle_field('B', [(0, 0, 0)], 2, 1)
    B3 = current_circle_field('B', [(0, 0, 0)], 2, 1, squeeze=False)
    B4 = current_circle_field('B', [(0, 0, 0)]*2, 2, 1)

    assert B1.ndim == 1
    assert B2.ndim == 1
    assert B3.ndim == 2
    assert B4.ndim == 2


def test_func_polyline1():
    """test getBHv with Polyline"""
    H = current_polyline_field(
        'H',
        observers=[(1, 1, 1), (1, 2, 3), (2, 2, 2)],
        currents=1,
        segments_start=(0, 0, 0),
        segments_end=[(0, 0, 0), (2, 2, 2), (2, 2, 2)],
    )
    x = (
        np.array([[0, 0, 0], [0.02672612, -0.05345225, 0.02672612], [0, 0, 0]])
        * 10
        / 4
        / np.pi
    )
    np.testing.assert_allclose(x, H, rtol=1e-05, atol=1e-08)


def test_func_polyline2():
    """test line with pos and rot arguments"""
    x = 0.14142136 * 1e-6

    # z-line on x=1
    B1 = current_polyline_field(
        'B',
        observers=[(0, 0, 0)],
        currents=1,
        segments_start=(1, 0, -1),
        segments_end=(1, 0, 1),
    )
    expected = np.array([0, -x, 0])
    np.testing.assert_allclose(B1, expected, rtol=1e-05, atol=1e-08)


def test_func_polyline3():
    """test line with pos and rot arguments"""
    x = 0.14142136 * 1e-6
    # move z-line to x=-1
    B2 = current_polyline_field(
        'B',
        positions=(-2, 0, 0),
        observers=[(0, 0, 0)],
        currents=1,
        segments_start=(1, 0, -1),
        segments_end=(1, 0, 1),
    )
    np.testing.assert_allclose(B2, np.array([0, x, 0]), rtol=1e-05, atol=1e-08)


def test_func_polyline4():
    """test line with pos and rot arguments"""
    x = 0.14142136 * 1e-6
    # rotate 1
    rot = R.from_euler("z", 90, degrees=True)
    B3 = current_polyline_field(
        'B',
        observers=[(0, 0, 0)],
        currents=1,
        segments_start=(1, 0, -1),
        segments_end=(1, 0, 1),
        orientations=rot
    )
    expected = np.array([x, 0, 0])

    np.testing.assert_allclose(B3, expected, rtol=1e-05, atol=1e-08)


def test_func_polyline5():
    """test line with pos and rot arguments"""
    x = 0.14142136 * 1e-6
    # rotate 1
    rot = R.from_euler("x", 90, degrees=True)
    B3 = current_polyline_field(
        'B',
        observers=[(0, 0, 0)],
        currents=1,
        segments_start=(1, 0, -1),
        segments_end=(1, 0, 1),
        orientations=rot
    )
    expected = np.array([0, 0, -x])

    np.testing.assert_allclose(B3, expected, rtol=1e-05, atol=1e-08)


def test_func_polyline6():
    """test line with pos and rot arguments"""
    x = 0.14142136 * 1e-6
    # rotate 1
    rot = R.from_euler("y", 90, degrees=True)
    B3 = current_polyline_field(
        'B',
        observers=[(0, 0, 0)],
        currents=1,
        segments_start=(1, 0, -1),
        segments_end=(1, 0, 1),
        orientations=rot
    )
    expected = np.array([0, -x, 0])

    np.testing.assert_allclose(B3, expected, rtol=1e-05, atol=1e-08)
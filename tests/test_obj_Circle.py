import numpy as np
import pytest

import magpylib as magpy
from magpylib._src.exceptions import MagpylibDeprecationWarning


def test_Circle_basic_B():
    """Basic Circle class test"""
    src = magpy.current.Circle(current=123, diameter=2)
    sens = magpy.Sensor(position=(1, 2, 3))

    B = src.getB(sens)
    Btest = np.array([0.44179833, 0.88359665, 0.71546231]) * 1e-6
    np.testing.assert_allclose(B, Btest)


def test_BHJM_circle():
    """test explicit field output values"""
    s = magpy.current.Circle(current=1, diameter=1)

    B_c1d1z0 = 1.2566370614359172 * 1e-6
    B_test = s.getB([0, 0, 0])
    assert abs(B_c1d1z0 - B_test[2]) < 1e-14

    B_c1d1z1 = 0.11239703569665165 * 1e-6
    B_test = s.getB([0, 0, 1])
    assert abs(B_c1d1z1 - B_test[2]) < 1e-14

    s = magpy.current.Circle(current=1, diameter=2)
    B_c1d2z0 = 0.6283185307179586 * 1e-6
    B_test = s.getB([0, 0, 0])
    assert abs(B_c1d2z0 - B_test[2]) < 1e-14

    B_c1d2z1 = 0.22214414690791835 * 1e-6
    B_test = s.getB([0, 0, 1])
    assert abs(B_c1d2z1 - B_test[2]) < 1e-14


def test_Circle_basic_H():
    """Basic Circle class test"""
    src = magpy.current.Circle(current=123, diameter=2)
    sens = magpy.Sensor(position=(1, 2, 3))

    H = src.getH(sens)
    Htest = np.array([0.44179833, 0.88359665, 0.71546231]) * 10 / 4 / np.pi
    np.testing.assert_allclose(H, Htest)


# def test_Circular_problem_positions():
#     """ Circle on z and on loop
#     """
#     src = magpy.current.Circle(current=1, diameter=2)
#     sens = magpy.Sensor()
#     sens.move([[0,1,0],[1,0,0]], start=1)

#     B = src.getB(sens)
#     Btest = np.array([[0,0,0.6283185307179586], [0,0,0], [0,0,0]])
#     np.testing.assert_allclose(B, Btest)


def test_repr():
    """test __repr__"""
    dip = magpy.current.Circle(current=1, diameter=1)
    assert repr(dip)[:6] == "Circle", "Circle repr failed"


def test_old_Loop_deprecation_warning():
    """test old calss deprecation warning"""
    with pytest.warns(MagpylibDeprecationWarning):
        old_class = magpy.current.Loop()

    new_class = magpy.current.Circle()
    assert isinstance(old_class, magpy.current.Circle)
    assert isinstance(new_class, magpy.current.Circle)

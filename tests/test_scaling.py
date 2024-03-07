import numpy as np

import magpylib as magpy


def test_scaling_loop():
    """
    The field of a current loop must satisfy
    B(i0,d,x,y,z) = B(a*i0,a*d,a*x,a*y,a*z)
    """
    c1 = magpy.current.Circle(current=123, diameter=10)
    B1 = c1.getB([1, 2, 3])
    c2 = magpy.current.Circle(current=1230, diameter=100)
    B2 = c2.getB([10, 20, 30])
    c3 = magpy.current.Circle(current=12300, diameter=1000)
    B3 = c3.getB([100, 200, 300])
    c4 = magpy.current.Circle(current=123000, diameter=10000)
    B4 = c4.getB([1000, 2000, 3000])

    np.testing.assert_allclose(B1, B2)
    np.testing.assert_allclose(B1, B3)
    np.testing.assert_allclose(B1, B4)

import numpy as np

import magpylib as magpy


def test_scaling_loop():
    """
    The field of a current loop must satisfy
    B(i0,d,x,y,z) = B(a*i0,a*d,a*x,a*y,a*z)
    """
    c1 = magpy.current.Circle(123, 10)
    B1 = c1.getB([1, 2, 3])
    c2 = magpy.current.Circle(1230, 100)
    B2 = c2.getB([10, 20, 30])
    c3 = magpy.current.Circle(12300, 1000)
    B3 = c3.getB([100, 200, 300])
    c4 = magpy.current.Circle(123000, 10000)
    B4 = c4.getB([1000, 2000, 3000])

    assert np.allclose(B1, B2)
    assert np.allclose(B1, B3)
    assert np.allclose(B1, B4)

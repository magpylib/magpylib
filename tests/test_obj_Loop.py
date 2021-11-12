import numpy as np
import magpylib as magpy


def test_Loop_basic_B():
    """ Basic Loop class test
    """
    src = magpy.current.Loop(current=123, diameter=2)
    sens = magpy.Sensor(position=(1,2,3))

    B = src.getB(sens)
    Btest = np.array([0.44179833, 0.88359665, 0.71546231])
    assert np.allclose(B, Btest)


def test_Loop_basic_H():
    """ Basic Loop class test
    """
    src = magpy.current.Loop(current=123, diameter=2)
    sens = magpy.Sensor(position=(1,2,3))

    H = src.getH(sens)
    Htest = np.array([0.44179833, 0.88359665, 0.71546231])*10/4/np.pi
    assert np.allclose(H, Htest)


def test_Cicular_problem_positions():
    """ Loop on z and on loop
    """
    src = magpy.current.Loop(current=1, diameter=2)
    sens = magpy.Sensor()
    sens.move([[0,1,0],[1,0,0]], start=1)

    B = src.getB(sens)
    Btest = np.array([[0,0,0.6283185307179586], [0,0,0], [0,0,0]])
    assert np.allclose(B, Btest)


def test_repr():
    """ test __repr__
    """
    dip = magpy.current.Loop(current=1, diameter=1)
    assert dip.__repr__()[:4] == 'Loop', 'Loop repr failed'

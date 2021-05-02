import numpy as np
import magpylib as mag3


def test_Circular_basic_B():
    """ Basic Circular class test
    """
    src = mag3.current.Circular(current=123, dim=2)
    sens = mag3.Sensor(pos=(1,2,3))

    B = src.getB(sens)
    Btest = np.array([0.44179833, 0.88359665, 0.71546231])
    assert np.allclose(B, Btest)


def test_Circular_basic_H():
    """ Basic Circular class test
    """
    src = mag3.current.Circular(current=123, dim=2)
    sens = mag3.Sensor(pos=(1,2,3))

    H = src.getH(sens)
    Htest = np.array([0.44179833, 0.88359665, 0.71546231])*10/4/np.pi
    assert np.allclose(H, Htest)


def test_Cicular_problem_positions():
    """ Circular on z and on loop
    """
    src = mag3.current.Circular(current=1, dim=2)
    sens = mag3.Sensor()
    sens.move_to([0,1,0], steps=1)
    sens.move_to([1,0,0], steps=1)

    B = src.getB(sens)
    Btest = np.array([[0,0,0.6283185307179586], [0,0,0], [0,0,0]])
    assert np.allclose(B, Btest)


def test_repr():
    """ test __repr__
    """
    dip = mag3.current.Circular(current=1, dim=1)
    assert dip.__repr__()[:8] == 'Circular', 'Circular repr failed'

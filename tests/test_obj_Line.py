import numpy as np
import magpylib as mag3

def test_Line_basic1():
    """ Basic Line class test
    """
    src = mag3.current.Line(current=100, vertices=[(1,1,-1),(1,1,1)])
    sens = mag3.Sensor()
    B = src.getB(sens)

    x = 5.77350269
    Btest = np.array([x,-x,0])

    assert np.allclose(B,Btest)


def test_Line_basic2():
    """ Basic Line class test 2
    """
    src = mag3.current.Line(current=-100, vertices=[(1,1,-1),(1,1,1)])
    sens = mag3.Sensor()
    H = src.getH(sens)

    x = 5.77350269/4/np.pi*10
    Htest = np.array([-x,x,0])

    assert np.allclose(H,Htest)


def test_Line_basic3():
    """ Basic Line class test 3
    """
    line1 = mag3.current.Line(current=100, vertices=[(1,1,-1),(1,1,1)])
    line2 = mag3.current.Line(current=100, vertices=[(1,1,-1),(1,1,1),(1,1,-1),(1,1,1)])
    sens = mag3.Sensor()
    B = mag3.getB([line1,line2], sens)

    x = 5.77350269
    Btest = np.array([(x,-x,0)]*2)

    assert np.allclose(B,Btest)


def test_Line_repr():
    """ Line repr test
    """
    line = mag3.current.Line(current=100, vertices=[(1,1,-1),(1,1,1)])
    assert line.__repr__()[:4] == 'Line', 'Line repr failed'


def test_Line_specials():
    """ Line specials tests
    """
    line = mag3.current.Line(current=100, vertices=[(0,0,0),(1,1,1)])
    b = line.getB([0,0,0])
    assert np.allclose(b,np.zeros(3))

    line = mag3.current.Line(current=100, vertices=[(0,0,0),(0,0,0)])
    b = line.getB([1,2,3])
    assert np.allclose(b,np.zeros(3))

    line = mag3.current.Line(current=0, vertices=[(1,2,3),(3,2,1)])
    b = line.getB([0,0,0])
    assert np.allclose(b,np.zeros(3))

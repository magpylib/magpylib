import numpy as np
from scipy.spatial.transform import Rotation as R
import magpylib
from magpylib.magnet import Box, Cylinder, Sphere
from magpylib import getBv, getHv, getB, getH


def test_getBv1():
    """test field wrapper functions
    """
    pos_obs = (11,2,2)
    mag = [111,222,333]
    dim = [3,3]

    pm = Cylinder(mag, dim)
    pm.move([(.5,0,0)]*15, increment=True)
    pm.rotate_from_angax(np.linspace(0,666,25), 'y', anchor=0)
    pm.move([(0,x,0) for x in np.linspace(0,5,5)])
    B2 = pm.getB(pos_obs)

    pos = pm.position
    rot = pm.rot

    dic = {
        'src_type': 'Cylinder',
        'pos_obs': pos_obs,
        'magnetization': mag,
        'dimension': dim,
        'position': pos,
        'rot':rot
        }
    B1 = getBv(**dic)

    assert np.allclose(B1, B2, rtol=1e-12, atol=1e-12)


def test_getBv2():
    """test field wrapper functions
    """
    pos_obs = (11,2,2)
    mag = [111,222,333]
    dim = [3,3]
    pos = [(1,1,1),(2,2,2),(3,3,3),(5,5,5)]

    dic = {
        'src_type': 'Cylinder',
        'pos_obs': pos_obs,
        'magnetization': mag,
        'dimension': dim,
        'position': pos
        }
    B1 = getBv(**dic)

    pm = Cylinder(mag, dim, position=pos)
    B2 = getB([pm],pos_obs)

    assert np.allclose(B1, B2, rtol=1e-12, atol=1e-12)


def test_getHv1():
    """test field wrapper functions
    """
    pos_obs = (11,2,2)
    mag = [111,222,333]
    dim = [3,3]

    dic = {
        'src_type': 'Cylinder',
        'pos_obs': pos_obs,
        'magnetization': mag,
        'dimension': dim,
        }
    B1 = getHv(**dic)

    pm = Cylinder(mag, dim)
    B2 = pm.getH(pos_obs)

    assert np.allclose(B1, B2, rtol=1e-12, atol=1e-12)


def test_getHv2():
    """test field wrapper functions
    """
    pos_obs = (1,2,2)
    mag = [[111,222,333],[22,2,2],[22,-33,-44]]
    dim = [3,3]

    magpylib.Config.ITER_CYLINDER=75
    dic = {
        'src_type': 'Cylinder',
        'pos_obs': pos_obs,
        'magnetization': mag,
        'dimension': dim
        }
    B1 = getHv(**dic)

    B2 = []
    for i in range(3):
        pm = Cylinder(mag[i],dim)
        B2 += [getH([pm], pos_obs)]
    B2 = np.array(B2)

    assert np.allclose(B1, B2, rtol=1e-12, atol=1e-12)


def test_getBv3():
    """test field wrapper functions
    """
    n = 25
    pos_obs = np.array([1,2,2])
    mag = [[111,222,333],]*n
    dim = [3,3,3]
    pos = np.array([0,0,0])
    rot = R.from_quat([(t,.2,.3,.4) for t in np.linspace(0,.1,n)])

    dic = {
        'src_type': 'Box',
        'pos_obs': pos_obs,
        'magnetization': mag,
        'dimension': dim,
        'position': pos,
        'rot': rot
        }
    B1 = getBv(**dic)

    B2 = []
    for i in range(n):
        pm = Box(mag[i],dim,pos,rot[i])
        B2 += [pm.getB(pos_obs)]
    B2 = np.array(B2)
    print(B1-B2)
    assert np.allclose(B1, B2, rtol=1e-12, atol=1e-12)


def test_getHv3():
    """test field wrapper functions
    """
    pos_obs = (1,2,2)
    mag = [[111,222,333],[22,2,2],[22,-33,-44]]
    dim = 3

    dic = {
        'src_type': 'Sphere',
        'pos_obs': pos_obs,
        'magnetization': mag,
        'dimension': dim
        }
    B1 = getHv(**dic)

    B2 = []
    for i in range(3):
        pm = Sphere(mag[i],dim)
        B2 += [getH([pm], pos_obs)]
    B2 = np.array(B2)

    assert np.allclose(B1, B2, rtol=1e-12, atol=1e-12)


def test_getBv4():
    """test field wrapper functions
    """
    n = 25
    pos_obs = np.array([1,2,2])
    mag = [[111,222,333],]*n
    dim = 3
    pos = np.array([0,0,0])
    rot = R.from_quat([(t,.2,.3,.4) for t in np.linspace(0,.1,n)])

    dic = {
        'src_type': 'Sphere',
        'pos_obs': pos_obs,
        'magnetization': mag,
        'dimension': dim,
        'position': pos,
        'rot': rot
        }
    B1 = getBv(**dic)

    B2 = []
    for i in range(n):
        pm = Sphere(mag[i],dim,pos,rot[i])
        B2 += [pm.getB(pos_obs)]
    B2 = np.array(B2)
    print(B1-B2)
    assert np.allclose(B1, B2, rtol=1e-12, atol=1e-12)


def test_geBHv_dipole():
    """ test if Dipole implementation gives correct output
    """
    B = getBv(src_type='Dipole', moment=(1,2,3), pos_obs = (1,1,1))
    Btest = np.array([0.07657346,0.06125877,0.04594407])
    assert np.allclose(B,Btest)

    H = getHv(src_type='Dipole', moment=(1,2,3), pos_obs = (1,1,1))
    Htest = np.array([0.06093522,0.04874818,0.03656113])
    assert np.allclose(H,Htest)


def test_geBHv_circular():
    """ test if Circular implementation gives correct output
    """
    B = getBv(src_type='Circular', current=1, dimension=2, pos_obs = (0,0,0))
    Btest = np.array([0,0,0.6283185307179586])
    assert np.allclose(B,Btest)

    H = getHv(src_type='Circular', current=1, dimension=2, pos_obs = (0,0,0))
    Htest = np.array([0,0,0.6283185307179586*10/4/np.pi])
    assert np.allclose(H,Htest)


def test_getBHv_squeeze():
    """ test if squeeze works
    """
    B1 = getBv(src_type='Circular', current=1, dimension=2, pos_obs = (0,0,0))
    B2 = getBv(src_type='Circular', current=1, dimension=2, pos_obs = [(0,0,0)])
    B3 = getBv(src_type='Circular', current=1, dimension=2, pos_obs = [(0,0,0)], squeeze=False)
    B4 = getBv(src_type='Circular', current=1, dimension=2, pos_obs = [(0,0,0)]*2)

    assert B1.ndim == 1
    assert B2.ndim == 1
    assert B3.ndim == 2
    assert B4.ndim == 2


def test_getBHv_line():
    """ test getBHv with Line
    """
    H = getHv(
        src_type='Line',
        pos_obs=[(1,1,1),(1,2,3),(2,2,2)],
        current=1,
        pos_start=(0,0,0),
        pos_end=[(0,0,0),(2,2,2),(2,2,2)])
    x = np.array([[0,0,0], [0.02672612, -0.05345225, 0.02672612], [0,0,0]])*10/4/np.pi
    assert np.allclose(x, H)


def test_getBHv_line2():
    """ test line with pos and rot arguments
    """
    x = 0.14142136

    # z-line on x=1
    B1 = getBv(src_type='Line', pos_obs=(0,0,0), current=1,
        pos_start=(1,0,-1), pos_end=(1,0,1))
    assert np.allclose(B1, np.array([0,-x,0]))

    # move z-line to x=-1
    B2 = getBv(src_type='Line', position=(-2,0,0), pos_obs=(0,0,0), current=1,
        pos_start=(1,0,-1), pos_end=(1,0,1))
    assert np.allclose(B2, np.array([0,x,0]))

    # rotate 1
    rot = R.from_euler('z', 90, degrees=True)
    B3 = getBv(src_type='Line', rot = rot, pos_obs=(0,0,0), current=1,
        pos_start=(1,0,-1), pos_end=(1,0,1))
    assert np.allclose(B3, np.array([x,0,0]))

    # rotate 2
    rot = R.from_euler('x', 90, degrees=True)
    B4 = getBv(src_type='Line', rot = rot, pos_obs=(0,0,0), current=1,
        pos_start=(1,0,-1), pos_end=(1,0,1))
    assert np.allclose(B4, np.array([0,0,-x]))

    # rotate 3
    rot = R.from_euler('y', 90, degrees=True)
    B5 = getBv(src_type='Line', rot = rot, pos_obs=(0,0,0), current=1,
        pos_start=(1,0,-1), pos_end=(1,0,1))
    assert np.allclose(B5, np.array([0,-x,0]))

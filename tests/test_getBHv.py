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
    pm.move_to((5,0,0),steps=15)
    pm.rotate_from_angax(666,'y',anchor=0,steps=25)
    pm.move_by((0,10,0),steps=-20)
    B2 = pm.getB(pos_obs)

    pos = pm.pos
    rot = pm.rot

    dic = {
        'src_type': 'Cylinder',
        'pos_obs': pos_obs,
        'mag': mag,
        'dim': dim,
        'pos': pos,
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
        'mag': mag,
        'dim': dim,
        'pos': pos
        }
    B1 = getBv(**dic)

    pm = Cylinder(mag, dim, pos=pos)
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
        'mag': mag,
        'dim': dim,
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

    dic = {
        'src_type': 'Cylinder',
        'pos_obs': pos_obs,
        'mag': mag,
        'dim': dim,
        'niter': 75
        }
    B1 = getHv(**dic)

    magpylib.Config.ITER_CYLINDER=75
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
        'mag': mag,
        'dim': dim,
        'pos': pos,
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
        'mag': mag,
        'dim': dim,
        'niter': 75
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
        'mag': mag,
        'dim': dim,
        'pos': pos,
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

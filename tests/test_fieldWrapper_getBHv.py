import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib3.magnet import Box, Cylinder
from magpylib3 import getBv, getHv, getB, getH


def test_fieldWrapper_getBv1():
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

    dict = {
        'src_type': 'Cylinder',
        'pos_obs': pos_obs,
        'mag': mag,
        'dim': dim,
        'pos': pos,
        'rot':rot
        }
    B1 = getBv(**dict)

    assert np.allclose(B1, B2, rtol=1e-12, atol=1e-12)


def test_fieldWrapper_getBv2():
    pos_obs = (11,2,2)
    mag = [111,222,333]
    dim = [3,3]
    pos = [(1,1,1),(2,2,2),(3,3,3),(5,5,5)]

    dict = {
        'src_type': 'Cylinder',
        'pos_obs': pos_obs,
        'mag': mag,
        'dim': dim,
        'pos': pos
        }
    B1 = getBv(**dict)

    pm = Cylinder(mag, dim, pos=pos)
    B2 = getB([pm],pos_obs)

    assert np.allclose(B1, B2, rtol=1e-12, atol=1e-12)


def test_fieldWrapper_getHv1():
    pos_obs = (11,2,2)
    mag = [111,222,333]
    dim = [3,3]

    dict = {
        'src_type': 'Cylinder',
        'pos_obs': pos_obs,
        'mag': mag,
        'dim': dim,
        }
    B1 = getHv(**dict)

    pm = Cylinder(mag, dim)
    B2 = pm.getH(pos_obs)

    assert np.allclose(B1, B2, rtol=1e-12, atol=1e-12)


def test_fieldWrapper_getHv2():
    pos_obs = (1,2,2)
    mag = [[111,222,333],[22,2,2],[22,-33,-44]]
    dim = [3,3]

    dict = {
        'src_type': 'Cylinder',
        'pos_obs': pos_obs,
        'mag': mag,
        'dim': dim,
        'niter': 75
        }
    B1 = getHv(**dict)

    B2 = []
    for i in range(3):
        pm = Cylinder(mag[i],dim)
        B2 += [getH([pm], pos_obs, niter=75)]
    B2 = np.array(B2)

    assert np.allclose(B1, B2, rtol=1e-12, atol=1e-12)


def test_fieldWrapper_getBv3():
    n = 25
    pos_obs = np.array([1,2,2])
    mag = [[111,222,333],]*n
    dim = [3,3,3]
    pos = np.array([0,0,0])
    rot = R.from_quat([(t,.2,.3,.4) for t in np.linspace(0,.1,n)])

    dict = {
        'src_type': 'Box',
        'pos_obs': pos_obs,
        'mag': mag,
        'dim': dim,
        'pos': pos,
        'rot': rot
        }
    B1 = getBv(**dict)

    B2 = []
    for i in range(n):
        pm = Box(mag[i],dim,pos,rot[i])
        B2 += [pm.getB(pos_obs)]
    B2 = np.array(B2)
    print(B1-B2)
    assert np.allclose(B1, B2, rtol=1e-12, atol=1e-12)
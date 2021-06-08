import os
import pickle
import numpy as np
import magpylib as mag3
from magpylib.magnet import Cylinder

# # GENERATE DATA
# N = 22

# mags = (np.random.rand(N,3)-0.5)*1000
# dims = (np.random.rand(N,2)-0.5)*5
# posos = (np.random.rand(N,77,3)-0.5)*10 #readout at 333 positions

# angs =  (np.random.rand(N,18)-0.5)*2*10 # each step rote by max 10 deg
# axs =   (np.random.rand(N,18,3)-0.5)
# anchs = (np.random.rand(N,18,3)-0.5)*5.5
# movs =  (np.random.rand(N,18,3)-0.5)*0.5

# B = []
# for mag,dim,ang,ax,anch,mov,poso in zip(mags,dims,angs,axs,anchs,movs,posos):
#     pm = Cylinder(mag,dim)
#     # 18 subsequent operations
#     for a,aa,aaa,mv in zip(ang,ax,anch,mov):
#         pm.move(mv).rotate_from_angax(a,aa,aaa)
#     B += [pm.getB(poso,niter=100)]
# B = np.array(B)
# inp = [mags,dims,posos,angs,axs,anchs,movs,B]
# pickle.dump(inp,open('testdata_Cylinder.p', 'wb'))


def test_Cylinder_basics():
    """  test Cylinder fundamentals, test against magpylib2 fields
    """
    data = pickle.load(open(os.path.abspath('./tests/testdata/testdata_Cylinder.p'), 'rb'))
    mags,dims,posos,angs,axs,anchs,movs,B = data

    Btest = []
    mag3.Config.ITER_CYLINDER = 100
    for mag,dim,ang,ax,anch,mov,poso in zip(mags,dims,angs,axs,anchs,movs,posos):
        pm = Cylinder(mag,dim)

        # 18 subsequent operations
        for a,aa,aaa,mv in zip(ang,ax,anch,mov):
            pm.move(mv).rotate_from_angax(a,aa,aaa)

        Btest += [pm.getB(poso)]
    Btest = np.array(Btest)

    assert np.allclose(B, Btest), "test_Cylinder failed big time"


def test_Cylinder_add():
    """ testing __add__
    """
    src1 = Cylinder((1,2,3),(1,2))
    src2 = Cylinder((1,2,3),(1,2))
    col = src1 + src2
    assert isinstance(col,mag3.Collection), 'adding cylinder fail'


def test_Cylinder_squeeze():
    """ testing squeeze output
    """
    src1 = Cylinder((1,1,1),(1,1))
    sensor = mag3.Sensor(pixel=[(1,2,3),(1,2,3)])
    B = src1.getB(sensor)
    assert B.shape==(2,3)
    H = src1.getH(sensor)
    assert H.shape==(2,3)

    B = src1.getB(sensor,squeeze=False)
    assert B.shape==(1,1,1,2,3)
    H = src1.getH(sensor,squeeze=False)
    assert H.shape==(1,1,1,2,3)


def test_repr():
    """ test __repr__
    """
    pm2 = mag3.magnet.Cylinder((1,2,3),(2,3))
    assert pm2.__repr__()[:8] == 'Cylinder', 'Cylinder repr failed'

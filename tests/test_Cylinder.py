import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import magpylib
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
#         pm.move_by(mv).rotate_from_angax(a,aa,aaa)
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
    for mag,dim,ang,ax,anch,mov,poso in zip(mags,dims,angs,axs,anchs,movs,posos):
        pm = Cylinder(mag,dim)

        # 18 subsequent operations
        for a,aa,aaa,mv in zip(ang,ax,anch,mov):
            pm.move_by(mv).rotate_from_angax(a,aa,aaa)

        Btest += [pm.getB(poso, niter=100)]
    Btest = np.array(Btest)

    assert np.allclose(B, Btest), "test_Cylinder failed big time"


def test_Cylinder_display():
    """ testing display
    """
    fig = plt.figure(figsize=(8, 8),facecolor='w', dpi=100)
    ax = fig.gca(projection='3d')
    src = Cylinder((1,2,3),(1,2))
    x = src.display(axis=ax,show_path='all')
    assert x is None, 'display test fail'


def test_Cylinder_add():
    """ testing __add__
    """
    src1 = Cylinder((1,2,3),(1,2))
    src2 = Cylinder((1,2,3),(1,2))
    col = src1 + src2
    assert isinstance(col,magpylib.Collection), 'adding cylinder fail'

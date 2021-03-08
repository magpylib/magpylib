import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from magpylib.magnet import Box

# """data generation for test_Box()"""

# N = 100

# mags = (np.random.rand(N,3)-0.5)*1000
# dims = (np.random.rand(N,3)-0.5)*5
# posos = (np.random.rand(N,333,3)-0.5)*10 #readout at 333 positions

# angs =  (np.random.rand(N,18)-0.5)*2*10 # each step rote by max 10 deg
# axs =   (np.random.rand(N,18,3)-0.5)
# anchs = (np.random.rand(N,18,3)-0.5)*5.5
# movs =  (np.random.rand(N,18,3)-0.5)*0.5

# B = []
# for mag,dim,ang,ax,anch,mov,poso in zip(mags,dims,angs,axs,anchs,movs,posos):
#     pm = Box(mag,dim)

#     # 18 subsequent operations
#     for a,aa,aaa,mv in zip(ang,ax,anch,mov):
#         pm.move_by(mv).rotate_from_angax(a,aa,aaa)

#     B += [pm.getB(poso)]
# B = np.array(B)

# inp = [mags,dims,posos,angs,axs,anchs,movs,B]

# pickle.dump(inp,open(os.path.abspath('./../'testdata_Box.p', 'wb'))


def test_Box():
    """box test
    """
    # data generated below
    data = pickle.load(open(os.path.abspath('./tests/testdata/testdata_Box.p'), 'rb'))
    mags,dims,posos,angs,axs,anchs,movs,B = data

    btest = []
    for mag,dim,ang,ax,anch,mov,poso in zip(mags,dims,angs,axs,anchs,movs,posos):
        pm = Box(mag,dim)

        # 18 subsequent operations
        for a,aa,aaa,mv in zip(ang,ax,anch,mov):
            pm.move_by(mv).rotate_from_angax(a,aa,aaa)

        btest += [pm.getB(poso)]
    btest = np.array(btest)

    assert np.allclose(B, btest), "test_Box failed big time"


def test_Box_display():
    """ testing display
    """
    fig = plt.figure(figsize=(8, 8),facecolor='w', dpi=100)
    ax = fig.gca(projection='3d')
    src = Box((1,2,3),(1,2,3))
    x = src.display(axis=ax,show_path=False,direc=True)
    assert x is None, 'display test fail'

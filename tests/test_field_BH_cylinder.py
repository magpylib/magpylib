import pickle
import os
import numpy as np
from magpylib._lib.fields.field_BH_cylinder import field_BH_cylinder
from magpylib import Config

# # GENERATE TEST DATA
# n = 500
# mag3.Config.EDGESIZE = 1e-14
# # dim general
# dim_gen = np.random.rand(n,2)
# # dim edge
# dim_edge = np.array([(2,2)]*n)

# # pure edge positions
# x = np.random.rand(n)*2*np.pi
# xs = 1e-14*(np.random.rand(n)-.5) + np.sin(x)
# ys = 1e-14*(np.random.rand(n)-.5) + np.cos(x)
# zs = 1e-14*(np.random.rand(n)-.5) + 1
# zs[500:] -= 2
# pos_edge = np.c_[xs,ys,zs]
# # general positions
# pos_gen = (np.random.rand(n,3)-0.5)*.5 # half inner half outer
# # mixed positions
# aa = np.r_[pos_edge, pos_gen]
# np.random.shuffle(aa)
# pos_mix = aa[:n]

# # general mag
# mag_gen = (np.random.rand(n,3)-0.5)*1000
# # pure axial mag
# mag_ax = (np.random.rand(n,3)-0.5)*1000
# mag_ax[:,0] = 0
# mag_ax[:,1] = 0
# # pure diametral mag
# mag_tv = (np.random.rand(n,3)-0.5)*1000
# mag_tv[:,2] = 0
# # mixed mag
# mm = np.r_[mag_gen, mag_ax, mag_tv]
# np.random.shuffle(mm)
# mag_mix = mm[:n]

# mags = [mag_gen, mag_ax, mag_tv, mag_mix]
# poss = [pos_edge, pos_gen, pos_mix]
# dims = [dim_edge, dim_gen, dim_edge]
# Bs = []
# for mag in mags:
#     for dim,pos in zip(dims,poss):
#         Bs += [field_BH_cylinder(True, mag, dim, pos, 100)]
# Bs = np.array(Bs)
# pickle.dump([mags, dims,poss,Bs], open('testdata_field_BH_cylinder.p','wb'))

def test_field_BH_cylinder():
    """ test cylinder field
    """
    Config.EDGESIZE=1e-14
    mags, dims, poss, B = pickle.load(open(
        os.path.abspath('tests/testdata/testdata_field_BH_cylinder.p'),'rb'))
    Btest = []
    for mag in mags:
        for dim,pos in zip(dims,poss):
            Btest += [field_BH_cylinder(True, mag, dim, pos, 100)]
    Btest = np.array(Btest)
    assert np.allclose(B, Btest), 'Cylinder field computation broken'


def test_field_BH_cylinder_mag0():
    """test cylinder field mag=0
    """
    n = 10
    mag = np.zeros((n,3))
    dim = np.random.rand(n,2)
    pos = np.random.rand(n,3)
    B = field_BH_cylinder(True, mag, dim, pos, 11)
    assert np.allclose(mag,B), 'Cylinder mag=0 case broken'

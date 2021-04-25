import pickle
import os
import numpy as np
from magpylib._lib.fields.field_BH_box import field_BH_box
from magpylib._lib.fields.field_BH_cylinder import field_BH_cylinder
from magpylib._lib.fields.field_BH_sphere import field_BH_sphere
from magpylib._lib.fields.field_BH_dipole import field_BH_dipole
from magpylib import Config

# # GENERATE TEST DATA
# n = 500
# mag3.Config.EDGESIZE = 1e-14

# # dim general
# dim_gen = np.random.rand(n,3)
# # dim edge
# dim_edge = np.array([(2,2,2)]*n)

# # pure edge positions
# pos_edge = (np.random.rand(n,3)-0.5)*2
# pos_edge[:,0]=1 + (np.random.rand(n)-0.5)*1e-14
# pos_edge[:,1]=1  + (np.random.rand(n)-0.5)*1e-14
# for i in range(n):
#     np.random.shuffle(pos_edge[i])
# # general positions
# pos_gen = (np.random.rand(n,3)-0.5)*.5 # half inner half outer
# # mixed positions
# aa = np.r_[pos_edge, pos_gen]
# np.random.shuffle(aa)
# pos_mix = aa[:n]

# # general mag (no special cases at this point)
# mag = (np.random.rand(n,3)-0.5)*1000

# poss = [pos_edge, pos_gen, pos_mix]
# dims = [dim_edge, dim_gen, dim_edge]

# Bs = []
# for dim,pos in zip(dims,poss):
#     Bs += [field_BH_box(True, mag, dim, pos)]
# Bs = np.array(Bs)

# print(Bs)
# pickle.dump([mag, dims, poss, Bs], open('testdata_field_BH_box.p','wb'))


def test_field_BH_box():
    """ test box field
    """
    Config.EDGESIZE=1e-14
    mag, dims, poss, B = pickle.load(open(
        os.path.abspath('tests/testdata/testdata_field_BH_box.p') ,'rb'))
    Btest = []
    for dim,pos in zip(dims,poss):
        Btest += [field_BH_box(True, mag, dim, pos)]
    Btest = np.array(Btest)
    assert np.allclose(B, Btest), 'Box field computation broken'


def test_field_BH_box_mag0():
    """ test box field mag=0
    """
    n = 10
    mag = np.zeros((n,3))
    dim = np.random.rand(n,3)
    pos = np.random.rand(n,3)
    B = field_BH_box(True, mag, dim, pos)
    assert np.allclose(mag,B), 'Box mag=0 case broken'


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
    Config.EDGESIZE = 1e-14
    Config.ITER_CYLINDER = 100
    mags, dims, poss, B = pickle.load(open(
        os.path.abspath('tests/testdata/testdata_field_BH_cylinder.p'),'rb'))
    Btest = []
    for mag in mags:
        for dim,pos in zip(dims,poss):
            Btest += [field_BH_cylinder(True, mag, dim, pos)]
    Btest = np.array(Btest)
    assert np.allclose(B, Btest), 'Cylinder field computation broken'


def test_field_BH_cylinder_mag0():
    """test cylinder field mag=0
    """
    n = 10
    mag = np.zeros((n,3))
    dim = np.random.rand(n,2)
    pos = np.random.rand(n,3)
    B = field_BH_cylinder(True, mag, dim, pos)
    assert np.allclose(mag,B), 'Cylinder mag=0 case broken'


def test_field_sphere_vs_v2():
    """ testing against old version
    """
    result_v2 = np.array([
        [22., 44., 66.],
        [22., 44., 66.],
        [38.47035383, 30.77628307, 23.0822123 ],
        [0.60933932, 0.43524237, 1.04458169],
        [22., 44., 66.],
        [-0.09071337, -0.18142674, -0.02093385],
        [-0.17444878, -0.0139559,  -0.10466927],
        ])

    dim = np.array([1.23]*7)
    mag = np.array([(33,66,99)]*7)
    poso = np.array([(0,0,0),(.2,.2,.2),(.4,.4,.4),(-1,-1,-2),(.1,.1,.1),(1,2,-3),(-3,2,1)])
    B = field_BH_sphere(True, mag, dim, poso )

    assert np.allclose(result_v2, B), 'vs_v2 failed'


def test_field_BH_sphere_mag0():
    """ test box field mag=0
    """
    n = 10
    mag = np.zeros((n,3))
    dim = np.random.rand(n)
    pos = np.random.rand(n,3)
    B = field_BH_sphere(True, mag, dim, pos)
    assert np.allclose(mag,B), 'Box mag=0 case broken'


def test_field_dipole1():
    """ Test standard dipole field output computed with mathematica
    """
    poso = np.array([(1,2,3),(-1,2,3)])
    mom = np.array([(2,3,4),(0,-3,-2)])
    B = field_BH_dipole(True, mom, poso)*np.pi
    Btest = np.array([
        (0.01090862,0.02658977,0.04227091),
        (0.0122722,-0.01022683,-0.02727156),
        ])

    assert np.allclose(B,Btest)


def test_field_dipole2():
    """ test nan return when pos_obs=0
    """
    poso = np.array([(0,0,0)])
    mom = np.array([(-1,2,3)])
    np.seterr(all='ignore')
    B = field_BH_dipole(True, mom, poso)*np.pi
    np.seterr(all='warn')

    assert all(np.isnan(B[0]))

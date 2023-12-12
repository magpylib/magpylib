import os
import pickle

import numpy as np

import magpylib as magpy

# # """data generation for test_Sphere()"""

# # N = 100

# # mags = (np.random.rand(N,3)-0.5)*1000
# # dims = np.random.rand(N)*5
# # posos = (np.random.rand(N,333,3)-0.5)*10 #readout at 333 positions

# # angs =  (np.random.rand(N,18)-0.5)*2*10 # each step rote by max 10 deg
# # axs =   (np.random.rand(N,18,3)-0.5)
# # anchs = (np.random.rand(N,18,3)-0.5)*5.5
# # movs =  (np.random.rand(N,18,3)-0.5)*0.5

# # B = []
# # for mag,dim,ang,ax,anch,mov,poso in zip(mags,dims,angs,axs,anchs,movs,posos):
# #     pm = magpy.magnet.Sphere(mag,dim)

# #     # 18 subsequent operations
# #     for a,aa,aaa,mv in zip(ang,ax,anch,mov):
# #         pm.move(mv).rotate_from_angax(a,aa,aaa)

# #     B += [pm.getB(poso)]
# # B = np.array(B)

# # inp = [mags,dims,posos,angs,axs,anchs,movs,B]

# # pickle.dump(inp, open('testdata_Sphere.p', 'wb'))


def test_Sphere_basics():
    """test Cuboid fundamentals, test against magpylib2 fields"""
    # data generated below
    with open(os.path.abspath("./tests/testdata/testdata_Sphere.p"), "rb") as f:
        data = pickle.load(f)
    mags, dims, posos, angs, axs, anchs, movs, B = data

    btest = []
    for mag, dim, ang, ax, anch, mov, poso in zip(
        mags, dims, angs, axs, anchs, movs, posos
    ):
        pm = magpy.magnet.Sphere(mag, dim)

        # 18 subsequent operations
        for a, aa, aaa, mv in zip(ang, ax, anch, mov):
            pm.move(mv).rotate_from_angax(a, aa, aaa, start=-1)

        btest += [pm.getB(poso)]
    btest = np.array(btest)

    assert np.allclose(B, btest), "test_Sphere failed big time"


def test_Sphere_add():
    """testing __add__"""
    src1 = magpy.magnet.Sphere(magnetization=(1, 2, 3), diameter=11)
    src2 = magpy.magnet.Sphere((1, 2, 3), 11)
    col = src1 + src2
    assert isinstance(col, magpy.Collection), "adding cuboids fail"


def test_Sphere_squeeze():
    """testing squeeze output"""
    src1 = magpy.magnet.Sphere((1, 1, 1), 1)
    sensor = magpy.Sensor(pixel=[(1, 2, 3), (1, 2, 3)])
    B = src1.getB(sensor)
    assert B.shape == (2, 3)
    H = src1.getH(sensor)
    assert H.shape == (2, 3)

    B = src1.getB(sensor, squeeze=False)
    assert B.shape == (1, 1, 1, 2, 3)
    H = src1.getH(sensor, squeeze=False)
    assert H.shape == (1, 1, 1, 2, 3)


def test_repr():
    """test __repr__"""
    pm3 = magpy.magnet.Sphere((1, 2, 3), 3)
    assert repr(pm3)[:6] == "Sphere", "Sphere repr failed"


def test_sphere_object_vs_lib():
    """
    tests object vs lib computation
    this also checks if np.int (from array slice) is allowed as input
    """
    mag = np.array([(10, 20, 30)])
    dia = np.array([1])
    pos = np.array([(2, 2, 2)])
    B1 = magpy.core.magnet_sphere_field("B", pos, mag, dia)[0]

    src = magpy.magnet.Sphere(mag[0], dia[0])
    B2 = src.getB(pos)

    np.testing.assert_allclose(B1, B2)

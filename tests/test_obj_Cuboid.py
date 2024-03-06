import os
import pickle

import numpy as np

import magpylib as magpy
from magpylib._src.fields.field_BH_cuboid import BHJM_magnet_cuboid
from magpylib._src.obj_classes.class_Sensor import Sensor

# # # """data generation for test_Cuboid()"""

# # N = 100

# # mags  = (np.random.rand(N,3)-0.5)*1000
# # dims  =  np.random.rand(N,3)*5
# # posos = (np.random.rand(N,333,3)-0.5)*10 #readout at 333 positions

# # angs  =  (np.random.rand(N,18)-0.5)*2*10 # each step rote by max 10 deg
# # axs   =   (np.random.rand(N,18,3)-0.5)
# # anchs = (np.random.rand(N,18,3)-0.5)*5.5
# # movs  =  (np.random.rand(N,18,3)-0.5)*0.5

# # B = []
# # for mag,dim,ang,ax,anch,mov,poso in zip(mags,dims,angs,axs,anchs,movs,posos):
# #     pm = magpy.magnet.Cuboid(mag,dim)

# #     # 18 subsequent operations
# #     for a,aa,aaa,mv in zip(ang,ax,anch,mov):
# #         pm.move(mv).rotate_from_angax(a,aa,aaa)

# #     B += [pm.getB(poso)]
# # B = np.array(B)

# # inp = [mags,dims,posos,angs,axs,anchs,movs,B]

# # pickle.dump(inp, open(os.path.abspath('testdata_Cuboid.p'), 'wb'))


def test_Cuboid_basics():
    """test Cuboid fundamentals"""
    # data generated in comment above
    with open(os.path.abspath("./tests/testdata/testdata_Cuboid.p"), "rb") as f:
        data = pickle.load(f)
    mags, dims, posos, angs, axs, anchs, movs, B = data

    btest = []
    for mag, dim, ang, ax, anch, mov, poso in zip(
        mags, dims, angs, axs, anchs, movs, posos
    ):
        pm = magpy.magnet.Cuboid(polarization=mag, dimension=np.abs(dim))

        # 18 subsequent operations
        for a, aa, aaa, mv in zip(ang, ax, anch, mov):
            pm.move(mv).rotate_from_angax(a, aa, aaa, start=-1)

        btest += [pm.getB(poso)]
    btest = np.array(btest)

    np.testing.assert_allclose(B, btest)


def test_Cuboid_add():
    """testing __add__"""
    src1 = magpy.magnet.Cuboid(polarization=(1, 2, 3), dimension=(1, 2, 3))
    src2 = magpy.magnet.Cuboid(polarization=(1, 2, 3), dimension=(1, 2, 3))
    col = src1 + src2
    assert isinstance(col, magpy.Collection), "adding cuboides fail"


def test_Cuboid_squeeze():
    """testing squeeze output"""
    src1 = magpy.magnet.Cuboid(polarization=(1, 1, 1), dimension=(1, 1, 1))
    sensor = Sensor(pixel=[(1, 2, 3), (1, 2, 3)])
    B = src1.getB(sensor)
    assert B.shape == (2, 3)
    H = src1.getH(sensor)
    assert H.shape == (2, 3)

    B = src1.getB(sensor, squeeze=False)
    assert B.shape == (1, 1, 1, 2, 3)
    H = src1.getH(sensor, squeeze=False)
    assert H.shape == (1, 1, 1, 2, 3)


def test_repr_cuboid():
    """test __repr__"""
    pm1 = magpy.magnet.Cuboid(polarization=(1, 2, 3), dimension=(1, 2, 3))
    pm1.style.label = "cuboid_01"
    assert repr(pm1)[:6] == "Cuboid", "Cuboid repr failed"
    assert "label='cuboid_01'" in repr(pm1), "Cuboid repr failed"


def test_cuboid_object_vs_lib():
    """
    includes a test of the input copy problem
    """

    a = 1
    pol = np.array([(10, 20, 30)])
    dim = np.array([(a, a, a)])
    pos = np.array([(2 * a, 2 * a, 2 * a)])
    B0 = BHJM_magnet_cuboid(field="B", observers=pos, polarization=pol, dimension=dim)
    H0 = BHJM_magnet_cuboid(field="H", observers=pos, polarization=pol, dimension=dim)

    src = magpy.magnet.Cuboid(polarization=pol[0], dimension=dim[0])
    B1 = src.getB(pos)
    H1 = src.getH(pos)

    np.testing.assert_allclose(B0[0], B1)
    np.testing.assert_allclose(H0[0], H1)


def test_getM():
    """getM test"""
    m0 = (0, 0, 0)
    m1 = (10, 200, 3000)
    cube = magpy.magnet.Cuboid(dimension=(2, 2, 2), magnetization=m1)
    obs = [
        (2, 2, 2),
        (0, 0, 0),
        (0.5, 0.5, 0.5),
        (3, 0, 0),
    ]
    sens = magpy.Sensor(pixel=obs)

    M1 = cube.getM(obs)
    M2 = magpy.getM(cube, sens)
    M3 = sens.getM(cube)
    Mtest = np.array([m0, m1, m1, m0])

    np.testing.assert_allclose(M1, Mtest)
    np.testing.assert_allclose(M2, Mtest)
    np.testing.assert_allclose(M3, Mtest)


def test_getJ():
    """getM test"""
    j0 = (0, 0, 0)
    j1 = (0.1, 0.2, 0.3)
    cube = magpy.magnet.Cuboid(
        dimension=(2, 2, 2),
        polarization=j1,
    )
    obs = [
        (-2, 2, -2),
        (0, 0, 0),
        (-0.5, -0.5, 0.5),
        (-3, 0, 0),
    ]
    sens = magpy.Sensor(pixel=obs)

    J1 = cube.getJ(obs)
    J2 = magpy.getJ(cube, sens)
    J3 = sens.getJ(cube)

    Jtest = np.array([j0, j1, j1, j0])
    np.testing.assert_allclose(J1, Jtest)
    np.testing.assert_allclose(J2, Jtest)
    np.testing.assert_allclose(J3, Jtest)

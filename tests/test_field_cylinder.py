"""
Testing all cases against a large set of pre-computed values
"""

import numpy as np
from magpylib._lib.fields.field_BH_cylinder_tile import (field_H_cylinder_tile,
    field_BH_cylinder_tile)
import magpylib as magpy


# creating test data

# from florian_run_analytic_paper_final import H_total_final
# N=1111
# null = np.zeros(N)
# R = np.random.rand(N)*10
# R1,R2 = np.random.rand(2,N)*5
# R2 = R1+R2
# PHI,PHI1,PHI2 = (np.random.rand(3,N)-.5)*10
# PHI2 = PHI1+PHI2
# Z,Z1,Z2 = (np.random.rand(3,N)-.5)*10
# Z2 = Z1+Z2
# mag = np.random.rand(N,3)

# DATA = []

# # cases [112, 212, 132, 232]
# r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
# z1=z
# phi1 = phi
# r = null
# obs_pos = np.array([r, phi, z]).T
# dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
# H1 = field_H_cylinder_tile(obs_pos, dim, mag)
# H2 = H_total_final(obs_pos, dim, mag)
# assert np.allclose(H1,H2)

# DATA += [H2]

# # cases [122, 222, 132, 232]
# r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
# z1=z
# phi1 = phi+np.pi
# r = null
# obs_pos = np.array([r, phi, z]).T
# dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
# H1 = field_H_cylinder_tile(obs_pos, dim, mag)
# H2 = H_total_final(obs_pos, dim, mag)
# assert np.allclose(H1,H2)

# DATA += [H2]

# # cases [113, 213, 133, 233, 115, 215, 135, 235]
# r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
# z1 = z
# phi1 = phi
# r1 = null
# obs_pos = np.array([r, phi, z]).T
# dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
# H1 = field_H_cylinder_tile(obs_pos, dim, mag)
# H2 = H_total_final(obs_pos, dim, mag)
# assert np.allclose(np.nan_to_num(H1), np.nan_to_num(H2))

# DATA += [H2]


# # cases [123, 223, 133, 233, 125, 225, 135, 235]
# r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
# z1 = z
# phi1 = phi+np.pi
# r1 = null
# obs_pos = np.array([r, phi, z]).T
# dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
# H1 = field_H_cylinder_tile(obs_pos, dim, mag)
# H2 = H_total_final(obs_pos, dim, mag)
# assert np.allclose(np.nan_to_num(H1), np.nan_to_num(H2))

# DATA += [H2]

# # cases [125, 225, 135, 235, 124, 224, 134, 234]
# r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
# z1 = z
# phi1 = phi+np.pi
# r = r2
# obs_pos = np.array([r, phi, z]).T
# dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
# H1 = field_H_cylinder_tile(obs_pos, dim, mag)
# H2 = H_total_final(obs_pos, dim, mag)
# assert np.allclose(np.nan_to_num(H1), np.nan_to_num(H2))

# DATA += [H2]

# # cases [211, 221, 212, 222]
# r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
# phi1 = phi
# phi2 = phi+np.pi
# r = null
# r1 = null
# obs_pos = np.array([r, phi, z]).T
# dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
# H1 = field_H_cylinder_tile(obs_pos, dim, mag)
# H2 = H_total_final(obs_pos, dim, mag)
# assert np.allclose(np.nan_to_num(H1), np.nan_to_num(H2))

# DATA += [H2]

# # cases [214, 224, 215, 225]
# r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
# phi1 = phi
# phi2 = phi+np.pi
# r = r1
# obs_pos = np.array([r, phi, z]).T
# dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
# H1 = field_H_cylinder_tile(obs_pos, dim, mag)
# H2 = H_total_final(obs_pos, dim, mag)
# assert np.allclose(np.nan_to_num(H1), np.nan_to_num(H2))

# DATA += [H2]

# # cases [111, 211, 121, 221, 112, 212, 122, 222]
# r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
# z = z1
# phi1 = phi
# phi2 = phi+np.pi
# r = null
# r1 = null
# obs_pos = np.array([r, phi, z]).T
# dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
# H1 = field_H_cylinder_tile(obs_pos, dim, mag)
# H2 = H_total_final(obs_pos, dim, mag)
# assert np.allclose(np.nan_to_num(H1), np.nan_to_num(H2))

# DATA += [H2]

# # cases [111, 211, 131, 231, 112, 212, 132, 232]
# r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
# z = z1
# phi1 = phi
# r = null
# r1 = null
# obs_pos = np.array([r, phi, z]).T
# dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
# H1 = field_H_cylinder_tile(obs_pos, dim, mag)
# H2 = H_total_final(obs_pos, dim, mag)
# assert np.allclose(np.nan_to_num(H1), np.nan_to_num(H2))

# DATA += [H2]

# # cases [115, 215, 135, 235, 114, 214, 134, 234]
# r, r1, r2, phi, phi1, phi2, z, z1, z2 = R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2
# z = z1
# phi1 = phi
# r = r2
# obs_pos = np.array([r, phi, z]).T
# dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
# H1 = field_H_cylinder_tile(obs_pos, dim, mag)
# H2 = H_total_final(obs_pos, dim, mag)
# assert np.allclose(np.nan_to_num(H1), np.nan_to_num(H2))

# DATA += [H2]

# DATA = np.array(DATA)
# np.save('data_test_cy_cases', DATA)

# DATA_INPUT = np.array([R, R1, R2, PHI, PHI1, PHI2, Z, Z1, Z2])
# np.save('data_test_cy_cases_inp', DATA_INPUT)
# np.save('data_test_cy_cases_inp2', mag)


DATA_INPUT = np.load('tests/testdata/testdata_cy_cases_inp.npy')
mag = np.load('tests/testdata/testdata_cy_cases_inp2.npy')
DATA = np.load('tests/testdata/testdata_cy_cases.npy')
null = np.zeros(1111)


def test_cases0():
    """ cases [112, 212, 132, 232]
    """
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = DATA_INPUT
    z1=z
    phi1 = phi
    r = null
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H = field_H_cylinder_tile(obs_pos, dim, mag)
    H0 = DATA[0]
    assert np.allclose(H, H0)


def test_cases1():
    """ cases [122, 222, 132, 232]
    """
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = DATA_INPUT
    z1=z
    phi1 = phi+np.pi
    r = null
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H = field_H_cylinder_tile(obs_pos, dim, mag)
    H0 = DATA[1]
    assert np.allclose(H, H0)


def test_cases2():
    """cases [113, 213, 133, 233, 115, 215, 135, 235]
    """
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = DATA_INPUT
    z1 = z
    phi1 = phi
    r1 = null
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H = field_H_cylinder_tile(obs_pos, dim, mag)
    H0 = DATA[2]
    assert np.allclose(np.nan_to_num(H), np.nan_to_num(H0))


def test_cases3():
    """ cases [123, 223, 133, 233, 125, 225, 135, 235]
    """
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = DATA_INPUT
    z1 = z
    phi1 = phi+np.pi
    r1 = null
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H = field_H_cylinder_tile(obs_pos, dim, mag)
    H0 = DATA[3]
    assert np.allclose(np.nan_to_num(H), np.nan_to_num(H0))


def test_cases4():
    """ cases [125, 225, 135, 235, 124, 224, 134, 234]
    """
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = DATA_INPUT
    z1 = z
    phi1 = phi+np.pi
    r = r2
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H = field_H_cylinder_tile(obs_pos, dim, mag)
    H0 = DATA[4]
    assert np.allclose(np.nan_to_num(H), np.nan_to_num(H0))


def test_cases5():
    """ cases [211, 221, 212, 222]
    """
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = DATA_INPUT
    phi1 = phi
    phi2 = phi+np.pi
    r = null
    r1 = null
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H = field_H_cylinder_tile(obs_pos, dim, mag)
    H0 = DATA[5]
    assert np.allclose(np.nan_to_num(H), np.nan_to_num(H0))


def test_cases6():
    """ cases [214, 224, 215, 225]
    """
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = DATA_INPUT
    phi1 = phi
    phi2 = phi+np.pi
    r = r1
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H = field_H_cylinder_tile(obs_pos, dim, mag)
    H0 = DATA[6]
    assert np.allclose(np.nan_to_num(H), np.nan_to_num(H0))


def test_cases7():
    """ cases [111, 211, 121, 221, 112, 212, 122, 222]
    """
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = DATA_INPUT
    z = z1
    phi1 = phi
    phi2 = phi+np.pi
    r = null
    r1 = null
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H = field_H_cylinder_tile(obs_pos, dim, mag)
    H0 = DATA[7]
    assert np.allclose(np.nan_to_num(H), np.nan_to_num(H0))


def test_cases8():
    """ cases [111, 211, 131, 231, 112, 212, 132, 232]
    """
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = DATA_INPUT
    z = z1
    phi1 = phi
    r = null
    r1 = null
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H = field_H_cylinder_tile(obs_pos, dim, mag)
    H0 = DATA[8]
    assert np.allclose(np.nan_to_num(H), np.nan_to_num(H0))


def test_cases9():
    """ cases [115, 215, 135, 235, 114, 214, 134, 234]
    """
    r, r1, r2, phi, phi1, phi2, z, z1, z2 = DATA_INPUT
    z = z1
    phi1 = phi
    r = r2
    obs_pos = np.array([r, phi, z]).T
    dim = np.array([r1, r2, phi1, phi2, z1, z2]).T
    H = field_H_cylinder_tile(obs_pos, dim, mag)
    H0 = DATA[9]
    assert np.allclose(np.nan_to_num(H), np.nan_to_num(H0))


# from magpylib._lib.fields.field_BH_cylinder_old import field_BH_cylinder
# import magpylib as magpy
# magpy.Config.ITER_CYLINDER = 10000
# N = 100
# mag = (np.random.rand(N, 3)-.5)*1000
# dim = np.random.rand(N, 3)
# poso = (np.random.rand(N, 3)-.5)
# dim2 = dim[:,:2]
# H0 = field_BH_cylinder(True, mag, dim2, poso)
# np.save('testdata_full_cyl', np.array([mag,dim,poso,H0]))
def test_cylinder_field1():
    """ test the new cylinder field against old, full-cylinder
    implementations
    """
    N = 100
    magg, dim, poso, H0 = np.load('tests/testdata/testdata_full_cyl.npy')

    nulll = np.zeros(N)
    eins = np.ones(N)
    d,h,_ = dim.T
    dim6 = np.array([nulll, d/2, nulll, eins*360, -h/2, h/2]).T
    H1 = field_BH_cylinder_tile(True, magg, dim6, poso)

    assert np.allclose(H1, H0)


def test_cylinder_field2():
    """ testing B for all input combinations in/out/surface of Tile solution"""
    src = magpy.magnet.CylinderSegment((22,33,44), (1,2,2,0,90))

    binn = ( 5.52525937, 13.04561569, 40.11111556)
    bout = (0.0177018,  0.1277188,  0.27323195)
    nulll = (0,0,0)

    # only inside
    btest = np.array([binn]*3)
    B = src.getB([[.5,.6,.3]]*3)
    assert np.allclose(B, btest)

    # only surf
    btest = np.array([nulll]*3)
    B = src.getB([[1,0,0]]*3)
    assert np.allclose(B, btest)

    # only outside
    btest = np.array([bout]*3)
    B = src.getB([[1,2,3]]*3)
    assert np.allclose(B, btest)

    # surf + out
    btest = np.array([nulll,nulll,bout])
    B = src.getB([.6,0,1],[1,0,.5],[1,2,3])
    assert np.allclose(B, btest)

    # surf + in
    btest = np.array([nulll,nulll,binn])
    B = src.getB([0,.5,1],[1,0,.5],[.5,.6,.3])
    assert np.allclose(B, btest)

    # in + out
    btest = np.array([bout,binn])
    B = src.getB([1,2,3],[.5,.6,.3])
    assert np.allclose(B, btest)

    # in + out + surf
    btest = np.array([nulll,nulll,binn,bout,nulll, nulll])
    B = src.getB([.5,.5,1],[0,1,.5],[.5,.6,.3],[1,2,3],[.5,.6,-1], [0,1,-.3])
    assert np.allclose(B, btest)


def test_cylinder_field3():
    """ testing H for all input combinations in/out/surface of Tile solution"""
    src = magpy.magnet.CylinderSegment((22,33,44), (1,2,2,0,90))

    hinn = (-13.11018204, -15.87919449,  -3.09467591)
    hout = (0.01408664, 0.1016354,  0.21743108)
    nulll = (0,0,0)

    # only inside
    htest = np.array([hinn]*3)
    H = src.getH([[.5,.6,.3]]*3)
    assert np.allclose(H, htest)

    # only surf
    htest = np.array([nulll]*3)
    H = src.getH([[1,0,0]]*3)
    assert np.allclose(H, htest)

    # only outside
    htest = np.array([hout]*3)
    H = src.getH([[1,2,3]]*3)
    assert np.allclose(H, htest)

    # surf + out
    htest = np.array([nulll,nulll,hout])
    H = src.getH([.6,0,1],[1,0,.5],[1,2,3])
    assert np.allclose(H, htest)

    # surf + in
    htest = np.array([nulll,nulll,hinn])
    H = src.getH([0,.5,1],[1,0,.5],[.5,.6,.3])
    assert np.allclose(H, htest)

    # in + out
    htest = np.array([hout,hinn])
    H = src.getH([1,2,3],[.5,.6,.3])
    assert np.allclose(H, htest)

    # in + out + surf
    htest = np.array([nulll,nulll,hinn,hout,nulll, nulll])
    H = src.getH([.5,.5,1],[0,1,.5],[.5,.6,.3],[1,2,3],[.5,.6,-1], [0,1,-.3])
    assert np.allclose(H, htest)


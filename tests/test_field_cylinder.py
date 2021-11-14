"""
Testing all cases against a large set of pre-computed values
"""

import numpy as np
from magpylib._src.fields.field_BH_cylinder_tile import (magnet_cyl_tile_H_Slanovc2021,
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
# H1 = magnet_cyl_tile_H_Slanovc2021(obs_pos, dim, mag)
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
# H1 = magnet_cyl_tile_H_Slanovc2021(obs_pos, dim, mag)
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
# H1 = magnet_cyl_tile_H_Slanovc2021(obs_pos, dim, mag)
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
# H1 = magnet_cyl_tile_H_Slanovc2021(obs_pos, dim, mag)
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
# H1 = magnet_cyl_tile_H_Slanovc2021(obs_pos, dim, mag)
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
# H1 = magnet_cyl_tile_H_Slanovc2021(obs_pos, dim, mag)
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
# H1 = magnet_cyl_tile_H_Slanovc2021(obs_pos, dim, mag)
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
# H1 = magnet_cyl_tile_H_Slanovc2021(obs_pos, dim, mag)
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
# H1 = magnet_cyl_tile_H_Slanovc2021(obs_pos, dim, mag)
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
# H1 = magnet_cyl_tile_H_Slanovc2021(obs_pos, dim, mag)
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
    H = magnet_cyl_tile_H_Slanovc2021(mag, dim, obs_pos)
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
    H = magnet_cyl_tile_H_Slanovc2021(mag, dim, obs_pos)
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
    H = magnet_cyl_tile_H_Slanovc2021(mag, dim, obs_pos)
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
    H = magnet_cyl_tile_H_Slanovc2021(mag, dim, obs_pos)
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
    H = magnet_cyl_tile_H_Slanovc2021(mag, dim, obs_pos)
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
    H = magnet_cyl_tile_H_Slanovc2021(mag, dim, obs_pos)
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
    H = magnet_cyl_tile_H_Slanovc2021(mag, dim, obs_pos)
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
    H = magnet_cyl_tile_H_Slanovc2021(mag, dim, obs_pos)
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
    H = magnet_cyl_tile_H_Slanovc2021(mag, dim, obs_pos)
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
    H = magnet_cyl_tile_H_Slanovc2021(mag, dim, obs_pos)
    H0 = DATA[9]
    assert np.allclose(np.nan_to_num(H), np.nan_to_num(H0))


# from magpylib._src.fields.field_BH_cylinder_old import field_BH_cylinder
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


def test_cylinder_slanovc_field2():
    """ testing B for all input combinations in/out/surface of Tile solution"""
    src = magpy.magnet.CylinderSegment((22,33,44), (.5,1,2,0,90))

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


def test_cylinder_slanovc_field3():
    """ testing H for all input combinations in/out/surface of Tile solution"""
    src = magpy.magnet.CylinderSegment((22,33,44), (.5,1,2,0,90))

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


def test_cylinder_rauber_field4():
    """
    test continuiuty across indefinite form in cylinder_rauber field when observer at r=r0
    """
    src = magpy.magnet.Cylinder((22,33,0), (2,2))
    es = list(10**-np.linspace(11,15,50))
    xs = np.r_[1-np.array(es), 1, 1+np.array(es)[::-1]]
    possis = [(x,0,1.5) for x in xs]
    B = src.getB(possis)
    B = B/B[25]
    assert np.all(abs(1-B) < 1e-8)


def test_cylinder_tile_negative_phi():
    """ same result for phi>0 and phi<0 inputs
    """
    src1 = magpy.magnet.CylinderSegment((11,22,33), (2,4,4,0,45))
    src2 = magpy.magnet.CylinderSegment((11,22,33), (2,4,4,-360,-315))
    B1 = src1.getB((1,.5,.1))
    B2 = src2.getB((1,.5,.1))
    assert np.allclose(B1,B2)


def test_cylinder_tile_vs_fem():
    """ test against fem results
    """
    fd1, fd2, fd3, fd4 = np.load('tests/testdata/testdata_femDat_cylinder_tile2.npy')

    # chosen magnetization vectors
    mag1 = np.array((1,-1,0))/np.sqrt(2)*1000
    mag2 = np.array((0,0,1))*1000
    mag3 = np.array((0,1,-1))/np.sqrt(2)*1000

    # Magpylib magnet collection
    m1 = magpy.magnet.CylinderSegment(mag1, (1,2,1,-90,0))
    m2 = magpy.magnet.CylinderSegment(mag2, (1,2.5,1.5,200,250))
    m3 = magpy.magnet.CylinderSegment(mag3, (.75,3,0.5,70,180))
    col = m1+m2+m3

    # create observer circles (see FEM screen shot)
    n = 101
    ts = np.linspace(0,359.999,n)*np.pi/180
    poso1 = np.array([0.5*np.cos(ts), 0.5*np.sin(ts), np.zeros(n)]).T
    poso2 = np.array([1.5*np.cos(ts), 1.5*np.sin(ts), np.zeros(n)]).T
    poso3 = np.array([1.5*np.cos(ts), 1.5*np.sin(ts), np.ones(n)]).T
    poso4 = np.array([3.5*np.cos(ts), 3.5*np.sin(ts), np.zeros(n)]).T

    # compute and plot fields
    B1 = col.getB(poso1)
    B2 = col.getB(poso2)
    B3 = col.getB(poso3)
    B4 = col.getB(poso4)

    amp1 = np.linalg.norm(B1, axis=1)
    amp2 = np.linalg.norm(B2, axis=1)
    amp3 = np.linalg.norm(B3, axis=1)
    amp4 = np.linalg.norm(B4, axis=1)

    assert np.amax((fd1[:,1:]*1000-B1).T/amp1)<0.05
    assert np.amax((fd2[5:-5,1:]*1000-B2[5:-5]).T/amp2[5:-5])<0.05
    assert np.amax((fd3[:,1:]*1000-B3).T/amp3)<0.05
    assert np.amax((fd4[:,1:]*1000-B4).T/amp4)<0.05

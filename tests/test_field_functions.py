import pickle
import os
import numpy as np
from magpylib._lib.fields.field_BH_cuboid import field_BH_cuboid
from magpylib._lib.fields.field_BH_cylinder_tile import field_BH_cylinder_tile
from magpylib._lib.fields.field_BH_sphere import field_BH_sphere
from magpylib._lib.fields.field_BH_dipole import field_BH_dipole
from magpylib._lib.fields.field_BH_circular import field_BH_circular
from magpylib._lib.fields.field_BH_line import field_BH_line, field_BH_line_from_vert
from magpylib import Config

# # GENERATE TEST DATA
# n = 500
# magpy.Config.EDGESIZE = 1e-14

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
#     Bs += [field_BH_cuboid(True, mag, dim, pos)]
# Bs = np.array(Bs)

# print(Bs)
# pickle.dump([mag, dims, poss, Bs], open('testdata_field_BH_cuboid.p','wb'))


def test_field_BH_cuboid():
    """ test cuboid field
    """
    Config.EDGESIZE=1e-14
    mag, dims, poss, B = pickle.load(open(
        os.path.abspath('tests/testdata/testdata_field_BH_cuboid.p') ,'rb'))
    Btest = []
    for dim,pos in zip(dims,poss):
        Btest += [field_BH_cuboid(True, mag, dim, pos)]
    Btest = np.array(Btest)
    assert np.allclose(B, Btest), 'Cuboid field computation broken'


def test_field_BH_cuboid_mag0():
    """ test cuboid field magnetization=0
    """
    n = 10
    mag = np.zeros((n,3))
    dim = np.random.rand(n,3)
    pos = np.random.rand(n,3)
    B = field_BH_cuboid(True, mag, dim, pos)
    assert np.allclose(mag,B), 'Cuboid magnetization=0 case broken'


def test_field_BH_cylinder_tile_mag0():
    """test cylinder_tile field magnetization=0
    """
    n = 10
    mag = np.zeros((n,3))
    r1,r2,phi1,phi2,z1,z2 = np.random.rand(6,n)
    r2=r1+r2
    phi2=phi1+phi2
    z2=z1+z2
    dim = np.array([r1,r2,phi1,phi2,z1,z2]).T
    pos = np.random.rand(n,3)
    B = field_BH_cylinder_tile(True, mag, dim, pos)
    assert np.allclose(mag,B), 'CylinderTile magnetization=0 case broken'


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
    """ test cuboid field magnetization=0
    """
    n = 10
    mag = np.zeros((n,3))
    dim = np.random.rand(n)
    pos = np.random.rand(n,3)
    B = field_BH_sphere(True, mag, dim, pos)
    assert np.allclose(mag,B), 'Cuboid magnetization=0 case broken'


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


def test_field_circular():
    """ test if field function gives correct outputs
    """
    # from hyperphysics
    # current = 1A
    # loop radius = 1mm
    # B at center = 0.6283185307179586 mT
    # B at 1mm on zaxis = 0.22214414690791835 mT
    pos_test_hyper = [[0,0,0], [0,0,1]]
    Btest_hyper = [[0,0,0.6283185307179586], [0,0,0.22214414690791835]]

    # from magpylib 2
    pos_test_mag2 = [[1,2,3], [-3,2,1], [1,-.2,.3], [1,.2,-1],
        [-.1,-.2,3], [-1,.2,-.3], [3,-3,-3], [-2,-.2,-.3]]
    Btest_mag2 = [[0.44179833, 0.88359665, 0.71546231],
        [-0.53137126,  0.35424751, -0.59895825],
        [ 72.87320789, -14.57464158,  22.07633404],
        [-13.75612867,  -2.75122573,  11.36467552],
        [-0.10884885, -0.21769769,  2.41206364],
        [ 72.87320789, -14.57464158,  22.07633404],
        [-0.27939151,  0.27939151,  0.01220605],
        [ 3.25697271,  0.32569727, -5.49353046]]

    pos_test = np.array(pos_test_hyper + pos_test_mag2 + [[1,0,0]])
    Btest = np.array(Btest_hyper + Btest_mag2 + [[0,0,0]])

    current = np.array([1,1] + [123]*8 + [123])
    dim = np.array([2,2] + [2]*8 + [2])

    B = field_BH_circular(True, current, dim, pos_test)

    assert np.allclose(B, Btest)

    Htest = Btest*10/4/np.pi
    H = field_BH_circular(False, current, dim, pos_test)
    assert np.allclose(H, Htest)


def test_field_circular2():
    """ test if field function accepts correct inputs
    """
    curr = np.array([1])
    dim = np.array([2])
    poso = np.array([[0,0,0]])
    B = field_BH_circular(True, curr, dim, poso)

    curr = np.array([1]*2)
    dim = np.array([2]*2)
    poso = np.array([[0,0,0]]*2)
    B2 = field_BH_circular(True, curr, dim, poso)

    assert np.allclose(B,B2[0])
    assert np.allclose(B,B2[1])


def test_field_line():
    """ test line current for all cases
    """

    c1 = np.array([1])
    po1 = np.array([(1,2,3)])
    ps1 = np.array([(0,0,0)])
    pe1 = np.array([(2,2,2)])

    # only normal
    B1 = field_BH_line(True, c1, ps1, pe1, po1)
    x1 = np.array([[ 0.02672612, -0.05345225, 0.02672612]])
    assert np.allclose(x1, B1)

    # only on_line
    po1b = np.array([(1,1,1)])
    B2 = field_BH_line(True, c1, ps1, pe1, po1b)
    x2 = np.zeros((1,3))
    assert np.allclose(x2, B2)

    # only zero-segment
    B3 = field_BH_line(True, c1, ps1, ps1, po1)
    x3 = np.zeros((1,3))
    assert np.allclose(x3, B3)

    # only on_line and zero_segment
    c2 = np.array([1]*2)
    ps2 = np.array([(0,0,0)]*2)
    pe2 = np.array([(0,0,0),(2,2,2)])
    po2 = np.array([(1,2,3),(1,1,1)])
    B4 = field_BH_line(True, c2, ps2, pe2, po2)
    x4 = np.zeros((2,3))
    assert np.allclose(x4, B4)

    # normal + zero_segment
    po2b = np.array([(1,2,3),(1,2,3)])
    B5 = field_BH_line(True, c2, ps2, pe2, po2b)
    x5 = np.array([[0,0,0],[ 0.02672612, -0.05345225, 0.02672612]])
    assert np.allclose(x5, B5)

    # normal + on_line
    pe2b = np.array([(2,2,2)]*2)
    B6 = field_BH_line(True, c2, ps2, pe2b, po2)
    x6 = np.array([[0.02672612, -0.05345225, 0.02672612],[0,0,0]])
    assert np.allclose(x6, B6)

    # normal + zero_segment + on_line
    c4 = np.array([1]*3)
    ps4 = np.array([(0,0,0)]*3)
    pe4 = np.array([(0,0,0),(2,2,2),(2,2,2)])
    po4 = np.array([(1,2,3),(1,2,3),(1,1,1)])
    B7 = field_BH_line(True, c4, ps4, pe4, po4)
    x7 = np.array([[0,0,0], [0.02672612, -0.05345225, 0.02672612], [0,0,0]])
    assert np.allclose(x7, B7)


def test_field_line_from_vert():
    """ test the Line field from vertex input
    """
    p = np.array([(1,2,2), (1,2,3), (-1,0,-3)])
    curr = np.array([1, 5, -3])

    vert1 = np.array([(0,0,0),(1,1,1),(2,2,2),(3,3,3),(1,2,3),(-3,4,-5)])
    vert2 = np.array([(0,0,0),(3,3,3),(-3,4,-5)])
    vert3 = np.array([(1,2,3),(-2,-3,3),(3,2,1),(3,3,3)])

    pos_tiled = np.tile(p, (3,1))
    B_vert = field_BH_line_from_vert(True, curr, [vert1,vert2,vert3], pos_tiled)

    B = []
    for i,vert in enumerate([vert1,vert2,vert3]):
        for pos in p:
            p1 = vert[:-1]
            p2 = vert[1:]
            po = np.array([pos]*(len(vert)-1))
            cu = np.array([curr[i]]*(len(vert)-1))
            B += [np.sum(field_BH_line(True, cu, p1, p2, po), axis=0)]
    B = np.array(B)

    assert np.allclose(B_vert, B)

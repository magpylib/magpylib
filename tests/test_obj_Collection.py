import pickle
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import magpylib as mag3

# # GENERATE TESTDATA
# N = 5
# mags = (np.random.rand(N,6,3)-0.5)*1000
# dims3 = (np.random.rand(N,3,3)-0.5)*5     # 5x box
# dims2 = (np.random.rand(N,3,2)-0.5)*5     # 5x cylinder
# posos = (np.random.rand(N,23,3)-0.5)*10 #readout at 333 positions

# angs =  (np.random.rand(N,18)-0.5)*2*10 # each step rote by max 10 deg
# axs =   (np.random.rand(N,18,3)-0.5)
# anchs = (np.random.rand(N,18,3)-0.5)*5.5
# movs =  (np.random.rand(N,18,3)-0.5)*0.5
# rvs = (np.random.rand(N,3)-.5)*0.1

# B = []
# for mag,dim2,dim3,ang,ax,anch,mov,poso,rv in zip(mags,dims2,dims3,angs,axs,anchs,movs,posos,rvs):
#     rot = R.from_rotvec(rv)
#     pm1b = mag3.magnet.Box(mag[0],dim3[0])
#     pm2b = mag3.magnet.Box(mag[1],dim3[1])
#     pm3b = mag3.magnet.Box(mag[2],dim3[2])
#     pm4b = mag3.magnet.Cylinder(mag[3],dim2[0])
#     pm5b = mag3.magnet.Cylinder(mag[4],dim2[1])
#     pm6b = mag3.magnet.Cylinder(mag[5],dim2[2])

#     # 18 subsequent operations
#     for a,aa,aaa,mv in zip(ang,ax,anch,mov):
#         for pm in [pm1b,pm2b,pm3b,pm4b,pm5b,pm6b]:
#             pm.move(mv).rotate_from_angax(a,aa,aaa).rotate(rot,aaa)
#     B += [mag3.getB([pm1b,pm2b,pm3b,pm4b,pm5b,pm6b], poso, sumup=True, niter=100)]
# B = np.array(B)
# inp = [mags,dims2,dims3,posos,angs,axs,anchs,movs,rvs,B]
# pickle.dump(inp,open('testdata_Collection.p', 'wb'))


def test_Collection_basics():
    """  test Collection fundamentals, test against magpylib2 fields
    """
    # pylint: disable=pointless-statement
    # data generated below
    data = pickle.load(open(os.path.abspath('./tests/testdata/testdata_Collection.p'), 'rb'))
    mags,dims2,dims3,posos,angs,axs,anchs,movs,rvs,_ = data

    B1,B2,B3 = [],[],[]
    for mag,dim2,dim3,ang,ax,anch,mov,poso,rv in zip(mags,dims2,dims3,angs,
                                                     axs,anchs,movs,posos,rvs):
        rot = R.from_rotvec(rv)

        pm1b = mag3.magnet.Box(mag[0],dim3[0])
        pm2b = mag3.magnet.Box(mag[1],dim3[1])
        pm3b = mag3.magnet.Box(mag[2],dim3[2])
        pm4b = mag3.magnet.Cylinder(mag[3],dim2[0])
        pm5b = mag3.magnet.Cylinder(mag[4],dim2[1])
        pm6b = mag3.magnet.Cylinder(mag[5],dim2[2])

        pm1 = mag3.magnet.Box(mag[0],dim3[0])
        pm2 = mag3.magnet.Box(mag[1],dim3[1])
        pm3 = mag3.magnet.Box(mag[2],dim3[2])
        pm4 = mag3.magnet.Cylinder(mag[3],dim2[0])
        pm5 = mag3.magnet.Cylinder(mag[4],dim2[1])
        pm6 = mag3.magnet.Cylinder(mag[5],dim2[2])

        col1 = mag3.Collection(pm1,[pm2,pm3])
        col1 + pm4
        col2 = mag3.Collection(pm5,pm6)
        col1 + col2
        col1 - pm5 - pm4
        col1.remove(pm1)
        col3 = col1.copy() + pm5 + pm4 + pm1
        col1.add(pm5,pm4,pm1)

        # 18 subsequent operations
        for a,aa,aaa,mv in zip(ang,ax,anch,mov):
            for pm in [pm1b,pm2b,pm3b,pm4b,pm5b,pm6b]:
                pm.move(mv).rotate_from_angax(a,aa,aaa).rotate(rot,aaa)

            col1.move(mv).rotate_from_angax(a,aa,aaa).rotate(rot,aaa)

        B1 += [mag3.getB([pm1b,pm2b,pm3b,pm4b,pm5b,pm6b], poso, sumup=True)]
        B2 += [col1.getB(poso)]
        B3 += [col3.getB(poso)]

    B1 = np.array(B1)
    B2 = np.array(B2)
    B3 = np.array(B3)

    assert np.allclose(B1, B2), 'Collection testfail1'
    assert np.allclose(B1, B3), 'Collection testfail2'


def test_col_get_item():
    """ test get_item with collections
    """
    pm1 = mag3.magnet.Box((1,2,3),(1,2,3))
    pm2 = mag3.magnet.Box((1,2,3),(1,2,3))
    pm3 = mag3.magnet.Box((1,2,3),(1,2,3))

    col = mag3.Collection(pm1,pm2,pm3)
    assert col[1]==pm2, 'get_item failed'


def test_col_getH():
    """ test collection getH
    """
    pm1 = mag3.magnet.Sphere((1,2,3),3)
    pm2 = mag3.magnet.Sphere((1,2,3),3)
    col = mag3.Collection(pm1,pm2)
    H = col.getH((0,0,0))
    H1 = pm1.getH((0,0,0))
    assert np.all(H==2*H1), 'col getH fail'


def test_col_reset_path():
    """ testing display
    """
    pm1 = mag3.magnet.Box((1,2,3),(1,2,3))
    pm2 = mag3.magnet.Box((1,2,3),(1,2,3))
    col = mag3.Collection(pm1,pm2)
    col.move([(1,2,3)]*10)
    col.reset_path()
    assert col[0].position.ndim==1, 'col reset path fail'
    assert col[1].position.ndim==1, 'col reset path fail'


def test_Collection_squeeze():
    """ testing squeeze output
    """
    pm1 = mag3.magnet.Box((1,2,3),(1,2,3))
    pm2 = mag3.magnet.Box((1,2,3),(1,2,3))
    col = mag3.Collection(pm1,pm2)
    sensor = mag3.Sensor(pixel=[(1,2,3),(1,2,3)])
    B = col.getB(sensor)
    assert B.shape==(2,3)
    H = col.getH(sensor)
    assert H.shape==(2,3)

    B = col.getB(sensor,squeeze=False)
    assert B.shape==(1,1,1,2,3)
    H = col.getH(sensor,squeeze=False)
    assert H.shape==(1,1,1,2,3)


def test_Collection_with_Dipole():
    """ Simple test of Dipole in Collection
    """
    src = mag3.misc.Dipole(moment=(1,2,3),position=(1,2,3))
    col = mag3.Collection(src)
    sens = mag3.Sensor()

    B = mag3.getB(col,sens)
    Btest = np.array([0.00303828,0.00607656,0.00911485])
    assert np.allclose(B, Btest)


def test_repr_collection():
    """ test __repr__
    """
    pm1 = mag3.magnet.Box((1,2,3),(1,2,3))
    pm2 = mag3.magnet.Cylinder((1,2,3),(2,3))
    col = mag3.Collection(pm1,pm2)
    assert col.__repr__()[:10]== 'Collection', 'Collection repr failed'


def test_adding_sources():
    """ test if all sources can be added
    """
    src1 = mag3.magnet.Box((1,2,3), (1,2,3))
    src2 = mag3.magnet.Cylinder((1,2,3), (1,2))
    src3 = mag3.magnet.Sphere((1,2,3), 1)
    src4 = mag3.current.Circular(1,1)
    src5 = mag3.current.Line(1, [(1,2,3),(2,3,4)])
    src6 = mag3.misc.Dipole((1,2,3))
    col = src1 + src2 + src3 + src4 + src5 + src6

    strs = ''
    for src in col:
        strs += str(src)[:3]

    assert strs == 'BoxCylSphCirLinDip'

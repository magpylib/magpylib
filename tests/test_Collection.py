import pickle
import numpy as np
import magpylib3 as mag3
from scipy.spatial.transform import Rotation as R

# # GENERATE TESTDATA
# N = 10
# mags = (np.random.rand(N,6,3)-0.5)*1000  
# dims3 = (np.random.rand(N,3,3)-0.5)*5     # 5x box
# dims2 = (np.random.rand(N,3,2)-0.5)*5     # 5x cylinder
# posos = (np.random.rand(N,333,3)-0.5)*10 #readout at 333 positions

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


def test_Collection():
    # data generated below
    data = pickle.load(open('tests/testdata/testdata_Collection.p', 'rb'))
    mags,dims2,dims3,posos,angs,axs,anchs,movs,rvs,Btest = data

    B1,B2,B3 = [],[],[]
    for mag,dim2,dim3,ang,ax,anch,mov,poso,rv in zip(mags,dims2,dims3,angs,axs,anchs,movs,posos,rvs):
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

        B1 += [mag3.getB([pm1b,pm2b,pm3b,pm4b,pm5b,pm6b], poso, sumup=True, niter=100)]
        B2 += [col1.getB(poso,niter=100)]
        B3 += [col3.getB(poso,niter=100)]
        
    B1 = np.array(B1)
    B2 = np.array(B2)
    B3 = np.array(B3)

    assert np.allclose(B1,B2), 'Collection testfail1'
    assert np.allclose(B1,B3), 'Collection testfail2'
    assert np.allclose(B1,Btest), 'Collection testfail3'

    



#     pm1b = mag3.magnet.Box(mag[0],dim3[0])
#     pm2b = mag3.magnet.Box(mag[1],dim3[1])
#     pm3b = mag3.magnet.Box(mag[2],dim3[2])
#     pm4b = mag3.magnet.Cylinder(mag[3],dim2[0])
#     pm5b = mag3.magnet.Cylinder(mag[4],dim2[1])
#     pm6b = mag3.magnet.Cylinder(mag[5],dim2[2])

#     col1 = mag3.Collection(pm1,[pm2,pm3])
#     col1 + pm4
#     col2 = mag3.Collection(pm5,pm6)
#     col1 + col2
#     col1 - pm5 - pm4
#     col1.remove(pm1)
#     col3 = col1.copy() + pm5 + pm4 + pm1
#     col1.add(pm5,pm4,pm1)
    
#     # 18 subsequent operations
#     for a,aa,aaa,mv in zip(ang,ax,anch,mov):
#         for pm in [pm1b,pm2b,pm3b,pm4b,pm5b,pm6b]:
#             pm.move(mv).rotate_from_angax(a,aa,aaa).rotate(rot,aaa)

#         col1.move(mv).rotate_from_angax(a,aa,aaa).rotate(rot,aaa)

#     B1 = mag3.getB([pm1b,pm2b,pm3b,pm4b,pm5b,pm6b], poso, sumup=True, niter=100)
#     B2 = col1.getB(poso,niter=100)
#     B3 = col3.getB(poso,niter=100)
#     print(np.allclose(B1,B2))
#     print(np.allclose(B1,B3))
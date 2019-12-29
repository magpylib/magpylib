#%% MAIN

import numpy as np
from magpylib.source.magnet import Box, Cylinder, Sphere
from magpylib.source.moment import Dipole
from magpylib.source.current import Circular
from magpylib.vector import getBv_magnet, getBv_current, getBv_moment
from magpylib.math import axisFromAngles
from magpylib.math import angleAxisRotationV

def test_vectorMagnet():

    # calculate the B-field for the 3axis joystick system with
    # vector and non-vecor code + compare

    #base geometry
    displM = 3
    dCoT = 0
    gap = 1
    a,b,c = 4,4,4
    Mx, My,Mz = 0,1000,0

    mag = [Mx,My,Mz]
    dim = [a,b,c]
    posM = [displM,0,c/2+gap]
    posS = [0,0,0]
    anch = [0,0,gap+c+dCoT]

    Nphi = 3
    Npsi = 33
    Nth = 11
    NN = Nphi*Npsi*Nth
    PHI = np.linspace(0,360,Nphi+1)[:-1]
    PSI = np.linspace(0,360,Npsi)
    TH = np.linspace(0,10,Nth)

    MAG = np.array([mag]*NN)
    POSo = np.array([posS]*NN)
    POSm = np.array([posM]*NN)

    ANG1 = np.array(list(PHI)*(Npsi*Nth))
    AX1 = np.array([[0,0,1]]*NN)
    ANCH1 = np.array([anch]*NN)

    ANG2 = np.array([a for a in TH for _ in range(Nphi*Npsi)])
    angles = np.array([a for a in PSI for _ in range(Nphi)]*Nth)
    AX2 = angleAxisRotationV(np.array([[1,0,0]]*NN),angles,np.array([[0,0,1]]*NN),np.array([[0,0,0]]*NN))  
    ANCH2 = np.array([anch]*NN)


    # BOX ---------------------------------------------------------
    # classic
    def getB(phi,th,psi):
        pm = Box(mag=mag,dim=dim,pos=posM)
        axis = axisFromAngles([psi,90])    
        pm.rotate(phi,[0,0,1],anchor=[0,0,0])    
        pm.rotate(th,axis,anchor=anch)
        return pm.getB(posS)
    Bc = np.array([[[getB(phi,th,psi) for phi in PHI] for psi in PSI] for th in TH])

    # vector
    DIM = np.array([dim]*NN)
    Bv = getBv_magnet('box',MAG,DIM,POSm,POSo,[ANG1,ANG2],[AX1,AX2],[ANCH1,ANCH2])
    Bv = Bv.reshape([Nth,Npsi,Nphi,3])

    # assert
    assert np.amax(Bv-Bc) < 1e-10, "bad magpylib vector Box"

    # SPHERE ---------------------------------------------------------
    # classic
    dim2 = 4
    def getB2(phi,th,psi):
        pm = Sphere(mag=mag,dim=dim2,pos=posM)
        axis = axisFromAngles([psi,90])    
        pm.rotate(phi,[0,0,1],anchor=[0,0,0])    
        pm.rotate(th,axis,anchor=anch)
        return pm.getB(posS)
    Bc = np.array([[[getB2(phi,th,psi) for phi in PHI] for psi in PSI] for th in TH])

    # vector
    DIM2 = np.array([dim2]*NN)
    Bv = getBv_magnet('sphere',MAG,DIM2,POSm,POSo,[ANG1,ANG2],[AX1,AX2],[ANCH1,ANCH2])
    Bv = Bv.reshape([Nth,Npsi,Nphi,3])

    #assert
    assert np.amax(Bv-Bc) < 1e-10, "bad magpylib vector Sphere"


def test_vectorMagnetCylinder():

    MAG = np.array([[0,0,-44],[0,0,55],[11,22,33],[-14,25,36],[17,-28,39],[-10,-21,32],[0,12,23],[0,-14,25],[16,0,27],[-18,0,29]])
    POSM = np.ones([10,3])
    POSO = MAG*0.1*np.array([.8,-1,-1.3])+POSM
    DIM = np.ones([10,2])

    Bv = getBv_magnet('cylinder',MAG,DIM,POSM,POSO)

    Bc = []
    for mag,posM,posO,dim in zip(MAG,POSM,POSO,DIM):
        pm = Cylinder(mag,dim,posM)
        Bc += [pm.getB(posO)]
    Bc = np.array(Bc)
    
    assert np.amax(abs(Bv-Bc)) < 1e-15

    # inside cylinder testing and iterDia

    MAG = np.array([[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,0,1],[1,1,0],[1,1,1]])
    POSO = np.zeros([7,3])-.1
    DIM = np.ones([7,2])
    POSM = np.zeros([7,3])

    Bv = getBv_magnet('cylinder',MAG,DIM,POSM,POSO,Nphi0=11)

    Bc = []
    for mag,posM,posO,dim in zip(MAG,POSM,POSO,DIM):
        pm = Cylinder(mag,dim,posM,iterDia=11)
        Bc += [pm.getB(posO)]
    Bc = np.array(Bc)
    
    assert np.amax(abs(Bv-Bc)) < 1e-15



def test_vectorMomentDipole():

    MOM = np.array([[0,0,2],[0,0,55],[11,22,33],[-14,25,36],[17,-28,39],[-10,-21,32],[0,12,23],[0,-14,25],[16,0,27],[-18,0,29]])
    POSM = np.ones([10,3])
    POSO = MOM*0.1*np.array([.8,-1,-1.3])+POSM
    
    Bv = getBv_moment('dipole',MOM,POSM,POSO)

    Bc = []
    for mom,posM,posO in zip(MOM,POSM,POSO):
        pm = Dipole(mom,posM)
        Bc += [pm.getB(posO)]
    Bc = np.array(Bc)
    
    assert np.amax(abs(Bv-Bc)) < 1e-15



def test_vectorCurrentCircular():
    
    I = np.ones([10])
    D = np.ones([10])*4
    Pm = np.zeros([10,3])
    Po = np.array([[0,0,1],[0,0,-1],[1,1,0],[1,-1,0],[-1,-1,0],[-1,1,0],[5,5,0],[5,-5,0],[-5,-5,0],[-5,5,0]])

    Bc = []
    for i,d,pm,po in zip(I,D,Pm,Po):
        s = Circular(curr=i,dim=d,pos=pm)
        Bc += [s.getB(po)]
    Bc = np.array(Bc)

    Bv = getBv_current('circular',I,D,Pm,Po)

    assert np.amax(abs(Bc-Bv))<1e-10

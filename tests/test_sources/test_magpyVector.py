#%% MAIN

from magpylib._lib.mathLib_vector import angleAxisRotationV
import numpy as np
import magpylib as magpy
#def test_magpyVector():
 
# calculate the B-field for the 3axis joystick system with
# vector and non-vecor code + compare


#fixed values = base geometry
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

# magpylib base:
def getB(phi,th,psi):
    pm = magpy.source.moment.Dipole(moment=mag,pos=[displM,0,gap])
    axis = magpy.math.axisFromAngles([psi,90])    
    pm.rotate(phi,[0,0,1],anchor=[0,0,0])    
    pm.rotate(th,axis,anchor=anch)
    return pm.getB(posS)


Nphi = 10
Npsi = 100
Nth = 20
NN = Nphi*Npsi*Nth
PHI = np.linspace(0,360,Nphi)
PSI = np.linspace(0,360,Npsi)
TH = np.linspace(0,10,Nth)

Bs = np.array([[[getB(phi,th,psi) for phi in PHI] for th in TH] for psi in PSI])


MAG = np.array([mag]*NN)
DIM = np.array([dim]*NN)
POSo = np.array([posS]*NN)
POSm = np.array([posM]*NN)

ANG1 = np.array(list(PHI)*(Npsi*Nth))
AX1 = np.array([[0,0,1]]*NN)
ANCH1 = np.array([anch]*NN)

ANG2 = np.array([a for a in TH for _ in range(Nphi*Npsi)])
angles = np.array([a for a in PSI for _ in range(Nphi)]*Nth)
AX2 = angleAxisRotationV(angles,np.array([[0,0,1]]*NN),np.array([[1,0,0]]*NN))  
ANCH2 = np.array([anch]*NN)

Bv = magpy.getBv('box',MAG,DIM,POSo,POSm,[ANG1,ANG2],[AX1,AX2],[ANCH1,ANCH2])
B = Bv.reshape([Nth,Npsi,Nphi,3])



import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 8),facecolor='w', dpi=100)
ax = fig.gca(projection='3d')


plt.show()

'''
NN = len(PHI)*len(PSI)*len(TH)
MAG = np.array([mag]*NN)
DIM = np.array([dim]*NN)
POSo = np.array([posM]*NN)
POSm = np.array([posM]*NN)



    temp = np.linspace(0,360,N+1)[:-1]
    ANG1 = np.concatenate((temp,temp,temp,temp,temp))
    AX1 = np.array([[0,0,1]]*(N*5))
    ANCH = np.array([anch]*(5*N))

    P1,Q1 = getPQ(POSm,ANG1,AX1,ANCH)

    ANG2 = np.concatenate((np.zeros(N),np.ones(N)*TA1,np.ones(N)*TA2,np.ones(N)*TA3,np.ones(N)*TA4))
    AX2 = np.array([[0,0,1]]*N + [[1,0,0]]*N + [[-1,0,0]]*N + [[0,1,0]]*N + [[0,-1,0]]*N)

    P2,Q2 = getPQ(P1,ANG2,AX2,ANCH,Q1)

    Bv = getBV(MAG,DIM,P2,Q2,POSo)

'''
'''
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 8),facecolor='w', dpi=100)
ax = fig.gca(projection='3d')

for i in range(10):
    for j in range(3):
        ax.plot(Bs[:,i,j,0],Bs[:,i,j,1],Bs[:,i,j,2])

plt.show()
'''
'''

    MAG = np.array([mag]*(N*5))
    DIM = np.array([dim]*(N*5))
    POSo = np.array([posS]*(N*5))
    POSm = np.array([posM]*(N*5))

    temp = np.linspace(0,360,N+1)[:-1]
    ANG1 = np.concatenate((temp,temp,temp,temp,temp))
    AX1 = np.array([[0,0,1]]*(N*5))
    ANCH = np.array([anch]*(5*N))

    P1,Q1 = getPQ(POSm,ANG1,AX1,ANCH)

    ANG2 = np.concatenate((np.zeros(N),np.ones(N)*TA1,np.ones(N)*TA2,np.ones(N)*TA3,np.ones(N)*TA4))
    AX2 = np.array([[0,0,1]]*N + [[1,0,0]]*N + [[-1,0,0]]*N + [[0,1,0]]*N + [[0,-1,0]]*N)

    P2,Q2 = getPQ(P1,ANG2,AX2,ANCH,Q1)

    Bv = getBV(MAG,DIM,P2,Q2,POSo)

    return np.reshape(Bv,(5,N,3))
'''

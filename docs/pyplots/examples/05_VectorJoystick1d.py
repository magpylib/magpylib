import magpylib as magpy
import numpy as np
import matplotlib.pyplot as plt

import time

# vector size: we calculate the field N times with different inputs
N = 100000  

# Constant vectors
mag  = np.array([0,0,1000])    # magnet magnetization
dim  = np.array([2,2,2])       # magnet dimension
poso = np.array([0,0,0])       # position of observer
posm = np.array([0,0,3])       # initial magnet position
anch = np.array([0,0,8])       # rotation anchor
axis = np.array([1,0,0])       # rotation axis

# different angles for each evaluation
angs = np.linspace(-20,20,N) 

# Vectorizing input using numpy native instead of python loops
MAG = np.tile(mag,(N,1))        
DIM = np.tile(dim,(N,1))        
POSo = np.tile(poso,(N,1))
POSm = np.tile(posm,(N,1))  # inital magnet positions before rotations are applied
ANCH = np.tile(anch,(N,1))  # always same axis
AXIS = np.tile(axis,(N,1))  # always same anchor

# N-times evalulation of the field with different inputs
Bv = magpy.vector.getBv_magnet('box',MAG,DIM,POSo,POSm,[angs],[AXIS],[ANCH])

# plot field
plt.plot(angs,Bv[:,0])
plt.plot(angs,Bv[:,1])
plt.plot(angs,Bv[:,2])

plt.show()

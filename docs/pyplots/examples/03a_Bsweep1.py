import numpy as np
import matplotlib.pyplot as plt
from magpylib.source.magnet import Box
from magpylib import Collection

#create collection of four magnets
s1 = Box(mag=[ 500,0, 500], dim=[3,3,3], pos=[ 0,0, 3], angle=45, axis=[0,1,0])
s2 = Box(mag=[-500,0,-500], dim=[3,3,3], pos=[ 0,0,-3], angle=45, axis=[0,1,0])
s3 = Box(mag=[ 500,0,-500], dim=[4,4,4], pos=[ 4,0, 0], angle=45, axis=[0,1,0])
s4 = Box(mag=[-500,0, 500], dim=[4,4,4], pos=[-4,0, 0], angle=45, axis=[0,1,0])
c = Collection(s1,s2,s3,s4)

#create positions
xs = np.linspace(-8,8,100)
zs = np.linspace(-8,8,100)
posis = [[x,0,z] for z in zs for x in xs]

#calculate fields
Bs = c.getBsweep(posis)

#reshape array and calculate amplitude
Bs = Bs.reshape([100,100,3])
Bamp = np.linalg.norm(Bs,axis=2)

##define figure with 2d and 3d axes
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121,projection='3d')
ax2 = fig.add_subplot(122)

#add displaySystem on ax1
c.displaySystem(subplotAx=ax1,suppress=True)

##amplitude plot on ax2
X,Z = np.meshgrid(xs,zs)
ax2.contourf(X,Z,Bamp,100,cmap='rainbow')

#field plot on ax2
U,V = Bs[:,:,0], Bs[:,:,2]
ax2.streamplot(X, Z, U, V, color='k', density=2)

#display
plt.show()
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
zs = np.linspace(-6,6,100)
posis = [[x,0,z] for z in zs for x in xs]

#calculate fields
Bs = c.getBsweep(posis)

#reshape array and calculate amplitude
Bs = np.array(Bs).reshape([100,100,3])
Bamp = np.linalg.norm(Bs,axis=2)

##amplitude plot
X,Z = np.meshgrid(xs,zs)
plt.contourf(X,Z,Bamp,100,cmap='rainbow')

#field plot
U,V = Bs[:,:,0], Bs[:,:,2]
plt.streamplot(X, Z, U, V, color='k', density=2)

#display
plt.show()
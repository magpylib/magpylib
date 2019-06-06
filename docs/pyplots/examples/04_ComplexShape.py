import numpy as np
import matplotlib.pyplot as plt
from magpylib.source.magnet import Cylinder
from magpylib import Collection

#create collection of four magnets
s1 = Cylinder(mag=[0,0,1000], dim=[5,5])
s2 = Cylinder(mag=[0,0,-1000], dim=[2,6])
c = Collection(s1,s2)

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
plt.pcolor(xs,zs,Bamp,cmap='jet',vmin=-200)

#plot field lines
U,V = Bs[:,:,0], Bs[:,:,2]
plt.streamplot(X,Z,U,V,color='k',density=2)

#display
plt.show()
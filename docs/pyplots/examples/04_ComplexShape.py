import numpy as np
import matplotlib.pyplot as plt
from magpylib.source.magnet import Cylinder
import magpylib as magpy

# create collection of two magnets
s1 = Cylinder(mag=[0,0,1000], dim=[5,5])
s2 = Cylinder(mag=[0,0,-1000], dim=[2,6])
c = magpy.Collection(s1,s2)

# create positions
xs = np.linspace(-8,8,100)
zs = np.linspace(-6,6,100)
posis = [[x,0,z] for z in zs for x in xs]

# calculate field and amplitude
B = [c.getB(pos) for pos in posis]
Bs = np.array(B).reshape([100,100,3]) #reshape
Bamp = np.linalg.norm(Bs,axis=2)

# define figure with a 2d and a 3d axis
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121,projection='3d')
ax2 = fig.add_subplot(122)

# add displaySystem on ax1
magpy.displaySystem(c,subplotAx=ax1,suppress=True)
ax1.view_init(elev=75)

# amplitude plot on ax2
X,Z = np.meshgrid(xs,zs)
ax2.pcolor(xs,zs,Bamp,cmap='jet',vmin=-200)

# plot field lines on ax2
U,V = Bs[:,:,0], Bs[:,:,2]
ax2.streamplot(X,Z,U,V,color='k',density=2)

#display
plt.show()
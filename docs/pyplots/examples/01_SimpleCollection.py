# imports
import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy

# create figure
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

# create magnets
s1 = magpy.source.magnet.Box(mag=[0,0,600],dim=[3,3,3],pos=[-4,0,3])
s2 = magpy.source.magnet.Cylinder(mag=[0,0,500], dim=[3,5])

# manipulate magnets
s1.rotate(45,[0,1,0],anchor=[0,0,0])
s2.move([5,0,-4])

# create collection
c = magpy.Collection(s1,s2)

# display system geometry
magpy.displaySystem(c,subplotAx=ax1,suppress=True)

# calculate B-field on a grid
xs = np.linspace(-10,10,29)
zs = np.linspace(-10,10,29)
Bs = np.array([[c.getB([x,0,z]) for x in xs] for z in zs])

# display field in xz-plane using matplotlib
X,Z = np.meshgrid(xs,zs)
U,V = Bs[:,:,0], Bs[:,:,2]
ax2.streamplot(X, Z, U, V, color=np.log(U**2+V**2),density=2)

plt.show()
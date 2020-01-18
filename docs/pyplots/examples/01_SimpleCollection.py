import numpy as np
import matplotlib.pyplot as plt
from magpylib.source.magnet import Box,Cylinder
from magpylib import Collection, displaySystem

# create magnets
s1 = Box(mag=(0,0,600), dim=(3,3,3), pos=(-4,0,3))
s2 = Cylinder(mag=(0,0,500), dim=(3,5))

# create collection
c = Collection(s1,s2)

# manipulate magnets individually
s1.rotate(45,(0,1,0), anchor=(0,0,0))
s2.move((5,0,-4))

# manipulate collection
c.move((-2,0,0))

# calculate B-field on a grid
xs = np.linspace(-10,10,33)
zs = np.linspace(-10,10,44)
POS = np.array([(x,0,z) for z in zs for x in xs])
Bs = c.getB(POS).reshape(44,33,3)     #<--VECTORIZED

# create figure
fig = plt.figure(figsize=(9,5))
ax1 = fig.add_subplot(121, projection='3d')  # 3D-axis
ax2 = fig.add_subplot(122)                   # 2D-axis

# display system geometry on ax1
displaySystem(c, subplotAx=ax1, suppress=True)

# display field in xz-plane using matplotlib
X,Z = np.meshgrid(xs,zs)
U,V = Bs[:,:,0], Bs[:,:,2]
ax2.streamplot(X, Z, U, V, color=np.log(U**2+V**2))

plt.show()
import numpy as np
import matplotlib.pyplot as plt
import magpylib as mag3

# define Pyplot figure
fig, [ax1,ax2] = plt.subplots(1, 2, figsize=(10,5))

# define Magpylib source
src = mag3.magnet.Cuboid(magnetization=(500,0,500), dimension=(2,2,2))

# create a grid in the xz-symmetry plane
ts = np.linspace(-3, 3, 30)
grid = np.array([[(x,0,z) for x in ts] for z in ts])

# compute B field on grid using a source method
B = src.getB(grid)
ampB = np.linalg.norm(B, axis=2)

#compute H-field on grid using the top-level function
H = mag3.getH(src, grid)
ampH = np.linalg.norm(H, axis=2)

# display field with Pyplot
ax1.streamplot(grid[:,:,0], grid[:,:,2], B[:,:,0], B[:,:,2],
    density=2, color=np.log(ampB), linewidth=1, cmap='autumn')

ax2.streamplot(grid[:,:,0], grid[:,:,2], H[:,:,0], H[:,:,2],
    density=2, color=np.log(ampH), linewidth=1, cmap='winter')

# outline magnet boundary
for ax in [ax1,ax2]:
    ax.plot([1,1,-1,-1,1], [1,-1,-1,1,1], 'k--')

plt.tight_layout()
plt.show()

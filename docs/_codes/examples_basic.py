import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy

# setup matplotlib figure and subplots
fig = plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(131, projection='3d')  # 3D-axis
ax2 = fig.add_subplot(132, projection='3d')  # 3D-axis
ax3 = fig.add_subplot(133)                   # 2D-axis

# define two sources and display in figure
src1 = magpy.magnet.CylinderSegment(magnetization=(0,0,1000), dimension=(2,3,1,-45,45))
src2 = magpy.current.Circular(current=500, diameter=1)
magpy.display(src1, src2, axis=ax1)

# manipulate source position and orientation and display
src2.move((0,0,1))
src1.rotate_from_angax(90, 'y', anchor=0)
magpy.display(src1, src2, axis=ax2)

# create a grid
ts = np.linspace(-4,4,30)
grid = np.array([[(x,0,z) for x in ts] for z in ts])

# compute field on grid
B = magpy.getB([src1,src2], grid, sumup=True)
amp = np.linalg.norm(B, axis=2)

# display field in figure with matplotlib
ax3.streamplot(grid[:,:,0], grid[:,:,2], B[:,:,0], B[:,:,2],
    density=2, color=np.log(amp), linewidth=1, cmap='autumn')

plt.tight_layout()
plt.show()

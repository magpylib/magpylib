import numpy as np
import matplotlib.pyplot as plt
import magpylib as magpy

# create figure using Matplotlib
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121, projection='3d')  # 3D-axis
ax2 = fig.add_subplot(122,)                  # 2D-axis

# create a Magpylib collection of Circular Sources that form a coil
coil = magpy.Collection()
for z in np.linspace(-2,2,20):
    winding = magpy.current.Circular(
        current = 1,
        diameter = 5,
        position = (0,0,z))
    coil += winding

# display the coil on ax1
coil.display(canvas=ax1)

# create a grid
ts = np.linspace(-6,6,30)
grid = np.array([[(x,0,z) for x in ts] for z in ts])

# compute field on grid
B = magpy.getB(coil, grid)
amp = np.linalg.norm(B, axis=2)

# display field in figure with matplotlib
ax2.streamplot(grid[:,:,0], grid[:,:,2], B[:,:,0], B[:,:,2],
    density=2, color=np.log(amp), linewidth=1, cmap='autumn')

plt.tight_layout()
plt.show()

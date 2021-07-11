import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import magpylib as mag3

# setup matplotlib figure
ax = plt.subplot(projection='3d')

src = mag3.magnet.Cuboid(magnetization=(0,0,1000), dimension=(2,2,.5))

# manipulate position by hand
src.position = (-5,0,0)
for _ in range(4):
    mag3.display(src, axis=ax)
    src.position += (0,0,2)

# define source and apply move method
src.position = (0,0,0)
for _ in range(6):
    mag3.display(src, axis=ax)
    src.move((0,0,2))

# define source, move and rotate (use Rotation object) about self
src.position = (5,0,0)
for _ in range(8):
    mag3.display(src, axis=ax)
    src.move((0,0,2))
    src.rotate(R.from_euler('y', np.pi/4))

# define source and rotate (use rotate_from_angax method) about anchor
src.position = (12,0,0)
for _ in range(8):
    mag3.display(src, axis=ax)
    src.rotate_from_angax(-25, 'z', anchor=(0,0,0))

ax.set(xlim=(-12,12), ylim=(-12,12), zlim=(-2,22))
plt.show()

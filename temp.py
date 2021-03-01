import sys
import numpy as np
import magpylib3 as mag3

# o = np.array([[[]]])
# x = np.array([[1,1,1]])
# y = np.array([[1,1,1],[2,2,2]])

# print(o,x,y)
# print(np.r_[[o],x,x,y])
# sys.exit()


pm1 = mag3.magnet.Box((11,22,33),(1,2,3),pos=(-10,0,0))
pm2 = mag3.magnet.Cylinder((0,0,333),(1,2),pos=(10,0,0))
col = mag3.Collection(pm1,pm2)

# with mag3.multi_motion(col, steps=333):
#     col.rotate_from_angax(1111,'z',anchor=0)
#     col.move_by((0,0,15))

# col.display(show_path=True)

sens = mag3.Sensor()
sens.pos_pix = (1,1,1)
sens.rotate_from_angax(33,'z')
sens.move_by((0,1,1))

sens2 = mag3.Sensor(pos=(0,1,0))
sens2.pos_pix = np.array([(2,2,2),(3,3,3),(1,1,1),(-1,-1,-1)])*0.1

sens3 = mag3.Sensor()
ts = np.linspace(-.5,.5,5)
sens3.pos_pix = [(x,y,0) for x in ts for y in ts]
with mag3.multi_motion(sens3, steps=55):
    sens3.move_by((-5,0,-5))
    sens3.rotate_from_angax(455,'x',0)

pm = mag3.magnet.Box((1,1,1),(1,1,1))
with mag3.multi_motion(pm, steps=44):
    pm.move_by((10,0,0))
    pm.rotate_from_angax(666,'z')

pm2 = mag3.magnet.Cylinder((-1,-1,-1),(1,2),(0,0,3))
pm2.rotate_from_angax(45,'y')

mag3.display(sens,sens2,sens3,pm,pm2, markers=[(10,10,10)],direc=True)

pm2.display(direc=True)

# # This import registers the 3D projection, but is otherwise unused.
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# import matplotlib.pyplot as plt
# import numpy as np

# fig = plt.figure()
# ax = fig.gca(projection='3d')

# # Make the grid
# x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
#                       np.arange(-0.8, 1, 0.2),
#                       np.arange(-0.8, 1, 0.8))

# # Make the direction data for the arrows
# u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
# v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
# w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
#      np.sin(np.pi * z))
# print(x)
# ax.quiver([0], [0], [0], [1], [1], [1], length=0.1, normalize=True)

# plt.show()
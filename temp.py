import numpy as np
from numpy.lib.function_base import copy
from scipy.spatial.transform import Rotation as R
import sys
from magpylib3._lib.obj_classes import Box, Cylinder
import magpylib3 as mag3

# pm1 = Box((1,1,1),(1,2,3),pos=(3,0,0))
# pm2 = Box((1,1,1),(1,2,3),pos=(-3,0,0))
# col1 = mag3.Collection(pm1,pm2)
# col1.rotate_from_angax(2222,(0,0,1),anchor=(0,0,0),steps=111)
# col1.move_by((0,20,40),steps=-555)
# col1.rotate_from_angax(100,(0,0,1),anchor=(0,0,0),steps=-111)
# pm1.move_by((0,0,10),steps=-1)
# #pm1.pos = np.r_[pm1.pos, np.array([(0,0,0)])]

# pm1 = Cylinder((1,1,1),(2,3),pos=(15,0,0))
# pm2 = Cylinder((1,1,1),(2,3),pos=(10,0,0))
# col2 = mag3.Collection(pm1,pm2)
# col2.rotate_from_angax(-1111,(0,0,1),anchor=(20,0,0),steps=111)
# col2.move_by((0,0,20),steps=-111)
# col2.rotate_from_angax(-45,(1,0,0),anchor=0,steps=-111)

# mag3.display(col1,col2,show_path=True)

# x = np.array([[1,2,3],[11,22,33]])
# x = np.tile(x,4).reshape(2*4,3)
# print(x)

# sys.exit()

n = 100
s_pos = (0,0,0)
ax = (1,0,0)
anch=(0,0,10)

# path style code translation
pm1 = Cylinder((0,0,1000),(3,3),pos=(-5,0,3))
pm1.move_by((10,0,0),steps=n)
B1 = pm1.getB(s_pos)

# old style code translation
pm2 = Cylinder((0,0,1000),(3,3),pos=(0,0,3))
ts = np.linspace(-5,5,n+1)
possis = np.array([(t,0,0) for t in ts])
B2 = pm2.getB(possis[::-1])

print(np.allclose(B1,B2))


# path style code rotation
pm1 = Cylinder((0,0,1000),(3,3),pos=(0,0,3))
pm1.rotate_from_angax(-30,ax,anch)
pm1.rotate_from_angax(60,ax,anch,steps=n)
B1 = pm1.getB(s_pos)

# old style code rotation
pm2 = Cylinder((0,0,1000),(3,3),pos=(0,0,3))
pm2.rotate_from_angax(-30,ax,anch)
B2 = []
for _ in range(n+1):
    B2 += [pm2.getB(s_pos)]
    pm2.rotate_from_angax(60/n,ax,anch)
B2 = np.array(B2)

print(np.allclose(B1,B2))






# anch = (0,0,1.5+gap+5)
# sens = (0,0,0)
# pm1.rotate_from_angax(-30,(1,0,0),anch)
# pm1.rotate_from_angax(60,(1,0,0),anch,steps=55)
# pm1.display(show_path=True,markers=[anch,sens])

# B = pm1.getB((0,0,0))




# import matplotlib.pyplot as plt
# plt.plot(B)
# plt.show()

#pm1.move_by((1,2,3),steps=15)
#col = mag3.Collection(pm1,pm2)

#mag3.getB([pm1,pm2,col],[0,0,0])

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


pm1 = Cylinder((1,1,1),(2,3))
pm1.pos=(11,22,33)
pm2 = Box((1,1,1),(1,2,3))
pm2.move_by((2,2,2),steps=2)
col = mag3.Collection(pm1,pm2)
B = mag3.getB([pm2,pm2,pm2],[(1,1,1),(1,1,1),(1,1,1),(1,1,1)])
print(B.shape)
print(B)


#pm1.move_by((1,2,3),steps=15)
#col = mag3.Collection(pm1,pm2)

#mag3.getB([pm1,pm2,col],[0,0,0])

import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
from magpylib3._lib.obj_classes import Box
import magpylib3 as mag3

pm1 = Box((1,1,1),(1,2,3),pos=(10,0,0))
pm2 = Box((1,1,1),(1,2,3),pos=(-10,0,0))
pm3 = Box((1,1,1),(1,2,3),pos=(0,10,0))
pm4 = Box((1,1,1),(1,2,3),pos=(0,-10,0))

col = mag3.Collection(pm1,pm2,pm3,pm4)
col.move_by((0,0,30),steps=15)
col.rotate_from_angax(360, (0,0,1), anchor=(0,0,0), steps=-15)
#col.display(show_path=True, direc=False)


pm1 = Box((1,1,1),(1,2,3),pos=(-5,0,0))
pm1.rotate_from_angax(777,(0,0,1),(0,0,0),steps=35)
pm1.move_by((0,0,10),steps=-35)

pm2 = Box((1,1,1),(1,2,3),pos=(-7,0,0))
pm2.move_by((0,0,10),steps=35)
pm2.rotate_from_angax(777,(0,0,1),(0,0,0),steps=-35)

pm3 = Box((1,1,1),(1,2,3),pos=(-9,0,0))
pm3.move_by((0,0,10),steps=35)
pm3.rotate_from_angax(-777,(0,0,1),(0,0,0),steps=-35)

pm4 = Box((1,1,1),(1,2,3),pos=(-11,0,0))
pm4.rotate_from_angax(-777,(0,0,1),(0,0,0),steps=35)
pm4.move_by((0,0,10),steps=-35)

mag3.display(pm1,pm2,pm3,pm4,show_path=True, direc=False)

pm1 = Box((1,1,1),(1,2,3),pos=(5,0,0))
pm2 = Box((1,1,1),(1,2,3),pos=(-5,0,0))
col = mag3.Collection(pm1,pm2)

col.rotate_from_angax(2222,(0,0,1),anchor=(0,0,0),steps=111)
col.move_by((0,20,40),steps=-113)
col.rotate_from_angax(100,(0,0,1),anchor=(0,0,0),steps=-333)
pm1.move_by((0,0,-20),steps=-1)
col.display(show_path=True)



# rr = R.from_rotvec([(0,0,.1),(0,0,.2),(0,0,.3)])
# print(rr.apply(y))

#from magpylib3._lib.obj_classes.class_BaseGeo import BaseGeo

# rr = R.from_quat([(0,0,0,1),(1,0,0,1),(0,1,0,1)])
# r = R.from_quat((0,0,1,1))

# #print((rr*r).as_quat())
# #print(rr.as_quat())
# #print((rr*rr).as_quat())
# #print((rr*rr*rr).as_quat())

# #print(rr._quat)
# rr._quat = np.r_[rr._quat,rr._quat]
# #print(rr._quat)

# vec = np.array([1,2,3])
# print(rr.apply(vec))

# pm1 = Box((1,1,1),(1,2,3),pos=(10,0,0))
# pm2 = Box((1,1,1),(1,2,3),pos=(-10,0,0))
# pm3 = Box((1,1,1),(1,2,3),pos=(0,10,0))
# pm4 = Box((1,1,1),(1,2,3),pos=(0,-10,0))

# for pm in [pm1,pm2,pm3,pm4]:
#     pm.move_by((0,0,15),steps=20)
#     pm.rotate(R.from_rotvec((0,0,3)),anchor=(0,0,0),steps=-20)
#     pm.move_by((0,0,15),steps=20)
#     pm.rotate_from_angax(170,(0,0,1),anchor=(0,0,0),steps=-20)

#mag3.display(pm1,pm2,pm3,pm4,show_path=True)

#pm.move_by((0,5,5),steps=5)
#pm.rotate(R.from_rotvec((0,0,3)),anchor=(0,0,0),steps=13)

# print(pm.pos)
# print(pm.rot.as_rotvec())


# a=np.array([1,2,3,4,5])
# b=np.array([1,2,3])

# print(b*np.tile(a,(3,1)).T[1:])



# #print(bg._pos)
# #print(bg._rot.as_quat())
# #print('---')
# pm.rotate(R.from_quat([.1,0,0,.1]),steps=5,anchor=(1,1,1))
# #print(bg.pos)
## #print(bg.rot.as_rotvec())
# #print('---')
# pm.rotate(R.from_quat([.1,0,0,.1]),steps=5,anchor=0)
# #print(bg.pos)
# #print(bg.rot.as_rotvec())


# import magpylib3 as mag3
# # pm1 = Box((1,1,1),(.1,.2,.3))
# # pm1.move_by((1,1,1),steps=10)

# pm2 = Box((1,1,1),(.5,1,1.5),pos=(0,0,10))

# pm2.move_by((10,10,10),steps=10)
# print(pm2.pos)

# pm2.move_to((10,10,10),steps=-5)
# print(pm2.pos)

# pm2.move_to((10,10,10),steps=-12)
# print(pm2.pos)

# pm2.display(show_path=True)

# mag3.display(pm1,show_path=True)
# mag3.display(pm2,show_path=True)


# col = mag3.Collection()
# for p,o in zip(pm.pos, pm.rot):
#     pm = Box((1,1,1),(.1,.2,.3),pos=p,rot=o)
#     col + pm

# pm2 = mag3.magnet.Cylinder((1,1,1),(.2,.3),pos=(1,1,1))
# pm2.move_to((3,0,0),steps=5)
# pm2.rotate(R.from_quat([.1,0,.1,.1]),steps=15,anchor=0)
# pm2.rotate(R.from_quat([.1,0,.1,.1]),steps=15,anchor=0)

# mag3.display(col,pm2,show_path=False)
# mag3.display(col,pm2,show_path=True)
# mag3.display(col,pm2,show_path='all')





# print(bg._rot.as_quat())
# print(bg.rot.as_quat())

# print('tick')
# bg.rot = R.from_rotvec([1,2,3])
# print(bg._rot.as_quat())
# print(bg.rot.as_quat())

# print('tick')
# bg.rot = R.from_rotvec([[1,2,3],[1,2,3]])
# print(bg._rot.as_quat())
# print(bg.rot.as_quat())

# rr = R.from_rotvec([1,2,3])
# rr = R.from_rotvec([[1,2,3],[1,2,3],[2,3,4]])
# print(len(rr))
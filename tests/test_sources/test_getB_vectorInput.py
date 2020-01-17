import numpy as np
import time
import magpylib as magpy

POS = np.array([[1,2,3],[2,-3,4],[-3,4,5],[5,6,-7],[-3,-2,1],[-4,3,-2],[5,-4,-3],[-6,-5,-4]])

s = magpy.source.current.Circular(curr=123,dim=2,pos=(2,-4,5),angle=23,axis=(.2,-5,3))
B = np.array([s.getB(p) for p in POS])
Bv = s.getB(POS)
err = np.amax(np.abs(Bv-B))
assert err < 1e-12

VERT = POS*.333
s = magpy.source.current.Line(curr=123,vertices=VERT)
B = np.array([s.getB(p) for p in POS])
Bv = s.getB(POS)
err = np.amax(np.abs(Bv-B))
assert err < 1e-12

s = magpy.source.moment.Dipole(moment=(.1,-5,5.5),pos=(2,-4,5),angle=33,axis=(.2,-5,3))
B = np.array([s.getB(p) for p in POS])
Bv = s.getB(POS)
err = np.amax(np.abs(Bv-B))
assert err < 1e-12

s = magpy.source.magnet.Box(mag=(33,22,111),dim=(3,2,1),pos=(2,-4,5),angle=33,axis=(.2,-5,3))
B = np.array([s.getB(p) for p in POS])
Bv = s.getB(POS)
err = np.amax(np.abs(Bv-B))
assert err < 1e-12

s = magpy.source.magnet.Cylinder(mag=(33,22,111),dim=(3,1),pos=(2,-4,5),angle=33,axis=(.2,-5,3))
B = np.array([s.getB(p) for p in POS])
Bv = s.getB(POS)
err = np.amax(np.abs(Bv-B))
assert err < 1e-12

s = magpy.source.magnet.Sphere(mag=(33,22,111),dim=.123,pos=(2,-4,5),angle=33,axis=(.2,-5,3))
B = np.array([s.getB(p) for p in POS])
Bv = s.getB(POS)
err = np.amax(np.abs(Bv-B))
assert err < 1e-12
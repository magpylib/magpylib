# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 13:28:12 2019

@author: Gabriel
"""
import magPyLib as magPy
pm = magPy.magnet.Sphere(mag=[0,0,1000], dim=1)
print(pm.position, pm.angle, pm.axis)
pm.rotate(90, [0,1,0], CoR=[1,0,0])
print([pm.position, pm.angle, pm.axis])
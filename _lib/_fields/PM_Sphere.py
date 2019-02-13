# -*- coding: utf-8 -*-

# MAGNETIC FIELD CALCULATION FOR SPHERICAL MAGNET IN CANONICAL BASIS



#%% IMPORTS
from magPyLib.math._mathLibPrivate import fastNorm3D, fastSum3D


#%% CALCULATION

# The magnetic field of a spherical magnet with the center in the origin

# MAG = magnetization vector     [mT]  - takes arr3
# pos = position of the observer [mm]  - takes arr3
# D = diameter of sphere         [mm]  - takes float  

def Bfield_Sphere(MAG, pos, D): #returns array, takes (arr3, arr3, float)
    
    radius = D/2
    r = fastNorm3D(pos)

    if r > radius:
        return radius**3/3*(-MAG/r**3 + 3*fastSum3D(MAG*pos)*pos/r**5)
    else:
        return 2/3*MAG
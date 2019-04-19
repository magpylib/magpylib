#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from magpylib._lib.mathLibPrivate import fastNorm3D, fastSum3D
from numpy import pi,dot,array,NaN


#%% DIPOLE field

# describes the field of a dipole positioned at posM and pointing into the direction of M

# M    : arr3  [mT]    Magnetic moment, M = µ0*m
# pos  : arr3  [mm]    Position of observer
# posM : arr3  [mm]    Position of dipole moment

# |M| corresponds to the magnetic moment of a cube with remanence Br and Volume V such that
#       |M| [mT*mm^3]  =  Br[mT] * V[mm^3]

def Bfield_Dipole(M,pos):
    R = pos
    rr = fastSum3D(R*R)
    mr = fastSum3D(M*R)
    
    if rr == 0:
        print('Warning: getB Position directly on magnet surface')
        return array([NaN,NaN,NaN])
    
    return (3*R*mr-M*rr)/rr**(5/2)/(4*pi)
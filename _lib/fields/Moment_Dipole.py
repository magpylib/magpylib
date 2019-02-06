#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from numpy import pi,dot


#%% DIPOLE field

# describes the field of a dipole positioned at posM and pointing into the direction of M

# M    : arr3  [mT]    Magnetic moment, M = µ0*m
# pos  : arr3  [mm]    Position of observer
# posM : arr3  [mm]    Position of dipole moment


def Bfield_Dipole(M,pos,posM):
    R = posM-pos
    rr = dot(R,R)
    mr = dot(M,R)
    return (3*R*mr-M*rr)/rr**(5/2)/(4*pi)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from numpy import array
from magPyLib._lib.mathLibPrivate import fastSum3D, fastNorm3D, fastCross3D


#%% CURRENT LINE
# Describes the magnetic field of a line current. The line is given by a set of
#   data points and the field is the superopsition of the fields of all linear
#   segments.

# i0   : float       [A]       current lowing from the first point to the last point
# pos  : arr3 float  [mm]      Position of observer
# possis: list or arr of arr3floats [mm]    Positions that define the line
    
#source: http://www.phys.uri.edu/gerhard/PHY204/tsl216.pdf
# FieldOfLineCurrent


# observer at p0
# current I0 flows in straight line from p1 to p2
def Bfield_LineSegment(p0,p1,p2,I0):
    #must receive FLOAT only !!!!
    
    #projection of p0 onto line p1-p2
    p4 = p1+(p1-p2)*fastSum3D((p0-p1)*(p1-p2))/fastSum3D((p1-p2)*(p1-p2))

    #determine correct normal vector to surface spanned by triangle
    cross0 = fastCross3D(p2-p1,p0-p4)
    norm_cross0 = fastNorm3D(cross0)
    if norm_cross0 != 0.:
        eB = cross0/norm_cross0
    else:
        print('ERROR: Bline - p0, p1, p2 do not span triangle')
        sys.exit()

    #determine sinTHs and R
    norm_04 = fastNorm3D(p0-p4) # =R
    norm_01 = fastNorm3D(p0-p1)
    norm_02 = fastNorm3D(p0-p2)
    norm_12 = fastNorm3D(p1-p2)
    norm_41 = fastNorm3D(p4-p1)
    norm_42 = fastNorm3D(p4-p2)
    
    sinTh1 = norm_41/norm_01
    sinTh2 = norm_42/norm_02
    
    #determine how p1,p2,p4 are sorted on the line (to get sinTH signs)
    if norm_41 > norm_12 and norm_41 > norm_42: # both points below
        deltaSin = abs(sinTh1-sinTh2)
    elif norm_42 > norm_12 and norm_42 > norm_41:#both points above
        deltaSin = abs(sinTh2-sinTh1)
    else: #one above one below or one equals p4
        deltaSin = abs(sinTh1+sinTh2)

    B = I0*deltaSin/norm_04*eB/10 # missing 10**-6 from m->mm conversion #T->mT conversion
    
    return B

#determine total field from multiple segments
def Bfield_CurrentLine(p0,possis,I0):
    
    B = array([0.,0.,0.])
    for i in range(len(possis)-1):
        p1 = possis[i]
        p2 = possis[i+1]
        B += Bfield_LineSegment(p0,p1,p2,I0)
    return B



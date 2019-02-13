#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from numpy import sqrt,array,cos,sin
from magPyLib.math._mathLibPrivate import getPhi,ellipticK, ellipticE

#%% CIRCULAR CURRENT LOOP
# Describes the magnetic field of a circular current loop that lies parallel to
#    the x-y plane. The loop has the radius r0 and carries the current i0, the 
#    center of the current loop lies at posCL

# i0   : float  [A]     current in the loop
# d0   : float  [mm]    diameter of the current loop
# posCL: arr3  [mm]     Position of the center of the current loop
    
#source: calculation from Filipitsch Diplo

def Bfield_CircularCurrentLoop(i0,d0,pos):

    px,py,pz = pos

    r = sqrt(px**2+py**2)
    phi = getPhi(px,py)
    z = pz
    
    r0 = d0/2  #radius of current loop

    #avoid singularity at CL
    if abs(r-r0)<1e-10 and abs(z)<1e-10:
        print('WARNING: close to singularity - setting field to zero')
        return array([0,0,0])

    deltaP = sqrt((r+r0)**2+z**2)
    deltaM = sqrt((r-r0)**2+z**2)
    kappa = deltaP**2/deltaM**2
    kappaBar=1-kappa

    #avoid discontinuity at r=0
    if r<1e-10: 
        Br = 0.
    else:
        Br = -2*1e-7*i0*(z/r/deltaM)*(ellipticK(kappaBar)-(2-kappaBar)/(2-2*kappaBar)*ellipticE(kappaBar))    *1e3           #account for mm input
    Bz = -2*1e-7*i0*(1/deltaM)*(-ellipticK(kappaBar)+(2-kappaBar-4*(r0/deltaM)**2)/(2-2*kappaBar)*ellipticE(kappaBar))  *1e3 #account for mm input

    #transfer to cartesian coordinates
    Bcy = array([Br,0.,Bz])*1000. #mT output
    T_Cy_to_Kart =  array([[cos(phi),-sin(phi),0],[sin(phi),cos(phi),0],[0,0,1]])
    Bkart = T_Cy_to_Kart.dot(Bcy)

    return Bkart


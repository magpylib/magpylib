# -*- coding: utf-8 -*-

# MAGNETIC FIELD CALCULATION OF CUBE IN CANONICAL BASIS



#%% IMPORTS
from numpy import pi,sign,sqrt,log,array,arctan



#%% CALCULATIONS

# Describes the magnetic field of a rectangular cuboid magnet with sides parallel to its native
#   cartesian coordinates. The dimension is 2a x 2b x 2c and the magnetization 
#   is given by MAG. The center of the box is positioned at posM.

# MAG : arr3   [mT/mm³]     Magnetization per unit volume, MAG = mu0*mag = remanence field
# pos  : arr3  [mm]        Position of observer
# dim  : arr3  [mm]        dim = [A,B,C], Magnet dimension = A x B x C

#basic functions required to calculate the cuboid's fields
def F1(x,y,z,a,b,c):
    if z+c == 0:
        return pi/2*sign((x+a)*(y+b))
    return arctan( (x+a)*(y+b) / (z+c)/sqrt((x+a)**2 + (y+b)**2 + (z+c)**2) )   #SMOOTHEN THIS?!!!!

def F2(x,y,z,a,b,c):
    X = (sqrt((x+a)**2 + (y-b)**2 + (z+c)**2) + b - y)
    Y = (sqrt((x+a)**2 + (y+b)**2 + (z+c)**2) - b - y)
    if Y == 0 and X == 0: #verhindere division by zero und log(0)
        #ziehe den grenzwert anstelle von leichter Verschiebeung
        x += 0.00000000123
        y += 0.00000000223
        z += 0.00000000323
        X = (sqrt((x+a)**2 + (y-b)**2 + (z+c)**2) + b - y)
        Y = (sqrt((x+a)**2 + (y+b)**2 + (z+c)**2) - b - y)
    return X/Y

    #if Y != 0 and X!=0: 
    #    return X/Y
    #else: #verschiebung ist eine sehr unsaubere methode!!!!                MUSS VERBESSERT WERDEN!!!!!
    #    x += 0.00000000123
    #    y += 0.00000000223
    #    z += 0.00000000323
    #    return (sqrt((x+a)**2 + (y-b)**2 + (z+c)**2) + b - y) / (sqrt((x+a)**2 + (y+b)**2 + (z+c)**2) - b - y)

#calculating the field
def Bfield_Box(MAG, pos, dim): #returns arr3

    # Mag part in z-direction
    B0 = MAG[2]
    x,y,z = pos
    a,b,c = dim/2

    BxZ = B0/4/pi*log( F2(-x,y,-z,a,b,c) * F2(x,y,z,a,b,c) / F2(x,y,-z,a,b,c) / F2(-x,y,z,a,b,c))
    ByZ = -B0/4/pi*log( F2(-y,-x,-z,b,-a,c) * F2(y,-x,z,b,-a,c) / F2(y,-x,-z,b,-a,c) / F2(-y,-x,z,b,-a,c))
    BzZ = -B0/4/pi*(F1(-x,y,z,a,b,c) + F1(-x,y,-z,a,b,c) + F1(-x,-y,z,a,b,c) + F1(-x,-y,-z,a,b,c) + F1(x,y,z,a,b,c) +  F1(x,y,-z,a,b,c)  + F1(x,-y,z,a,b,c)  + F1(x,-y,-z,a,b,c))
    field = array([BxZ,ByZ,BzZ])

    # Mag part in x-direction - we use the solution for mag in z-direction and do a coordinate transformation
    B0 = MAG[0]
    x, y, z = -z, y, x
    a, b, c = c, b, a

    BzX = -B0/4/pi*log( F2(-x,y,-z,a,b,c) * F2(x,y,z,a,b,c) / F2(x,y,-z,a,b,c) / F2(-x,y,z,a,b,c))
    ByX = -B0/4/pi*log( F2(-y,-x,-z,b,-a,c) * F2(y,-x,z,b,-a,c) / F2(y,-x,-z,b,-a,c) / F2(-y,-x,z,b,-a,c))
    BxX = -B0/4/pi*(F1(-x,y,z,a,b,c) + F1(-x,y,-z,a,b,c) + F1(-x,-y,z,a,b,c) + F1(-x,-y,-z,a,b,c) + F1(x,y,z,a,b,c) +  F1(x,y,-z,a,b,c)  + F1(x,-y,z,a,b,c)  + F1(x,-y,-z,a,b,c))
    field += array([BxX,ByX,BzX])

    # Mag part in y-direction - we use the solution for mag in z-direction and do a coordinate transformation
    B0 = MAG[1]
    x, y, z = x, -z, y
    a, b, c = a, c, b

    BzY = -B0/4/pi*log( F2(-x,y,-z,a,b,c) * F2(x,y,z,a,b,c) / F2(x,y,-z,a,b,c) / F2(-x,y,z,a,b,c))
    BxY = B0/4/pi*log( F2(-y,-x,-z,b,-a,c) * F2(y,-x,z,b,-a,c) / F2(y,-x,-z,b,-a,c) / F2(-y,-x,z,b,-a,c))
    ByY = -B0/4/pi*(F1(-x,y,z,a,b,c) + F1(-x,y,-z,a,b,c) + F1(-x,-y,z,a,b,c) + F1(-x,-y,-z,a,b,c) + F1(x,y,z,a,b,c) +  F1(x,y,-z,a,b,c)  + F1(x,-y,z,a,b,c)  + F1(x,-y,-z,a,b,c))
    field += array([BxY,ByY,BzY])

    # add M when inside the box to make B out of H
    if abs(x) < a:
        if abs(y) < b:
            if abs(z) < c:
                field += MAG

    return field

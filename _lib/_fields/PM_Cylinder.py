# -*- coding: utf-8 -*-

# MAGNETIC FIELD CALCULATION OF CYLINDER IN CANONICAL BASIS



#%% IMPORTS
from numpy import pi,sqrt,array,arctan,cos,sin,arange
from magPyLib.math._mathLibPrivate import getPhi, elliptic


#%% Cylinder Field Calculation
# Describes the magnetic field of a cylinder with circular top and bottom and
#   arbitrary magnetization given by MAG. The axis of the cylinder is parallel
#   to the z-axis. The dimension are given by the radius r and the height h.
#   The center of the cylinder is positioned at the origin.

#basic functions required to calculate the diametral contributions
def Sphi(n,Nphi):
    if      n==0: return 1./3
    elif    n==Nphi: return 1./3
    elif    n%2 == 1: return 4./3
    elif    n%2 == 0: return 2./3

def I1(r,phi,z,r0,phi0,z0):
    if r**2+r0**2-2*r*r0*cos(phi-phi0) == 0:
        return -1/2/(z-z0)**2
    else:
        G = 1/sqrt(r**2+r0**2-2*r*r0*cos(phi-phi0)+(z-z0)**2)
        return (z-z0)*G/(r**2+r0**2-2*r*r0*cos(phi-phi0))


# MAG  : arr3  [mT/mm³]    Magnetization vector (per unit volume)
# pos  : arr3  [mm]        Position of observer
# dim  : arr3  [mm]        dim = [d,h], Magnet diameter r and height h

#this calculation returns the B-field from the statrt as it is based on a current equivalent
def Bfield_Cylinder(MAG, pos, dim, Nphi0): #returns arr3

    D,H = dim                # magnet dimensions
    R = D/2

    x,y,z = pos       # relative position
    r, phi = sqrt(x**2+y**2), getPhi(x,y)      # cylindrical coordinates
    
    # Mag part in z-direction
    B0z = MAG[2]     #z-part of magnetization
    zP,zM = z+H/2.,z-H/2.   # some important quantitites

    # get Br
    alphP = R/sqrt(zP**2+(r+R)**2)
    alphM = R/sqrt(zM**2+(r+R)**2)
    kP = sqrt((zP**2+(R-r)**2)/(zP**2+(R+r)**2))
    kM = sqrt((zM**2+(R-r)**2)/(zM**2+(R+r)**2))
    Br_Z = B0z*(alphP*elliptic(kP,1,1,-1)-alphM*elliptic(kM,1,1,-1))/pi

    Bx_Z = Br_Z*cos(phi)
    By_Z = Br_Z*sin(phi)

    # get Bz
    betP = zP/sqrt(zP**2+(r+R)**2)
    betM = zM/sqrt(zM**2+(r+R)**2)
    gamma = (R-r)/(R+r)
    kP = sqrt((zP**2+(R-r)**2)/(zP**2+(R+r)**2))
    kM = sqrt((zM**2+(R-r)**2)/(zM**2+(R+r)**2))
    Bz_Z = B0z*R/(R+r)*(betP*elliptic(kP,gamma**2,1,gamma)-betM*elliptic(kM,gamma**2,1,gamma))/pi

    Bfield=array([Bx_Z, By_Z, Bz_Z])
    
    
    # Mag part in xy-direction
    B0xy = sqrt(MAG[0]**2+MAG[1]**2)  #xy-magnetization amplitude
    if B0xy > 0:
        
        if MAG[0] > 0.:
            tetta = arctan(MAG[1]/MAG[0])
        elif MAG[0] < 0.:
            tetta = arctan(MAG[1]/MAG[0])+pi
        elif MAG[1] > 0:
            tetta = pi/2
        else:
            tetta = 3*pi/2
           
        if x > 0.:
            gamma = arctan(y/x)
        elif x < 0.:
            gamma = arctan(y/x)+pi
        elif y > 0:
            gamma = pi/2
        else:
            gamma = 3*pi/2 
        phi=gamma-tetta
        
            
        #R,H = dim
        phi0s = 2*pi/Nphi0
        Br_XY = B0xy*R/2/Nphi0*sum([
            sum([
                (-1)**(k+1)*Sphi(n,Nphi0)*cos(phi0s*n)*(r-R*cos(phi-phi0s*n))*I1(r,phi,z,R,phi0s*n,z0)
                for z0,k in zip([-H/2,H/2],[1,2])])
            for n in arange(Nphi0+1)])
        
        Bphi_XY = B0xy*R**2/2/Nphi0*sum([
            sum([
                (-1)**(k+1)*Sphi(n,Nphi0)*cos(phi0s*n)*sin(phi-phi0s*n)*I1(r,phi,z,R,phi0s*n,z0)
                for z0,k in zip([-H/2,H/2],[1,2])])
            for n in arange(Nphi0+1)])
        
        Bz_XY = B0xy*R/2/Nphi0*sum([
            sum([
                (-1)**k*Sphi(n,Nphi0)*cos(phi0s*n)/sqrt(r**2+R**2-2*r*R*cos(phi-phi0s*n)+(z-z0)**2)
                for z0,k in zip([-H/2,H/2],[1,2])])
            for n in arange(Nphi0+1)])
            
        phi=gamma
        Bx_XY = Br_XY*cos(phi)-Bphi_XY*sin(phi)
        By_XY = Br_XY*sin(phi)+Bphi_XY*cos(phi)

  
        Bfield = Bfield + array([Bx_XY, By_XY, Bz_XY])
        
        # add M if inside the cylinder to make B out of H :)
        if r < R and abs(z) < H/2:
            Bfield += array([MAG[0],MAG[1],0])
    
    return  Bfield


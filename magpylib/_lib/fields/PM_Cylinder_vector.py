# -------------------------------------------------------------------------------
# magpylib -- A Python 3 toolbox for working with magnetic fields.
# Copyright (C) Silicon Austria Labs, https://silicon-austria-labs.com/,
#               Michael Ortner <magpylib@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License along
# with this program.  If not, see <https://www.gnu.org/licenses/>.
# The acceptance of the conditions of the GNU Affero General Public License are
# compulsory for the usage of the software.
#
# For contact information, reach out over at <magpylib@gmail.com> or our issues
# page at https://www.github.com/magpylib/magpylib/issues.
# -------------------------------------------------------------------------------

# MAGNETIC FIELD CALCULATION OF CYLINDER IN CANONICAL BASIS

# VECTORIZED VERSION

# %% IMPORTS
from numpy import pi, sqrt, array, arctan2, cos, sin
import numpy as np
from magpylib._lib.mathLib_vector import celV

# Describes the magnetic field of a cylinder with circular top and bottom and
#   arbitrary magnetization given by MAG. The axis of the cylinder is parallel
#   to the z-axis. The dimension are given by the radius r and the height h.
#   The center of the cylinder is positioned at the origin.

# basic functions required to calculate the diametral contributions



# MAG  : arr3  [mT/mm³]    Magnetization vector (per unit volume)
# pos  : arr3  [mm]        Position of observer
# dim  : arr3  [mm]        dim = [d,h], Magnet diameter r and height h

# this calculation returns the B-field from the statrt as it is based on a current equivalent
def Bfield_CylinderV(MAG, POS, DIM, Nphi0):  # returns arr3

    D = DIM[:,0]                # magnet dimensions
    H = DIM[:,1]                # magnet dimensions
    R = D/2

    N = len(D)  # vector size

    X,Y,Z = POS[:,0],POS[:,1],POS[:,2]

    ### BEGIN AXIAL MAG CONTRIBUTION ###########################
    RR, PHI = sqrt(X**2+Y**2), arctan2(Y, X)      # cylindrical coordinates
    B0z = MAG[:,2]              # z-part of magnetization
    
    # some important quantitites
    zP, zM = Z+H/2., Z-H/2.   
    Rpr, Rmr = R+RR, R-RR

    SQ1 = sqrt(zP**2+Rpr**2)
    SQ2 = sqrt(zM**2+Rpr**2)

    alphP = R/SQ1
    alphM = R/SQ2
    betP = zP/SQ1
    betM = zM/SQ2
    kP = sqrt((zP**2+Rmr**2)/(zP**2+Rpr**2))
    kM = sqrt((zM**2+Rmr**2)/(zM**2+Rpr**2))
    gamma = Rmr/Rpr

    one = np.ones(N)

    # radial field
    Br_Z = B0z*(alphP*celV(kP, one, one, -one)-alphM*celV(kM, one, one, -one))/pi
    Bx_Z = Br_Z*cos(PHI)
    By_Z = Br_Z*sin(PHI)

    # axial field
    Bz_Z = B0z*R/(Rpr)*(betP*celV(kP, gamma**2, one, gamma) -
                        betM*celV(kM, gamma**2, one, gamma))/pi

    Bfield = np.c_[Bx_Z, By_Z, Bz_Z]  # contribution from axial magnetization

    ### BEGIN TRANS MAG CONTRIBUTION ###########################

    # Mag part in xy-direction requires a numerical algorithm
    # mask0 selects only input values where xy-MAG is non-zero
    B0xy = sqrt(MAG[:,0]**2+MAG[:,1]**2)
    mask0 = (B0xy > 0.) # finite xy-magnetization mask    
    N0 = np.sum(mask0)  #number of masked values

    if N0 >= 1:
        
        tetta = arctan2(MAG[mask0,1],MAG[mask0,0])
        gamma = arctan2(Y[mask0],X[mask0])
        phi = gamma-tetta

        phi0s = 2*pi/Nphi0  # discretization

        # prepare masked arrays for use in algorithm

        RR_m0 = RR[mask0]
        R_m0 = R[mask0]        
        rR2 = 2*R_m0*RR_m0
        r2pR2 = R_m0**2+RR_m0**2
        Z0_m0 = H[mask0]/2
        Z_m0 = Z[mask0]
        H_m0 = H[mask0]

        Sphi = np.arange(Nphi0+1)
        Sphi[Sphi%2==0] = 2.
        Sphi[Sphi%2==1] = 4.
        Sphi[0] = 1.
        Sphi[-1] = 1.

        SphiE = np.outer(Sphi,np.ones(N0))

        I1xE = np.ones([Nphi0+1,N0])
        phi0E = np.outer(np.arange(Nphi0+1),np.ones(N0))*phi0s

        Z_m0E =  np.outer(np.ones(Nphi0+1),Z_m0)
        Z0_m0E = np.outer(np.ones(Nphi0+1),Z0_m0)
        phiE =   np.outer(np.ones(Nphi0+1),phi)
        rR2E =   np.outer(np.ones(Nphi0+1),rR2)
        r2pR2E = np.outer(np.ones(Nphi0+1),r2pR2)

        # parts for multiple use
        cosPhi = cos(phiE-phi0E)
        
        # calc R-PHI components
        ma = (r2pR2E-rR2E*cosPhi == 0)
        I1xE[ma] = - (1/2)/(Z_m0E[ma]+Z0_m0E[ma])**2 + (1/2)/(Z_m0E[ma]-Z0_m0E[ma])**2

        nMa = np.logical_not(ma)
        rrc = r2pR2E[nMa]-rR2E[nMa]*cosPhi[nMa]
        Gm = 1/sqrt(rrc+(Z_m0E[nMa]+Z0_m0E[nMa])**2)
        Gp = 1/sqrt(rrc+(Z_m0E[nMa]-Z0_m0E[nMa])**2)
        I1xE[nMa] = ((Z_m0E+Z0_m0E)[nMa]*Gm-(Z_m0E-Z0_m0E)[nMa]*Gp)/rrc

        Summand = SphiE/3.*cos(phi0E)*I1xE

        Br_XY_m0   = B0xy[mask0]*R_m0/2/Nphi0*np.sum(Summand*(RR_m0-R_m0*cosPhi),axis=0)
        Bphi_XY_m0 = B0xy[mask0]*R_m0**2/2/Nphi0*np.sum(Summand*sin(phiE-phi0E),axis=0)

        # calc Z component
        Gzm = 1./sqrt(r2pR2-rR2*cosPhi+(Z_m0E+H_m0/2)**2)
        Gzp = 1./sqrt(r2pR2-rR2*cosPhi+(Z_m0E-H_m0/2)**2)
        SummandZ = SphiE/3.*cos(phi0E)*(Gzp-Gzm)
        Bz_XY_m0 = B0xy[mask0]*R_m0/2/Nphi0*np.sum(SummandZ,axis=0)

        # translate r,phi to x,y coordinates
        Bx_XY_m0 = Br_XY_m0*cos(gamma)-Bphi_XY_m0*sin(gamma)
        By_XY_m0 = Br_XY_m0*sin(gamma)+Bphi_XY_m0*cos(gamma)

        BfieldTrans = array([Bx_XY_m0, By_XY_m0, Bz_XY_m0]).T
        
        # add field from transversal mag to field from axial mag
        Bfield[mask0] += BfieldTrans

        # add M if inside the cylinder to make B out of H
        mask0Inside = mask0 * (RR<R) * (abs(Z)<H/2)
        Bfield[mask0Inside,:2] += MAG[mask0Inside,:2]
    
    ### END TRANS MAG CONTRIBUTION ###########################

    return(Bfield)



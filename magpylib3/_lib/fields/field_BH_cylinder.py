"""
Implementations of analytical expressions for the magnetic field of homogeneously magnetized Cylinders.
Computation details in function docstrings.
"""

import numpy as np
from magpylib3._lib.math_utility.special_functions import celv

def field_BH_cylinder(bh, mag, dim, pos_obs, niter):
    """ Wrapper function to select cylinder B- or H-field, which are treated equally
    at higher levels

    ### Args:
    - bh (boolean): True=B, False=H
    - mag (ndarray Nx3): homogeneous magnetization vector in units of mT
    - dim (ndarray Nx2): dimension of Cylinder side lengths in units of mm
    - pos_obs (ndarray Nx3): position of observer in units of mm
    - niter (int): number of iterations for diametral component

    ### Returns:
    - B/H-field (ndarray Nx3): magnetic field vectors at pos_obs in units of mT / kA/m
    """
    if bh:
        return field_B_cylinder(mag, dim, pos_obs, niter)
    else:
        return field_H_cylinder(mag, dim, pos_obs, niter)


# this calculation returns the B-field from the statrt as it is based on a current equivalent
def field_B_cylinder(MAG, DIM, POS, niter):  # returns arr3
    """[summary]

    Args:
        MAG ([type]): [description]
        POS ([type]): [description]
        DIM ([type]): [description]
        niter ([type]): [description]
    """

    D = DIM[:,0]                # magnet dimensions
    H = DIM[:,1]                # magnet dimensions
    R = D/2

    N = len(D)  # vector size

    X,Y,Z = POS[:,0],POS[:,1],POS[:,2]

    ### BEGIN AXIAL MAG CONTRIBUTION ###########################
    RR, PHI = np.sqrt(X**2+Y**2), np.arctan2(Y, X)      # cylindrical coordinates
    B0z = MAG[:,2]              # z-part of magnetization
    
    # some important quantitites
    zP, zM = Z+H/2., Z-H/2.   
    Rpr, Rmr = R+RR, R-RR

    SQ1 = np.sqrt(zP**2+Rpr**2)
    SQ2 = np.sqrt(zM**2+Rpr**2)

    alphP = R/SQ1
    alphM = R/SQ2
    betP = zP/SQ1
    betM = zM/SQ2
    kP = np.sqrt((zP**2+Rmr**2)/(zP**2+Rpr**2))
    kM = np.sqrt((zM**2+Rmr**2)/(zM**2+Rpr**2))
    gamma = Rmr/Rpr

    one = np.ones(N)

    # radial field
    Br_Z = B0z*(alphP*celv(kP, one, one, -one)-alphM*celv(kM, one, one, -one))/np.pi
    Bx_Z = Br_Z*np.cos(PHI)
    By_Z = Br_Z*np.sin(PHI)

    # axial field
    Bz_Z = B0z*R/(Rpr)*(betP*celv(kP, gamma**2, one, gamma) -
                        betM*celv(kM, gamma**2, one, gamma))/np.pi

    Bfield = np.c_[Bx_Z, By_Z, Bz_Z]  # contribution from axial magnetization

    ### BEGIN TRANS MAG CONTRIBUTION ###########################

    # Mag part in xy-direction requires a numerical algorithm
    # mask0 selects only input values where xy-MAG is non-zero
    B0xy = np.sqrt(MAG[:,0]**2+MAG[:,1]**2)
    mask0 = (B0xy > 0.) # finite xy-magnetization mask    
    N0 = np.sum(mask0)  #number of masked values

    if N0 >= 1:
        
        tetta = np.arctan2(MAG[mask0,1],MAG[mask0,0])
        gamma = np.arctan2(Y[mask0],X[mask0])
        phi = gamma-tetta

        phi0s = 2*np.pi/niter  # discretization

        # prepare masked arrays for use in algorithm

        RR_m0 = RR[mask0]
        R_m0 = R[mask0]        
        rR2 = 2*R_m0*RR_m0
        r2pR2 = R_m0**2+RR_m0**2
        Z0_m0 = H[mask0]/2
        Z_m0 = Z[mask0]
        H_m0 = H[mask0]

        Sphi = np.arange(niter+1)
        Sphi[Sphi%2==0] = 2.
        Sphi[Sphi%2==1] = 4.
        Sphi[0] = 1.
        Sphi[-1] = 1.

        SphiE = np.outer(Sphi,np.ones(N0))

        I1xE = np.ones([niter+1,N0])
        phi0E = np.outer(np.arange(niter+1),np.ones(N0))*phi0s

        Z_m0E =  np.outer(np.ones(niter+1),Z_m0)
        Z0_m0E = np.outer(np.ones(niter+1),Z0_m0)
        phiE =   np.outer(np.ones(niter+1),phi)
        rR2E =   np.outer(np.ones(niter+1),rR2)
        r2pR2E = np.outer(np.ones(niter+1),r2pR2)

        # parts for multiple use
        np.cosPhi = np.cos(phiE-phi0E)
        
        # calc R-PHI components
        ma = (r2pR2E-rR2E*np.cosPhi == 0)
        I1xE[ma] = - (1/2)/(Z_m0E[ma]+Z0_m0E[ma])**2 + (1/2)/(Z_m0E[ma]-Z0_m0E[ma])**2

        nMa = np.logical_not(ma)
        rrc = r2pR2E[nMa]-rR2E[nMa]*np.cosPhi[nMa]
        Gm = 1/np.sqrt(rrc+(Z_m0E[nMa]+Z0_m0E[nMa])**2)
        Gp = 1/np.sqrt(rrc+(Z_m0E[nMa]-Z0_m0E[nMa])**2)
        I1xE[nMa] = ((Z_m0E+Z0_m0E)[nMa]*Gm-(Z_m0E-Z0_m0E)[nMa]*Gp)/rrc

        Summand = SphiE/3.*np.cos(phi0E)*I1xE

        Br_XY_m0   = B0xy[mask0]*R_m0/2/niter*np.sum(Summand*(RR_m0-R_m0*np.cosPhi),axis=0)
        Bphi_XY_m0 = B0xy[mask0]*R_m0**2/2/niter*np.sum(Summand*np.sin(phiE-phi0E),axis=0)

        # calc Z component
        Gzm = 1./np.sqrt(r2pR2-rR2*np.cosPhi+(Z_m0E+H_m0/2)**2)
        Gzp = 1./np.sqrt(r2pR2-rR2*np.cosPhi+(Z_m0E-H_m0/2)**2)
        SummandZ = SphiE/3.*np.cos(phi0E)*(Gzp-Gzm)
        Bz_XY_m0 = B0xy[mask0]*R_m0/2/niter*np.sum(SummandZ,axis=0)

        # translate r,phi to x,y coordinates
        Bx_XY_m0 = Br_XY_m0*np.cos(gamma)-Bphi_XY_m0*np.sin(gamma)
        By_XY_m0 = Br_XY_m0*np.sin(gamma)+Bphi_XY_m0*np.cos(gamma)

        BfieldTrans = np.array([Bx_XY_m0, By_XY_m0, Bz_XY_m0]).T
        
        # add field from transversal mag to field from axial mag
        Bfield[mask0] += BfieldTrans

        # add M if inside the cylinder to make B out of H
        mask0Inside = mask0 * (RR<R) * (abs(Z)<H/2)
        Bfield[mask0Inside,:2] += MAG[mask0Inside,:2]
    
    ### END TRANS MAG CONTRIBUTION ###########################

    return(Bfield)


def field_H_cylinder(mag, dim, pos_obs, niter):  # returns arr3
    """[summary]

    Args:
        mag ([type]): [description]
        dim ([type]): [description]
        pos_obs ([type]): [description]
        niter ([type]): [description]

    Returns:
        [type]: [description]
    """
    B = field_B_cylinder(mag, pos_obs, dim, niter)

    pa = np.abs(pos_obs)
    c1 = pa[:,0]**2+pa[:,1]**2 < (dim[:,0]/2)**2
    c2 = pa[:,2]<dim[:,1]/2
    mask = c1*c2
    B[mask] -= mag[mask]

    mu0 = 4*np.pi*1e-7
    H = B/mu0/1000/1000 # to T, B to H, to kA/m
    
    return H
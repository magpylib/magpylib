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

import numpy as np

def QmultV(Q, P):
    """
    Implementation of the quaternion multiplication
    """
    Sig = np.array([[1,-1,-1,-1],[1,1,1,-1],[1,-1,1,1],[1,1,-1,1]])
    M = Q*np.array([P,np.roll(P[:,::-1],2,axis=1),np.roll(P,2,axis=1),P[:,::-1]])
    M = np.swapaxes(M,0,1)
    return np.sum(M*Sig,axis=2)


def QconjV(Q):
    """
    Implementation of the conjugation of a quaternion
    """
    Sig = np.array([1,-1,-1,-1])
    return Q*Sig


def getRotQuatV(ANGLE, AXIS):
    """
    ANGLE in [deg], AXIS dimensionless
    vectorized version of getRotQuat, returns the rotation quaternion which 
    describes the rotation given by angle and axis (see paper)
    NOTE: axis cannot be [0,0,0] !!! this would not describe a rotation. however
        sinPhi = 0 returns a 0-axis (but just in the Q this must still be 
        interpreted correctly as an axis)
    """
    Lax = np.linalg.norm(AXIS,axis=1)
    Uax = AXIS/Lax[:,None]   # normalize

    Phi = ANGLE/180*np.pi/2
    cosPhi = np.cos(Phi)
    sinPhi = np.sin(Phi)
    
    Q = np.array([cosPhi] + [Uax[:,i]*sinPhi for i in range(3)])

    return np.swapaxes(Q,0,1)

def QrotationV(Q,v):
    """
    replaces angle axis rotation by direct Q-rotation to skip this step speed
    when multiple subsequent rotations are given
    """
    Qv = np.pad(v,((0,0),(1,0)), mode='constant') 
    Qv_new = QmultV(Q, QmultV(Qv, QconjV(Q)))
    return Qv_new[:,1:]


def getAngAxV(Q):
    # UNUSED - KEEP FOR UNDERSTANDING AND TESTING
    # returns angle and axis for a quaternion orientation input
    angle = np.arccos(Q[:,0])*180/np.pi*2
    axis = Q[:,1:]
    
    # a quaternion with a 0-axis describes a unit rotation (0-angle).
    # there should still be a proper axis output but it is eliminated
    # by the term [Uax[:,i]*sinPhi for i in range(3)]) with sinPhi=0.
    # since for 0-angle the axis doesnt matter we can set it to [0,0,1] 
    # which is our defined initial orientation
    
    Lax = np.linalg.norm(axis,axis=1)
    mask = Lax!=0
    Uax = np.array([[0,0,1.]]*len(axis))     # set all to [0,0,1]
    Uax[mask] = axis[mask]/Lax[mask,None]   # use mask to normalize non-zeros
    return angle,Uax


def angleAxisRotationV_priv(ANGLE, AXIS, V):
    # vectorized version of angleAxisRotation_priv
    P = getRotQuatV(ANGLE, AXIS)
    Qv = np.pad(V,((0,0),(1,0)), mode='constant') 
    Qv_new = QmultV(P, QmultV(Qv, QconjV(P)))
    return Qv_new[:,1:]


def randomAxisV(N):
    """
    This is the vectorized (loop-free) version of randomAxis(). It generates an 
    N-sized vector of random `axes` (3-vector of length 1) from equal 
    angular distributions using a MonteCarlo scheme.

    Parameters
    -------
    N : int
        Size of random axis vector.

    Returns
    -------
    axes : Nx3 arr
        A  vector of random axes from an equal angular distribution of length 1.
    """
    
    # R = np.random.rand(N,3)*2-1
    
    # while True:
    #     lenR = np.linalg.norm(R,axis=1)
    #     mask = lenR > 1  #bad = True
    #     Nbad = np.sum(mask)
    #     if Nbad==0:
    #         return R/lenR
    #     else:
    #         R[mask] = np.random.rand(Nbad,3)*2-1

    R = np.random.rand(N,3)*2-1
        
    while True:
        lenR = np.linalg.norm(R,axis=1)
        mask = lenR > 1  #bad = True
        Nbad = np.sum(mask)
        if Nbad==0:
            return R/lenR[:,np.newaxis]
        else:
            R[mask] = np.random.rand(Nbad,3)*2-1



def axisFromAnglesV():
    """
    WIP
    """
    print('WIP')
    return 0

def anglesFromAxisV():
    """
    WIP
    """
    print('WIP')
    return 0

def angleAxisRotationV():
    """
    WIP
    """
    print('WIP')
    return 0
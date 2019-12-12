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
    This is the vectorized version of randomAxis(). It generates an 
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

    Example
    -------
    >>> import magpylib as magpy
    >>> import magpylib as magpy
    >>> AXS = magpy.math.randomAxisV(3)
    >>> print(AXS)
    >>> # Output: [[ 0.39480364 -0.53600779 -0.74620757]
    ... [ 0.02974442  0.10916333  0.9935787 ]
    ... [-0.54639126  0.76659756 -0.33731997]]
    """

    R = np.random.rand(N,3)*2-1
        
    while True:
        lenR = np.linalg.norm(R,axis=1)
        mask = lenR > 1  #bad = True
        Nbad = np.sum(mask)
        if Nbad==0:
            return R/lenR[:,np.newaxis]
        else:
            R[mask] = np.random.rand(Nbad,3)*2-1



def axisFromAnglesV(ANG):
    """
    This is the vectorized version of axisFromAngles(). It generates an Nx3 
    array of axis vectors from the Nx2 array of input angle pairs angles. 
    Each angle pair is (phi,theta) which are azimuth and polar angle of a 
    spherical coordinate system respectively.

    Parameters
    ----------
    ANG : arr Nx2 [deg]
        An N-sized array of angle pairs [phi th], azimuth and polar, in 
        units of deg.

    Returns    
    -------
    AXIS : arr Nx3
        An N-sized array of unit axis vectors oriented as given by the input ANG.
    
    Example
    -------
    >>> import magpylib as magpy
    >>> import numpy as np
    >>> ANGS = np.array([[0,90],[90,180],[90,0]])
    >>> AX = magpy.math.axisFromAnglesV(ANGS)
    >>> print(np.around(AX,4))
    >>> # Output: [[1.  0. 0.]  [0. 0. -1.]  [0. 0. 1.]]
    """
    PHI = ANG[:,0]/180*np.pi
    TH = ANG[:,1]/180*np.pi

    return np.array([np.cos(PHI)*np.sin(TH), np.sin(PHI)*np.sin(TH), np.cos(TH)]).transpose()



def anglesFromAxisV(AXIS):
    """
    This is the vectorized version of anglesFromAxis(). It takes an Nx3 array 
    of axis-vectors and returns an Nx2 array of angle pairs. Each angle pair is 
    (phi,theta) which are azimuth and polar angle in a spherical coordinate 
    system respectively.

    Parameters
    ----------
    AXIS : arr Nx3
        N-sized array of axis-vectors (do not have to be not be normalized).

    Returns
    -------
    ANGLES : arr Nx2 [deg]
        N-sized array of angle pairs [phi,th], azimuth and polar, that 
        chorrespond to the orientations given by the input axis vectors 
        in a spherical coordinate system.
     
    Example
    -------
    >>> import numpy as np
    >>> import magpylib as magpy
    >>> AX = np.array([[0,0,1],[0,0,1],[1,0,0]])
    >>> ANGS = magpy.math.anglesFromAxisV(AX)
    >>> print(ANGS)
    >>> # Output: [[0. 0.]  [90. 90.]  [0. 90.]])
    """

    Lax = np.linalg.norm(AXIS,axis=1)
    Uax = AXIS/Lax[:,np.newaxis]

    TH = np.arccos(Uax[:,2])/np.pi*180
    PHI = np.arctan2(Uax[:,1], Uax[:,0])/np.pi*180
    return np.array([PHI, TH]).transpose()


def angleAxisRotationV(POS,ANG,AXIS,ANCHOR):
    """
    This is the vectorized version of angleAxisRotation(). Each entry 
    of POS (arrNx3) is rotated according to the angles ANG (arrN), 
    about the axis vectors AXS (arrNx3) which pass throught the anchors 
    ANCH (arrNx3) where N refers to the length of the input vectors.

    Parameters
    ----------
    POS : arrNx3
        The input vectors to be rotated.

    ANG : arrN [deg]
        Rotation angles in units of [deg].

    AXIS : arrNx3
        Vector of rotation axes.

    anchor : arrNx3
        Vector of rotation anchors.

    Returns    
    -------
    newPOS : arrNx3
        Vector of rotated positions.

    >>> import magpylib as magpy
    >>> import numpy as np
    >>> POS = np.array([[1,0,0]]*5) # avoid this slow Python loop
    >>> ANG = np.linspace(0,180,5)
    >>> AXS = np.array([[0,0,1]]*5) # avoid this slow Python loop
    >>> ANCH = np.zeros((5,3))
    >>> POSnew = magpy.math.angleAxisRotationV(POS,ANG,AXS,ANCH)
    >>> print(np.around(POSnew,4))
    >>> # Output: [[ 1.      0.      0.    ]
    ...            [ 0.7071  0.7071  0.    ]
    ...            [ 0.      1.      0.    ]
    ...            [-0.7071  0.7071  0.    ]
    ...            [-1.      0.      0.    ]]
    """

    POS12 = POS-ANCHOR
    POS12rot = angleAxisRotationV_priv(ANG,AXIS,POS12)
    POSnew = POS12rot+ANCHOR

    return POSnew


# vectorized version of elliptic integral
def ellipticV(INPUT):

    kc = INPUT[:,0]
    p = INPUT[:,1]
    c = INPUT[:,2]
    s = INPUT[:,3]

    #if kc == 0:
    #    return NaN
    errtol = .000001
    N = len(kc)
    
    k = np.abs(kc)
    em = np.ones(N,dtype=float)

    cc = c.copy()
    pp = p.copy()
    ss = s.copy()
    
    # apply a mask for evaluation of respective cases
    mask = p>0
    maskInv = np.invert(mask)

    #if p>0:
    pp[mask] = np.sqrt(p[mask])
    ss[mask] = s[mask]/pp[mask]

    #else:
    f = kc[maskInv]*kc[maskInv]
    q = 1.-f
    g = 1. - pp[maskInv]
    f = f - pp[maskInv]
    q = q*(ss[maskInv] - c[maskInv]*pp[maskInv])
    pp[maskInv] = np.sqrt(f/g)
    cc[maskInv] = (c[maskInv]-ss[maskInv])/g
    ss[maskInv] = -q/(g*g*pp[maskInv]) + cc[maskInv]*pp[maskInv]

    f = cc.copy()
    cc = cc + ss/pp
    g = k/pp
    ss = 2*(ss + f*g)
    pp = g + pp
    g = em.copy()
    em = k + em
    kk = k.copy()

    #define a mask that adjusts with every evauation
    #   step so that only non-converged entries are
    #   further iterated.   
    mask = np.ones(N,dtype=bool)
    while np.any(mask):
        k[mask] = 2*np.sqrt(kk[mask])
        kk[mask] = np.copy(k[mask]*em[mask])
        f[mask] = cc[mask]
        cc[mask] = cc[mask] + ss[mask]/pp[mask]
        g[mask] = kk[mask]/pp[mask]
        ss[mask] = 2*(ss[mask] + f[mask]*g[mask])
        pp[mask] = g[mask] + pp[mask]
        g[mask] = em[mask]
        em[mask] = k[mask]+em[mask]

        #redefine mask so only non-convergent 
        #   entries are reiterated
        mask = (np.abs(g-k) > g*errtol)

    return(np.pi/2)*(ss+cc*em)/(em*(em+pp))
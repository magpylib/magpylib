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
import numpy
from numpy import arctan, pi, array, sqrt, NaN, cos, sin, arccos, float64, sqrt, arctan2


# IMPROVED ANGLE FUNCTIONS ################################################################

# avoid numerical problem to evaluate at 1.000000000001
def arccosSTABLE(x):
    '''
    arccos with improved numerical stability
    '''
    if 1 > x > -1:
        return arccos(x)
    elif x >= 1:
        return 0
    elif x <= -1:
        return pi


# FAST VERSIONS OF 3D VECTOR ALGEBRA ###########################################################
# fast versions for 3-vector computation only

def fastCross3D(u, v):
    '''
    outer product of 2 3-vectors
    '''
    return array([u[1]*v[2]-u[2]*v[1], -u[0]*v[2]+u[2]*v[0], u[0]*v[1]-u[1]*v[0]])


def fastSum3D(u):
    '''
    sum of 3-vector
    '''
    return u[0]+u[1]+u[2]


def fastNorm3D(u):
    '''
    norm of a 3-vector
    '''
    return sqrt(u[0]**2+u[1]**2+u[2]**2)


# QUATERNIONS for ANGLE-AXIS ROTATION ####################################################
# Quaterntions are defined as 4D Lists

def Qmult(Q, P):
    '''
    Quaternion multiplication
    '''
    r0 = Q[0]*P[0] - Q[1]*P[1] - Q[2]*P[2] - Q[3]*P[3]
    r1 = Q[0]*P[1] + Q[1]*P[0] + Q[2]*P[3] - Q[3]*P[2]
    r2 = Q[0]*P[2] - Q[1]*P[3] + Q[2]*P[0] + Q[3]*P[1]
    r3 = Q[0]*P[3] + Q[1]*P[2] - Q[2]*P[1] + Q[3]*P[0]
    return [r0, r1, r2, r3]


def Qnorm2(Q):
    '''
    Quaternion Norm**2
    '''
    return Q[0]**2 + Q[1]**2 + Q[2]**2 + Q[3]**2


def Qunit(Q):
    '''
    unit quaternion
    '''
    qnorm = sqrt(Qnorm2(Q))
    return [q/qnorm for q in Q]


def Qconj(Q):
    '''
    Conjugation of Quaternion
    '''
    return array([Q[0], -Q[1], -Q[2], -Q[3]])


def getRotQuat(angle, axis):
    '''
    getRotationQuaternion from axis angle (see Kuipers p.131)
    '''
    Lax = fastNorm3D(axis)
    Uax = axis/Lax

    Phi = angle/180*pi/2
    cosPhi = cos(Phi)
    sinPhi = sin(Phi)

    Q = [cosPhi] + [a*sinPhi for a in Uax]

    return Q


def angleAxisRotation_priv(angle, axis, v):
    '''
    Angle-Axis Rotation of Vector
    '''
    P = getRotQuat(angle, axis)

    Qv = [0, v[0], v[1], v[2]]
    Qv_new = Qmult(P, Qmult(Qv, Qconj(P)))

    return array(Qv_new[1:])


# SPECIAL FUNCTIONS ################################################################
# See https://dlmf.nist.gov/19.2


def cel(kc, p, c, s):
    '''
    Numerical scheme to evaluate the cel Bulirsch integral.
    Algorithm proposed in Derby et al., arXiev:00909.3880v1
    '''
    if kc == 0:
        return NaN
    errtol = .000001
    k = abs(kc)
    pp = p
    cc = c
    ss = s
    em = 1.
    if p > 0:
        pp = sqrt(p)
        ss = s/pp
    else:
        f = kc*kc
        q = 1.-f
        g = 1. - pp
        f = f - pp
        q = q*(ss - c*pp)
        pp = sqrt(f/g)
        cc = (c-ss)/g
        ss = -q/(g*g*pp) + cc*pp
    f = cc
    cc = cc + ss/pp
    g = k/pp
    ss = 2*(ss + f*g)
    pp = g + pp
    g = em
    em = k + em
    kk = k
    while abs(g-k) > g*errtol:
        k = 2*sqrt(kk)
        kk = k*em
        f = cc
        cc = cc + ss/pp
        g = kk/pp
        ss = 2*(ss + f*g)
        pp = g + pp
        g = em
        em = k+em
    return(pi/2)*(ss+cc*em)/(em*(em+pp))


def ellipticK(x):
    '''
    Legendres complete elliptic integral of the first kind
    '''
    return cel((1-x)**(1/2.), 1, 1, 1)


def ellipticE(x):
    '''
    Legendres complete elliptic integral of the second kind
    '''
    return cel((1-x)**(1/2.), 1, 1, 1-x)


def ellipticPi(x, y):
    '''
    Legendres complete elliptic integral of the third kind
    '''
    return cel((1-y)**(1/2.), 1-x, 1, 1)


# AXES AND ROTATIONS #############################################################

def randomAxis():
    """
    This function generates a random `axis` (3-vector of length 1) from an equal
    angular distribution using a MonteCarlo scheme.

    Returns
    -------
    axis : arr3
        A random axis from an equal angular distribution of length 1

    Example
    -------
    >>> magpylib as magPy
    >>> ax = magPy.math.randomAxis()
    >>> print(ax)
      [-0.24834468  0.96858637  0.01285925]

    """
    while True:
        r = numpy.random.rand(3)*2-1  # create random axis
        Lr2 = sum(r**2)  # get length
        if Lr2 <= 1:  # is axis within sphere?
            Lr = sqrt(Lr2)  # normalize
            return r/Lr


def axisFromAngles(angles):
    """
    This function generates an `axis` (3-vector of length 1) from two `angles` = [phi,th]
    that are defined as in spherical coordinates. phi = azimuth angle, th = polar angle.
    Vector input format can be either list, tuple or array of any data type (float, int).

    Parameters
    ----------
    angles : vec2 [deg]
        The two angels [phi,th], azimuth and polar, in units of deg.

    Returns    
    -------
    axis : arr3
        An axis of length that is oriented as given by the input angles.

    Example
    -------
    >>> magpylib as magPy
    >>> angles = [90,90]
    >>> ax = magPy.math.axisFromAngles(angles)
    >>> print(ax)
      [0.0  1.0  0.0]
    """
    phi, th = angles  # phi in [0,2pi], th in [0,pi]
    phi = phi/180*pi
    th = th/180*pi
    return array([cos(phi)*sin(th), sin(phi)*sin(th), cos(th)])


def anglesFromAxis(axis):
    """
    This function takes an arbitrary `axis` (3-vector) and returns the orientation
    given by the `angles` = [phi,th] that are defined as in spherical coordinates. 
    phi = azimuth angle, th = polar angle. Vector input format can be either 
    list, tuple or array of any data type (float, int).

    Parameters
    ----------
    axis : vec3
        Arbitrary input axis that defines an orientation.

    Returns
    -------
    angles : arr2 [deg]
        The angles [phi,th], azimuth and polar, that anchorrespond to the orientation 
        given by the input axis.

    Example
    -------
    >>> magpylib as magPy
    >>> axis = [1,1,0]
    >>> angles = magPy.math.anglesFromAxis(axis)
    >>> print(angles)
      [45. 90.]
    """
    ax = array(axis, dtype=float64, copy=False)

    Lax = fastNorm3D(ax)
    Uax = ax/Lax

    TH = arccos(Uax[2])/pi*180
    PHI = arctan2(Uax[1], Uax[0])/pi*180
    return array([PHI, TH])


def angleAxisRotation(position, angle, axis, anchor=[0, 0, 0]):
    """
    This function uses angle-axis rotation to rotate the `position` vector by
    the `angle` argument about an axis defined by the `axis` vector which passes
    through the center of rotation `anchor` vector. Scalar input is either integer
    or float.Vector input format can be either list, tuple or array of any data
    type (float, int).

    Parameters
    ----------
    position : vec3
        Input position to be rotated.

    angle : scalar [deg]
        Angle of rotation in untis of [deg]

    axis : vec3
        Axis of rotation

    anchor : vec3
        The Center of rotation which defines the position of the axis of rotation

    Returns    
    -------
    newPosition : arr3
        Rotated position

    Example
    -------
    >>> magpylib as magPy
    >>> from numpy import pi
    >>> position0 = [1,1,0]
    >>> angle = -90
    >>> axis = [0,0,1]
    >>> centerOfRotation = [1,0,0]
    >>> positionNew = magPy.math.angleAxisRotation(position0,angle,axis,anchor=centerOfRotation)
    >>> print(positionNew)
      [2. 0. 0.]
    """

    pos = array(position, dtype=float64, copy=False)
    ang = float(angle)
    ax = array(axis, dtype=float64, copy=False)
    anchor = array(anchor, dtype=float64, copy=False)

    pos12 = pos-anchor
    pos12Rot = angleAxisRotation_priv(ang, ax, pos12)
    posRot = pos12Rot+anchor

    return posRot

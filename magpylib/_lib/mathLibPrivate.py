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
from numpy import arctan, pi, array, sqrt, NaN, cos, sin, arccos

# %% total rotation matix for rotation aboutthree euler angles - first X, then Y, then Z
'''
def Mrot(Phis):
    Mx = array([[1,0,0], [0, cos(Phis[0]), -sin(Phis[0])], [0, sin(Phis[0]), cos(Phis[0])]])
    My = array([[cos(Phis[1]), 0, sin(Phis[1])], [0, 1, 0], [-sin(Phis[1]), 0, cos(Phis[1])]])
    Mz = array([[cos(Phis[2]), -sin(Phis[2]), 0], [sin(Phis[2]), cos(Phis[2]), 0], [0, 0, 1]])
    return dot(Mz,dot(My,Mx))

# inverse der totalen rotationsmatrix
def MrotInv(Phis):
    Mx = array([[1,0,0], [0, cos(Phis[0]), sin(Phis[0])], [0, -sin(Phis[0]), cos(Phis[0])]])
    My = array([[cos(Phis[1]), 0, -sin(Phis[1])], [0, 1, 0], [sin(Phis[1]), 0, cos(Phis[1])]])
    Mz = array([[cos(Phis[2]), sin(Phis[2]), 0], [-sin(Phis[2]), cos(Phis[2]), 0], [0, 0, 1]])
    return dot(Mx,dot(My,Mz))
'''

# %%SMOOTH VERSION OF ARCTAN
# get a smooth version of the cylindrical coordinates


def getPhi(x, y):
    if x > 0:
        return arctan(y/x)
    elif x < 0:
        if y >= 0:
            return arctan(y/x)+pi
        else:
            return arctan(y/x)-pi
    else:
        if y > 0:
            return pi/2
        else:
            return -pi/2


# %% NUMERICALY STABLE VERSION OF ARCCOS

# avoid numerical problem to evaluate at 1.000000000001
def arccosSTABLE(x):
    if 1 > x > -1:
        return arccos(x)
    elif x >= 1:
        return 0
    elif x <= -1:
        return pi


# %% FAST VERSIONS OF 3D VECTOR ALGEBRA

# more than 10-times faster than native np.cross (which is for arbitrary dimensions)
def fastCross3D(u, v):
    return array([u[1]*v[2]-u[2]*v[1], -u[0]*v[2]+u[2]*v[0], u[0]*v[1]-u[1]*v[0]])

# much faster than sum()


def fastSum3D(u):
    return u[0]+u[1]+u[2]

# much faster than np.la.norm (which is for arbitrary dimensions)


def fastNorm3D(u):
    return sqrt(u[0]**2+u[1]**2+u[2]**2)


# %% QUATERNIONS for ANGLE-AXIS ROTATION

# Quaterntions are defined as 4D Lists

# Quaternion multiplication
def Qmult(Q, P):
    r0 = Q[0]*P[0] - Q[1]*P[1] - Q[2]*P[2] - Q[3]*P[3]
    r1 = Q[0]*P[1] + Q[1]*P[0] + Q[2]*P[3] - Q[3]*P[2]
    r2 = Q[0]*P[2] - Q[1]*P[3] + Q[2]*P[0] + Q[3]*P[1]
    r3 = Q[0]*P[3] + Q[1]*P[2] - Q[2]*P[1] + Q[3]*P[0]
    return [r0, r1, r2, r3]

# Quaternion Norm**2


def Qnorm2(Q):
    return Q[0]**2 + Q[1]**2 + Q[2]**2 + Q[3]**2

# Unit Quaternion


def Qunit(Q):
    qnorm = sqrt(Qnorm2(Q))
    return [q/qnorm for q in Q]

# Conjugate Quaternion


def Qconj(Q):
    return array([Q[0], -Q[1], -Q[2], -Q[3]])

# getRotationQuaternion from axis angle (see Kuipers p.131)


def getRotQuat(angle, axis):
    Lax = fastNorm3D(axis)
    Uax = axis/Lax

    Phi = angle/180*pi/2
    cosPhi = cos(Phi)
    sinPhi = sin(Phi)

    Q = [cosPhi] + [a*sinPhi for a in Uax]

    return Q

# Angle-Axis Rotation of Vector


def angleAxisRotation(angle, axis, v):
    P = getRotQuat(angle, axis)

    Qv = [0, v[0], v[1], v[2]]
    Qv_new = Qmult(P, Qmult(Qv, Qconj(P)))

    return array(Qv_new[1:])


# %% ELLIPTICAL INTEGRALS

#from scipy.integrate import quad
# Algorithm to determine a special elliptic integral
# Algorithm proposed in Derby, Olbert 'Cylindrical Magnets and Ideal Solenoids'
# arXiev:00909.3880v1

def elliptic(kc, p, c, s):
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

# complete elliptic integral of the first kind: ellipticK
# E(x) = \int_0^pi/2 (1-x sin(phi)^2)^(-1/2) dphi
# Achtung: fur x>1 wird der output imaginaer und die derzeitigen algorithmen brechen zusammen


def ellipticK(x):
    return elliptic((1-x)**(1/2.), 1, 1, 1)

# complete elliptic integral of the second kind: ellipticE
# E(x) = \int_0^pi/2 (1-x sin(phi)^2)^(1/2) dphi
# Achtung: fur x>1 wird der output imaginaer und die derzeitigen algorithmen brechen zusammen


def ellipticE(x):
    return elliptic((1-x)**(1/2.), 1, 1, 1-x)

# complete elliptic integral of the third kind: ellipticPi


def ellipticPi(x, y):
    return elliptic((1-y)**(1/2.), 1-x, 1, 1)

# TESTING ALGORITHM ---------------------------------------------------------
# def integrand(phi,kc,p,c,s):
#    return (c*cos(phi)**2+s*sin(phi)**2)/(cos(phi)**2+p*sin(phi)**2)/sqrt(cos(phi)**2+kc**2*sin(phi)**2)

# def nelliptic(kc,p,c,s):
#    I = quad(integrand,0,pi/2,args=(kc,p,c,s))
#    return I

#from scipy.integrate import quad
# def integrand(phi,x):
#    return (1-x*sin(phi)**2)**(-1/2.)
# def nelliptic(x):
#    I = quad(integrand,0,pi/2,args=x)
#    return I
# print(nelliptic(-.51))

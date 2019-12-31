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

from numpy import array, NaN
from magpylib._lib.mathLib import fastSum3D, fastNorm3D, fastCross3D
from warnings import warn
import numpy as np
from numpy.linalg import norm

# %% CURRENT LINE
# Describes the magnetic field of a line current. The line is given by a set of
#   data points and the field is the superopsition of the fields of all linear
#   segments.

# i0   : float       [A]       current lowing from the first point to the last point
# pos  : arr3 float  [mm]      Position of observer
# possis: list or arr of arr3floats [mm]    Positions that define the line

# source: http://www.phys.uri.edu/gerhard/PHY204/tsl216.pdf
# FieldOfLineCurrent

# VECTORIZED VERSION
# here we only use vectorized code as this function will primarily be
#   used to call on multiple line segments. The vectorized code was
#   developed based on the depreciated version below.

def Bfield_LineSegmentV(p0, p1, p2, I0):
    ''' private
    base function determines the fields of given line segments
    p0 = observer position
    p1->p2 = current flows from vertex p1 to vertex p2
    I0 = current in [A]
    '''

    N = len(p0)
    fields = np.zeros((N,3)) # default values for mask0 and mask1

    # Check for zero-length segment
    mask0 = np.all(p1==p2,axis=1)

    # projection of p0 onto line p1-p2
    nm0 = np.invert(mask0)
    p1p2 = (p1[nm0]-p2[nm0])
    p4 = p1[nm0]+(p1p2.T*np.sum((p0[nm0]-p1[nm0])*p1p2,axis=1)/np.sum(p1p2**2,axis=1)).T

    # determine anchorrect normal vector to surface spanned by triangle
    cross0 = np.cross(-p1p2, p0[nm0]-p4)
    norm_cross0 = norm(cross0,axis=1)

    # on-line cases (include when position is on current path)
    mask1 = (norm_cross0 == 0)

    # normal cases
    nm1 = np.invert(mask1)
    eB = (cross0[nm1].T/norm_cross0[nm1]) #field direction

    # not mask0 and not mask1
    NM = np.copy(nm0)
    NM[NM==True] = nm1

    norm_04 = norm(p0[NM] -p4[nm1],axis=1)
    norm_01 = norm(p0[NM] -p1[NM],axis=1)
    norm_02 = norm(p0[NM] -p2[NM],axis=1)
    norm_12 = norm(p1[NM] -p2[NM],axis=1)
    norm_41 = norm(p4[nm1]-p1[NM],axis=1)
    norm_42 = norm(p4[nm1]-p2[NM],axis=1)

    sinTh1 = norm_41/norm_01
    sinTh2 = norm_42/norm_02

    deltaSin = np.empty((N))[NM]

    # determine how p1,p2,p4 are sorted on the line (to get sinTH signs)
    # both points below
    mask2 = ((norm_41>norm_12) * (norm_41>norm_42))
    deltaSin[mask2] = abs(sinTh1[mask2]-sinTh2[mask2])
    # both points above
    mask3 = ((norm_42>norm_12) * (norm_42>norm_41))
    deltaSin[mask3] = abs(sinTh2[mask3]-sinTh1[mask3])
    # one above one below or one equals p4
    mask4 = np.invert(mask2)*np.invert(mask3)
    deltaSin[mask4] = abs(sinTh1[mask4]+sinTh2[mask4])

    # missing 10**-6 from m->mm conversion #T->mT conversion
    fields[NM] = (I0[NM]*deltaSin/norm_04*eB).T/10

    return fields



def Bfield_CurrentLineV(VERT,i0,poso):
    ''' private
    determine total field from a multi-segment line current
    '''

    N = len(VERT)-1
    P0 = np.outer(np.ones((N)),poso)
    P1 = VERT[:-1]
    P2 = VERT[1:]
    I0 = np.ones((N))*i0

    Bv = Bfield_LineSegmentV(P0,P1,P2,I0)

    return np.sum(Bv,axis=0)





''' DEPRECHIATED VERSION (non-vectorized)

# observer at p0
# current I0 flows in straight line from p1 to p2
def Bfield_LineSegment(p0, p1, p2, I0):
    # must receive FLOAT only !!!!
    # Check for zero-length segment
    if all(p1==p2):
        warn("Zero-length segment line detected in vertices list,"
             "returning [0,0,0]", RuntimeWarning)
        return array([0, 0, 0])

    # projection of p0 onto line p1-p2
    p4 = p1+(p1-p2)*fastSum3D((p0-p1)*(p1-p2))/fastSum3D((p1-p2)*(p1-p2))

    # determine anchorrect normal vector to surface spanned by triangle
    cross0 = fastCross3D(p2-p1, p0-p4)
    norm_cross0 = fastNorm3D(cross0)
    if norm_cross0 != 0.:
        eB = cross0/norm_cross0
    else:  # on line case (p0,p1,p2) do not span a triangle
        norm_12 = fastNorm3D(p1-p2)
        norm_42 = fastNorm3D(p4-p2)
        norm_41 = fastNorm3D(p4-p1)

        if (norm_41 <= norm_12 and norm_42 <= norm_12):  # in-between the two points
            warn('Warning: getB Position directly on current line', RuntimeWarning)
            return array([NaN, NaN, NaN])
        else:
            return array([0, 0, 0])

    # determine sinTHs and R
    norm_04 = fastNorm3D(p0-p4)  # =R
    norm_01 = fastNorm3D(p0-p1)
    norm_02 = fastNorm3D(p0-p2)
    norm_12 = fastNorm3D(p1-p2)
    norm_41 = fastNorm3D(p4-p1)
    norm_42 = fastNorm3D(p4-p2)

    sinTh1 = norm_41/norm_01
    sinTh2 = norm_42/norm_02

    # determine how p1,p2,p4 are sorted on the line (to get sinTH signs)
    if norm_41 > norm_12 and norm_41 > norm_42:  # both points below
        deltaSin = abs(sinTh1-sinTh2)
    elif norm_42 > norm_12 and norm_42 > norm_41:  # both points above
        deltaSin = abs(sinTh2-sinTh1)
    else:  # one above one below or one equals p4
        deltaSin = abs(sinTh1+sinTh2)

    # missing 10**-6 from m->mm conversion #T->mT conversion
    B = I0*deltaSin/norm_04*eB/10

    return B
'''
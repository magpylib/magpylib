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
from magpylib._lib.mathLibPrivate import fastSum3D, fastNorm3D, fastCross3D
from warnings import warn

# %% CURRENT LINE
# Describes the magnetic field of a line current. The line is given by a set of
#   data points and the field is the superopsition of the fields of all linear
#   segments.

# i0   : float       [A]       current lowing from the first point to the last point
# pos  : arr3 float  [mm]      Position of observer
# possis: list or arr of arr3floats [mm]    Positions that define the line

# source: http://www.phys.uri.edu/gerhard/PHY204/tsl216.pdf
# FieldOfLineCurrent

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

# determine total field from multiple segments


def Bfield_CurrentLine(p0, possis, I0):

    B = array([0., 0., 0.])
    for i in range(len(possis)-1):
        p1 = possis[i]
        p2 = possis[i+1]
        B += Bfield_LineSegment(p0, p1, p2, I0)
    return B

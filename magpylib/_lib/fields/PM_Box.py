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
# -*- coding: utf-8 -*-

# MAGNETIC FIELD CALCULATION OF CUBE IN CANONICAL BASIS


# %% IMPORTS
from numpy import pi, sign, sqrt, log, array, arctan, NaN
from warnings import warn

# %% CALCULATIONS

# Describes the magnetic field of a cuboid magnet with sides parallel to its native
#   cartesian coordinates. The dimension is 2a x 2b x 2c and the magnetization
#   is given by MAG. The center of the box is positioned at posM.

# MAG : arr3   [mT/mm³]     Magnetization per unit volume, MAG = mu0*mag = remanence field
# pos  : arr3  [mm]        Position of observer
# dim  : arr3  [mm]        dim = [a,b,c], Magnet dimension = A x B x C

# basic functions required to calculate the cuboid's fields

# calculating the field
# returns arr3 of field vector in [mT], input in [mT] and [mm]
def Bfield_Box(MAG, pos, dim):

    MAGx, MAGy, MAGz = MAG/4/pi
    x, y, z = pos
    a, b, c = dim/2

    xma, xpa = x-a, x+a
    ymb, ypb = y-b, y+b
    zmc, zpc = z-c, z+c

    xma2, xpa2 = xma**2, xpa**2
    ymb2, ypb2 = ymb**2, ypb**2
    zmc2, zpc2 = zmc**2, zpc**2

    MMM = sqrt(xma2 + ymb2 + zmc2)
    PMP = sqrt(xpa2 + ymb2 + zpc2)
    PMM = sqrt(xpa2 + ymb2 + zmc2)
    MMP = sqrt(xma2 + ymb2 + zpc2)
    MPM = sqrt(xma2 + ypb2 + zmc2)
    PPP = sqrt(xpa2 + ypb2 + zpc2)
    PPM = sqrt(xpa2 + ypb2 + zmc2)
    MPP = sqrt(xma2 + ypb2 + zpc2)

    # special cases:
    #   0. volume cases      no quantities are zero
    #   1. on surfaces:      one quantity is zero
    #   2. on edge lines:    two quantities are zero
    #   3. on corners:       three quantities are zero
    CASE = 0
    for case in array([xma, xpa, ymb, ypb, zmc, zpc]):
        if (case < 1e-15 and -1e-15 < case):
            CASE += 1
    # rounding is required to catch numerical problem cases like .5-.55=.05000000000000001
    #   which then result in 'normal' cases but the square eliminates the small digits

    # case 1(on magnet): catch on magnet surface cases------------------------
    if CASE == 1:
        if abs(x) <= a:
            if abs(y) <= b:
                if abs(z) <= c:
                    warn('Warning: getB Position directly on magnet surface', RuntimeWarning)
                    return array([NaN, NaN, NaN])

    # cases 2 & 3 (edgelines edges and corners) ------------------------------
    if CASE > 1:

        # on corner and on edge cases - no phys solution
        # directly on magnet edge or corner - log singularity here as result of unphysical model
        if all([abs(x) <= a, abs(y) <= b, abs(z) <= c]):
            warn('Warning: getB Position directly on magnet surface', RuntimeWarning)
            return array([NaN, NaN, NaN])

        # problematic edgeline cases (here some specific LOG quantites become problematic and are obtained from a mirror symmetry)
        caseA = (xma < 0 and xpa < 0)
        caseB = (ymb > 0 and ypb > 0)
        caseC = (zmc > 0 and zpc > 0)

        if caseA:
            xma, xpa = -xma, -xpa
        elif caseB:
            ymb, ypb = -ymb, -ypb
        elif caseC:
            zmc, zpc = -zmc, -zpc

        LOGx = log(((xma+MMM)*(xpa+PPM)*(xpa+PMP)*(xma+MPP)) /
                   ((xpa+PMM)*(xma+MPM)*(xma+MMP)*(xpa+PPP)))
        LOGy = log(((-ymb+MMM)*(-ypb+PPM)*(-ymb+PMP)*(-ypb+MPP)) /
                   ((-ymb+PMM)*(-ypb+MPM)*(ymb-MMP)*(ypb-PPP)))
        LOGz = log(((-zmc+MMM)*(-zmc+PPM)*(-zpc+PMP)*(-zpc+MPP)) /
                   ((-zmc+PMM)*(zmc-MPM)*(-zpc+MMP)*(zpc-PPP)))

        if caseA:
            LOGx, xma, xpa = -LOGx, -xma, -xpa
        elif caseB:
            LOGy, ymb, ypb = -LOGy, -ymb, -ypb
        elif caseC:
            LOGz, zmc, zpc = -LOGz, -zmc, -zpc

    # case 0 and 1(off magnet): (most cases) -----------------------------------------------
    else:
        # these quantities have positive definite denominators in all cases 0 and 1
        LOGx = log(((xma+MMM)*(xpa+PPM)*(xpa+PMP)*(xma+MPP)) /
                   ((xpa+PMM)*(xma+MPM)*(xma+MMP)*(xpa+PPP)))
        LOGy = log(((-ymb+MMM)*(-ypb+PPM)*(-ymb+PMP)*(-ypb+MPP)) /
                   ((-ymb+PMM)*(-ypb+MPM)*(ymb-MMP)*(ypb-PPP)))
        LOGz = log(((-zmc+MMM)*(-zmc+PPM)*(-zpc+PMP)*(-zpc+MPP)) /
                   ((-zmc+PMM)*(zmc-MPM)*(-zpc+MMP)*(zpc-PPP)))

    # calculate unproblematic field components
    BxY = MAGy*LOGz
    BxZ = MAGz*LOGy
    ByX = MAGx*LOGz
    ByZ = -MAGz*LOGx
    BzX = MAGx*LOGy
    BzY = -MAGy*LOGx

    # calculate problematic field components (limit to surfaces)
    if xma == 0:
        BxX = MAGx*(-arctan((ymb*zmc)/(xpa*PMM)) + arctan((ypb*zmc)/(xpa*PPM)) + arctan((ymb*zpc) /
                                                                                        (xpa*PMP)) - arctan((ypb*zpc)/(xpa*PPP)) + (pi/2)*(sign(ymb)-sign(ypb))*(sign(zmc)-sign(zpc)))
    elif xpa == 0:
        BxX = -MAGx*(-arctan((ymb*zmc)/(xma*MMM)) + arctan((ypb*zmc)/(xma*MPM)) + arctan((ymb*zpc) /
                                                                                         (xma*MMP)) - arctan((ypb*zpc)/(xma*MPP)) + (pi/2)*(sign(ymb)-sign(ypb))*(sign(zmc)-sign(zpc)))
    else:
        BxX = MAGx*(arctan((ymb*zmc)/(xma*MMM)) - arctan((ymb*zmc)/(xpa*PMM)) - arctan((ypb*zmc)/(xma*MPM)) + arctan((ypb*zmc)/(xpa*PPM))
                    - arctan((ymb*zpc)/(xma*MMP)) + arctan((ymb*zpc)/(xpa*PMP)) + arctan((ypb*zpc)/(xma*MPP)) - arctan((ypb*zpc)/(xpa*PPP)))

    if ymb == 0:
        ByY = MAGy*(-arctan((xma*zmc)/(MPM*ypb)) + arctan((xpa*zmc)/(PPM*ypb)) + arctan((xma*zpc) /
                                                                                        (MPP*ypb)) - arctan((xpa*zpc)/(PPP*ypb)) + (pi/2)*(sign(xma)-sign(xpa))*(sign(zmc)-sign(zpc)))
    elif ypb == 0:
        ByY = -MAGy*(-arctan((xma*zmc)/(MMM*ymb)) + arctan((xpa*zmc)/(PMM*ymb)) + arctan((xma*zpc) /
                                                                                         (MMP*ymb)) - arctan((xpa*zpc)/(PMP*ymb)) + (pi/2)*(sign(xma)-sign(xpa))*(sign(zmc)-sign(zpc)))
    else:
        ByY = MAGy*(arctan((xma*zmc)/(ymb*MMM)) - arctan((xpa*zmc)/(ymb*PMM)) - arctan((xma*zmc)/(ypb*MPM)) + arctan((xpa*zmc)/(ypb*PPM))
                    - arctan((xma*zpc)/(ymb*MMP)) + arctan((xpa*zpc)/(ymb*PMP)) + arctan((xma*zpc)/(ypb*MPP)) - arctan((xpa*zpc)/(ypb*PPP)))

    if zmc == 0:
        BzZ = MAGz*(-arctan((xma*ymb)/(MMP*zpc)) + arctan((xpa*ymb)/(PMP*zpc)) + arctan((xma*ypb) /
                                                                                        (MPP*zpc)) - arctan((xpa*ypb)/(PPP*zpc)) + (pi/2)*(sign(xma)-sign(xpa))*(sign(ymb)-sign(ypb)))
    elif zpc == 0:
        BzZ = -MAGz*(-arctan((xma*ymb)/(MMM*zmc)) + arctan((xpa*ymb)/(PMM*zmc)) + arctan((xma*ypb) /
                                                                                         (MPM*zmc)) - arctan((xpa*ypb)/(PPM*zmc)) + (pi/2)*(sign(xma)-sign(xpa))*(sign(ymb)-sign(ypb)))
    else:
        BzZ = MAGz*(arctan((xma*ymb)/(zmc*MMM)) - arctan((xpa*ymb)/(zmc*PMM)) - arctan((xma*ypb)/(zmc*MPM)) + arctan((xpa*ypb)/(zmc*PPM))
                    - arctan((xma*ymb)/(zpc*MMP)) + arctan((xpa*ymb)/(zpc*PMP)) + arctan((xma*ypb)/(zpc*MPP)) - arctan((xpa*ypb)/(zpc*PPP)))

    Bxtot = (BxX+BxY+BxZ)
    Bytot = (ByX+ByY+ByZ)
    Bztot = (BzX+BzY+BzZ)
    field = array([Bxtot, Bytot, Bztot])

    # add M when inside the box to make B out of H-------------
    if abs(x) < a:
        if abs(y) < b:
            if abs(z) < c:
                field += MAG

    return field

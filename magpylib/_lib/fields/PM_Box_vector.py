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


from numpy import arctan2, transpose, sqrt, log, pi, array
from warnings import warn


def Bfield_BoxV(MAG, POS, DIM):
    #magnetic field of the BOX - vectorized
    # no special cases
    # no addition of MAG on inside

    MAGx,MAGy,MAGz = MAG[:,0]/4/pi, MAG[:,1]/4/pi, MAG[:,2]/4/pi
    x,y,z = POS[:,0],POS[:,1],POS[:,2]
    a,b,c = DIM[:,0]/2,DIM[:,1]/2,DIM[:,2]/2

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

    BxX = MAGx*(arctan2((ymb*zmc),(xma*MMM)) - arctan2((ymb*zmc),(xpa*PMM)) - arctan2((ypb*zmc),(xma*MPM)) + arctan2((ypb*zmc),(xpa*PPM))
                    - arctan2((ymb*zpc),(xma*MMP)) + arctan2((ymb*zpc),(xpa*PMP)) + arctan2((ypb*zpc),(xma*MPP)) - arctan2((ypb*zpc),(xpa*PPP)))

    ByY = MAGy*(arctan2((xma*zmc),(ymb*MMM)) - arctan2((xpa*zmc),(ymb*PMM)) - arctan2((xma*zmc),(ypb*MPM)) + arctan2((xpa*zmc),(ypb*PPM))
                    - arctan2((xma*zpc),(ymb*MMP)) + arctan2((xpa*zpc),(ymb*PMP)) + arctan2((xma*zpc),(ypb*MPP)) - arctan2((xpa*zpc),(ypb*PPP)))

    BzZ = MAGz*(arctan2((xma*ymb),(zmc*MMM)) - arctan2((xpa*ymb),(zmc*PMM)) - arctan2((xma*ypb),(zmc*MPM)) + arctan2((xpa*ypb),(zmc*PPM))
                    - arctan2((xma*ymb),(zpc*MMP)) + arctan2((xpa*ymb),(zpc*PMP)) + arctan2((xma*ypb),(zpc*MPP)) - arctan2((xpa*ypb),(zpc*PPP)))

    Bxtot = (BxX+BxY+BxZ)
    Bytot = (ByX+ByY+ByZ)
    Bztot = (BzX+BzY+BzZ)
    field = array([Bxtot, Bytot, Bztot])

    return transpose(field)
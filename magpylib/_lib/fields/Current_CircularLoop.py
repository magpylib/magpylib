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

from numpy import sqrt, array, cos, sin, NaN
from magpylib._lib.mathLibPrivate import getPhi, ellipticK, ellipticE
from warnings import warn

# %% CIRCULAR CURRENT LOOP
# Describes the magnetic field of a circular current loop that lies parallel to
#    the x-y plane. The loop has the radius r0 and carries the current i0, the
#    center of the current loop lies at posCL

# i0   : float  [A]     current in the loop
# d0   : float  [mm]    diameter of the current loop
# posCL: arr3  [mm]     Position of the center of the current loop

# source: calculation from Filipitsch Diplo


def Bfield_CircularCurrentLoop(i0, d0, pos):

    px, py, pz = pos

    r = sqrt(px**2+py**2)
    phi = getPhi(px, py)
    z = pz

    r0 = d0/2  # radius of current loop

    # avoid singularity at CL
    #    print('WARNING: close to singularity - setting field to zero')
    #    return array([0,0,0])
    rr0 = r-r0
    if (-1e-12 < rr0 and rr0 < 1e-12):  # rounding to eliminate the .5-.55 problem when sweeping
        if (-1e-12 < z and z < 1e-12):
            warn('Warning: getB Position directly on current line', RuntimeWarning)
            return array([NaN, NaN, NaN])

    deltaP = sqrt((r+r0)**2+z**2)
    deltaM = sqrt((r-r0)**2+z**2)
    kappa = deltaP**2/deltaM**2
    kappaBar = 1-kappa

    # avoid discontinuity at r=0
    if (-1e-12 < r and r < 1e-12):
        Br = 0.
    else:
        Br = -2*1e-4*i0*(z/r/deltaM)*(ellipticK(kappaBar) -
                                      (2-kappaBar)/(2-2*kappaBar)*ellipticE(kappaBar))
    Bz = -2*1e-4*i0*(1/deltaM)*(-ellipticK(kappaBar)+(2-kappaBar -
                                                      4*(r0/deltaM)**2)/(2-2*kappaBar)*ellipticE(kappaBar))

    # transfer to cartesian coordinates
    Bcy = array([Br, 0., Bz])*1000.  # mT output
    T_Cy_to_Kart = array(
        [[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]])
    Bkart = T_Cy_to_Kart.dot(Bcy)

    return Bkart

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

from numpy import sqrt, arctan2, empty, pi
from magpylib._lib.mathLib_vector import ellipticKV, ellipticEV, angleAxisRotationV_priv
import numpy as np


# %% CIRCULAR CURRENT LOOP
# Describes the magnetic field of a circular current loop that lies parallel to
#    the x-y plane. The loop has the radius r0 and carries the current i0, the
#    center of the current loop lies at posCL

# i0   : float  [A]     current in the loop
# d0   : float  [mm]    diameter of the current loop
# posCL: arr3  [mm]     Position of the center of the current loop

# VECTORIZATION

def Bfield_CircularCurrentLoopV(I0, D, POS):

    R = D/2  #radius

    N = len(D)  # vector size

    X,Y,Z = POS[:,0],POS[:,1],POS[:,2]

    RR, PHI = sqrt(X**2+Y**2), arctan2(Y, X)      # cylindrical coordinates


    deltaP = sqrt((RR+R)**2+Z**2)
    deltaM = sqrt((RR-R)**2+Z**2)
    kappa = deltaP**2/deltaM**2
    kappaBar = 1-kappa

    # allocate solution vector
    field_R = empty([N])

    # avoid discontinuity of Br on z-axis
    maskRR0 = RR == np.zeros([N])
    field_R[maskRR0] = 0
    # R-component computation
    notM = np.invert(maskRR0)
    field_R[notM] = -2*1e-4*(Z[notM]/RR[notM]/deltaM[notM])*(ellipticKV(kappaBar[notM]) -
                        (2-kappaBar[notM])/(2-2*kappaBar[notM])*ellipticEV(kappaBar[notM]))
    
    # Z-component computation
    field_Z = -2*1e-4*(1/deltaM)*(-ellipticKV(kappaBar)+(2-kappaBar -
                                                      4*(R/deltaM)**2)/(2-2*kappaBar)*ellipticEV(kappaBar))

    # transformation to cartesian coordinates
    Bcy = np.array([field_R,np.zeros(N),field_Z]).T
    AX = np.zeros([N,3])
    AX[:,2] = 1
    Bkart = angleAxisRotationV_priv(PHI/pi*180,AX,Bcy)

    return (Bkart.T * I0).T * 1000 # to mT


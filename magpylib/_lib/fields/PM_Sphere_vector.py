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

def Bfield_SphereV(MAG, pos, D):  # returns array, takes (arr3, arr3, float)

    # this function is an extension of PM_Sphere

    MAGT = np.transpose(MAG)
    posT = np.transpose(pos)

    radius = D/2
    r = np.linalg.norm(pos,axis=1)

    map1 = r>radius
    map3 = r<radius

    B1T = map1*radius**3/3*(-MAGT/r**3 + 3*np.sum(MAG*pos,axis=1)*posT/r**5)
    B3T = map3*2/3*MAGT

    B = np.transpose(B1T+B3T)

    return B
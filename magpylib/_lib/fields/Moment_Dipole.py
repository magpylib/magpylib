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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from magpylib._lib.mathLibPrivate import fastSum3D
from numpy import pi, array, NaN
from warnings import warn

# %% DIPOLE field

# describes the field of a dipole positioned at posM and pointing into the direction of M

# M    : arr3  [mT]    Magnetic moment, M = µ0*m
# pos  : arr3  [mm]    Position of observer
# posM : arr3  [mm]    Position of dipole moment

# |M| corresponds to the magnetic moment of a cube with remanence Br and Volume V such that
#       |M| [mT*mm^3]  =  Br[mT] * V[mm^3]

def Bfield_Dipole(M, pos):
    R = pos
    rr = fastSum3D(R*R)
    mr = fastSum3D(M*R)

    if rr == 0:
        warn('Warning: getB Position directly on magnet surface', RuntimeWarning)
        return array([NaN, NaN, NaN])

    return (3*R*mr-M*rr)/rr**(5/2)/(4*pi)

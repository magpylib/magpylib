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

# MAGNETIC FIELD CALCULATION FOR SPHERICAL MAGNET IN CANONICAL BASIS


# %% IMPORTS
from magpylib._lib.mathLibPrivate import fastNorm3D, fastSum3D
from numpy import array, NaN
from warnings import warn

# %% CALCULATION

# The magnetic field of a spherical magnet with the center in the origin

# MAG = magnetization vector     [mT]  - takes arr3
# pos = position of the observer [mm]  - takes arr3
# D = diameter of sphere         [mm]  - takes float


def Bfield_Sphere(MAG, pos, D):  # returns array, takes (arr3, arr3, float)

    radius = D/2
    r = fastNorm3D(pos)

    if r > radius:
        return radius**3/3*(-MAG/r**3 + 3*fastSum3D(MAG*pos)*pos/r**5)
    elif r == radius:
        warn('Warning: getB Position directly on magnet surface', RuntimeWarning)
        return array([NaN, NaN, NaN])
    else:
        return 2/3*MAG

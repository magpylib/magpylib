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

# MAGNETIC FIELD CALCULATION OF CYLINDER IN CANONICAL BASIS


# %% IMPORTS
from numpy import pi, sqrt, array, arctan, cos, sin, arange, NaN
from magpylib._lib.mathLibPrivate import getPhi, elliptic
from warnings import warn

# %% Cylinder Field Calculation
# Describes the magnetic field of a cylinder with circular top and bottom and
#   arbitrary magnetization given by MAG. The axis of the cylinder is parallel
#   to the z-axis. The dimension are given by the radius r and the height h.
#   The center of the cylinder is positioned at the origin.

# basic functions required to calculate the diametral contributions
def Sphi(n, Nphi):
    if n == 0:
        return 1./3
    elif n == Nphi:
        return 1./3
    elif n % 2 == 1:
        return 4./3
    elif n % 2 == 0:
        return 2./3


# MAG  : arr3  [mT/mm³]    Magnetization vector (per unit volume)
# pos  : arr3  [mm]        Position of observer
# dim  : arr3  [mm]        dim = [d,h], Magnet diameter r and height h

# this calculation returns the B-field from the statrt as it is based on a current equivalent
def Bfield_Cylinder(MAG, pos, dim, Nphi0):  # returns arr3

    D, H = dim                # magnet dimensions
    R = D/2

    x, y, z = pos       # relative position
    r, phi = sqrt(x**2+y**2), getPhi(x, y)      # cylindrical coordinates

    # Mag part in z-direction
    B0z = MAG[2]  # z-part of magnetization
    zP, zM = z+H/2., z-H/2.   # some important quantitites
    Rpr, Rmr = R+r, R-r

    # special cases:
    #   0. volume cases      no quantities are zero
    #   1. on surfaces:      one quantity is zero
    #   2. on edge lines:    two quantities are zero
    CASE = 0
    for case in array([Rmr, zP, zM]):
        if (case < 1e-15 and -1e-15 < case):
            CASE += 1
    # rounding is required to catch numerical problem cases like .5-.55=.05000000000000001
    #   which then result in 'normal' cases but the square eliminates the small digits

    # edge cases ----------------------------------------------
    if CASE == 2:
        warn('Warning: getB Position directly on magnet surface', RuntimeWarning)
        return array([NaN, NaN, NaN])

    # on-magnet surface cases----------------------------------
    elif CASE == 1:
        if Rmr == 0:  # on cylinder surface
            if abs(z) < H/2:  # directly on magnet
                warn('Warning: getB Position directly on magnet surface', RuntimeWarning)
                return array([NaN, NaN, NaN])
        else:  # on top or bottom surface
            if Rmr > 0:  # directly on magnet
                warn('Warning: getB Position directly on magnet surface', RuntimeWarning)
                return array([NaN, NaN, NaN])

    # Volume Cases and off-magnet surface cases----------------

    SQ1 = sqrt(zP**2+Rpr**2)
    SQ2 = sqrt(zM**2+Rpr**2)

    alphP = R/SQ1
    alphM = R/SQ2
    betP = zP/SQ1
    betM = zM/SQ2
    kP = sqrt((zP**2+Rmr**2)/(zP**2+Rpr**2))
    kM = sqrt((zM**2+Rmr**2)/(zM**2+Rpr**2))
    gamma = Rmr/Rpr

    # radial field
    Br_Z = B0z*(alphP*elliptic(kP, 1, 1, -1)-alphM*elliptic(kM, 1, 1, -1))/pi
    Bx_Z = Br_Z*cos(phi)
    By_Z = Br_Z*sin(phi)

    # axial field
    Bz_Z = B0z*R/(Rpr)*(betP*elliptic(kP, gamma**2, 1, gamma) -
                        betM*elliptic(kM, gamma**2, 1, gamma))/pi

    Bfield = array([Bx_Z, By_Z, Bz_Z])  # contribution from axial magnetization

    # Mag part in xy-direction requires a numeical algorithm
    B0xy = sqrt(MAG[0]**2+MAG[1]**2)  # xy-magnetization amplitude
    if B0xy > 0:

        if MAG[0] > 0.:
            tetta = arctan(MAG[1]/MAG[0])
        elif MAG[0] < 0.:
            tetta = arctan(MAG[1]/MAG[0])+pi
        elif MAG[1] > 0:
            tetta = pi/2
        else:
            tetta = 3*pi/2

        if x > 0.:
            gamma = arctan(y/x)
        elif x < 0.:
            gamma = arctan(y/x)+pi
        elif y > 0:
            gamma = pi/2
        else:
            gamma = 3*pi/2
        phi = gamma-tetta

        phi0s = 2*pi/Nphi0  # discretization

        rR2 = 2*r*R
        r2pR2 = r**2+R**2

        def I1x(phi0, z0):
            if r2pR2-rR2*cos(phi-phi0) == 0:
                return -1/2/(z-z0)**2
            else:
                G = 1/sqrt(r2pR2-rR2*cos(phi-phi0)+(z-z0)**2)
                return (z-z0)*G/(r2pR2-rR2*cos(phi-phi0))

        # radial component
        Br_XY = B0xy*R/2/Nphi0*sum([
            sum([
                (-1)**(k+1)*Sphi(n, Nphi0)*cos(phi0s*n) *
                (r-R*cos(phi-phi0s*n))*I1x(phi0s*n, z0)
                for z0, k in zip([-H/2, H/2], [1, 2])])
            for n in arange(Nphi0+1)])
        # angular component
        Bphi_XY = B0xy*R**2/2/Nphi0*sum([
            sum([
                (-1)**(k+1)*Sphi(n, Nphi0)*cos(phi0s*n) *
                sin(phi-phi0s*n)*I1x(phi0s*n, z0)
                for z0, k in zip([-H/2, H/2], [1, 2])])
            for n in arange(Nphi0+1)])
        # axial component
        Bz_XY = B0xy*R/2/Nphi0*sum([
            sum([
                (-1)**k*Sphi(n, Nphi0)*cos(phi0s*n) /
                sqrt(r2pR2-rR2*cos(phi-phi0s*n)+(z-z0)**2)
                for z0, k in zip([-H/2, H/2], [1, 2])])
            for n in arange(Nphi0+1)])

        # translate r,phi to x,y coordinates
        phi = gamma
        Bx_XY = Br_XY*cos(phi)-Bphi_XY*sin(phi)
        By_XY = Br_XY*sin(phi)+Bphi_XY*cos(phi)

        Bfield = Bfield + array([Bx_XY, By_XY, Bz_XY])

        # add M if inside the cylinder to make B out of H
        if r < R and abs(z) < H/2:
            Bfield += array([MAG[0], MAG[1], 0])

    return Bfield

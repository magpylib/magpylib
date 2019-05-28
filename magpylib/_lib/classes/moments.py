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
######### Type hint definitions ########
# These aren't type hints, but look good 
# in Spyder IDE. Pycharm recognizes it.
from typing import Tuple
Mx=My=Mz=0.0 # Zero Dipole Moment
#######################################


# %% IMPORTS
from numpy import float64, isnan, array
from magpylib._lib.mathLibPrivate import angleAxisRotation
from magpylib._lib.fields.Moment_Dipole import Bfield_Dipole
from magpylib._lib.classes.base import MagMoment
from magpylib._lib.classes.fieldsampler import FieldSampler
from magpylib._lib.utility import checkDimensions, getBField, rotateToCS


class Dipole(MagMoment):
    """ 
    This class represents a magnetic dipole. The dipole is constructed such that 
    its moment :math:`|M|` is given in :math:`[mT*mm^3]` and corresponds to the moment of a cuboid
    magnet with remanence field Br and Volume V such that :math:`|M| = Br*V`. Scalar
    input is either integer or float. Vector input format can be either list,
    tuple or array of any data type (float, int).


    Parameters
    ----------

    moment : vec3 [mT]
        Set magnetic dipole moment in units of [mT*mm^3].

    pos=[0,0,0] : vec3 [mm]
        Set position of the moment in units of [mm].

    angle=0.0 : scalar [deg]
        Set angle of orientation of the moment in units of [deg].

    axis=[0,0,1] : vec3 []
        Set axis of orientation of the moment.

    Attributes
    ----------

    moment : arr3 [mT]
        Magnetic dipole moment in units of [mT*mm^3] (:math:`|moment| = Br*V` of a
        cuboid magnet.)

    position : arr3 [mm]
        Position of the moment in units of [mm].

    angle : float [deg]
        Angle of orientation of the moment in units of [deg].

    axis : arr3 []
        Axis of orientation of the moment.

    Examples
    --------
    >>> magpylib as magpy
    >>> mom = magpy.source.moment.Dipole(moment=[0,0,1000])
    >>> B = mom.getB([1,0,1])
    >>> print(B)
      [0.33761862  0.  0.11253954]

    Note
    ----
    The following Methods are available to all source objects.
    """

    def getB(self, pos):  # Particular Line current B field calculation. Check RCS for getB() interface
        rotatedPos = rotateToCS(pos, self)
        return getBField(Bfield_Dipole(self.moment, rotatedPos),  # The B field
                         self)  # Object Angle/Axis properties

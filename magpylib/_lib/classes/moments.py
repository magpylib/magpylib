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
Mx = My = Mz = 0.0  # Zero Dipole Moment
#######################################

# %% IMPORTS
from numpy import array, float64, ndarray
from magpylib._lib.mathLibPrivate import angleAxisRotation
from magpylib._lib.classes.base import MagMoment
from magpylib._lib.fields.Moment_Dipole import Bfield_Dipole


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
        # secure input type and check input format
        p1 = array(pos, dtype=float64, copy=False)
        # relative position between mag and obs
        posRel = p1 - self.position
        # rotate this vector into the CS of the magnet (inverse rotation)
        rotatedPos = angleAxisRotation(self.angle, -self.axis, posRel) # pylint: disable=invalid-unary-operand-type
        # rotate field vector back
        BCm = angleAxisRotation(self.angle, self.axis, Bfield_Dipole(self.moment, rotatedPos))
        # BCm is the obtained magnetic field in Cm
        # the field is well known in the magnet coordinates.
        return BCm

    def __repr__(self):
        """
        This is for the IPython Console
        When you call a defined dipole, this method shows you all its components.

        Examples
        --------
        >>> from magpylib import source
        >>> d = source.moment.Dipole([1.0,2.0,3.0], [3.0,2.0,1.0])
        >>> d
            name: Dipole 
            moment: x: 1.0mT, y: 2.0mT, z: 3.0mT 
            position: x: 3.0mm, y: 2.0mm, z:1.0mm 
            angle: 0.0 Degrees 
            axis: x: 0.0, y: 0.0, z: 1.0
        """
        return "type: {} \n moment: x: {}, y: {}, z: {} \n position: x: {}, y: {}, z:{} \n angle: {}  \n axis: x: {}, y: {}, z: {}".format("moments.Dipole", *self.moment, *self.position, self.angle, *self.axis)

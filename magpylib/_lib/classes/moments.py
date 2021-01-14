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

from numpy import array, float64, ndarray, ones, tile, shape
from magpylib._lib.mathLib import angleAxisRotation_priv
from magpylib._lib.classes.base import MagMoment
from magpylib._lib.fields.Moment_Dipole import Bfield_Dipole
from magpylib._lib.fields.Moment_Dipole_vector import Bfield_DipoleV
from magpylib._lib.mathLib_vector import angleAxisRotationV_priv
from magpylib._lib.utility import unit_prefix

# tool-tip / intellisense helpers ---------------------------------------------
# Class initialization is done purely by kwargs. While some # of these can be 
# set to zero by default other MUST be given to make any sense 
# (e.g. magnetization). To improve tool tips and intellisense we inizilize them
# with names, e.g. mag=(Mx, My, Mz). This looks good, but it requires that
# these names are pre-initialzed:
Mx = My = Mz = 0.0

# -------------------------------------------------------------------------------
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
        
         # vectorized code if input is an Nx3 array
        if type(pos) == ndarray:
            if len(shape(pos))==2: # list of positions - use vectorized code
                # vector size
                NN = shape(pos)[0] 
                # prepare vector inputs
                POSREL = pos - self.position
                ANG = ones(NN)*self.angle
                AX = tile(self.axis,(NN,1))
                MOM = tile(self.moment,(NN,1))
                # compute rotations and field
                ROTATEDPOS = angleAxisRotationV_priv(ANG, -AX, POSREL)
                BB = Bfield_DipoleV(MOM,ROTATEDPOS)
                BCM = angleAxisRotationV_priv(ANG, AX, BB)

                return BCM
        
        # secure input type and check input format
        p1 = array(pos, dtype=float64, copy=False)
        # relative position between mag and obs
        posRel = p1 - self.position
        # rotate this vector into the CS of the magnet (inverse rotation)
        rotatedPos = angleAxisRotation_priv(self.angle, -self.axis, posRel) # pylint: disable=invalid-unary-operand-type
        # rotate field vector back
        BCm = angleAxisRotation_priv(self.angle, self.axis, Bfield_Dipole(self.moment, rotatedPos))
        # BCm is the obtained magnetic field in Cm
        # the field is well known in the magnet coordinates.
        return BCm

    def __repr__(self):
        return '\n '.join((
            super().__repr__(),
            "moment: x={}T, y={}T, z={}T".format(*(unit_prefix(mom/1000) for mom in self.moment))
        ))
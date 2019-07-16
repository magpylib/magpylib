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
'''
Base Functions
==============
Define base classes here on which the magnetic source objects are built on

    1. RCS class: the underlying relative coordintate system initiates position
                    and orientation and provides object movement and rotation
                    functionalities.
    2. HomoMag class: initializes the homogeneous magnetization for all homogeneous
                    magnet classes
    3. LineCurrent class: initializes a current for all line current classes
    4. MagMoment class: initializes a Moment for all line moment classes

'''


######### Type hint definitions ########
# These aren't type hints, but look good 
# in Spyder IDE. Pycharm recognizes it.
Auto = 0 # Maximum cores, for multicore
        # function. if 0 find max.
numpyArray = 0
constant = None
Mx=My=Mz=0.0 # Zero Moment
#######################################

# %% IMPORTS
from numpy import array, float64, pi, isnan, array
from magpylib._lib.mathLibPrivate import Qmult, Qconj, getRotQuat, arccosSTABLE, fastSum3D, fastNorm3D
from magpylib._lib.utility import checkDimensions
from magpylib._lib.classes.fieldsampler import FieldSampler
import sys


# %% FUNDAMENTAL CLASS - RCS (RELATIVE COORDINATE SYSTEM)
#       - initiates position, orientation
#       - adds moveBY, rotateBy

class RCS:
    """
    FUNDAMENTAL CLASS - RCS (RELATIVE COORDINATE SYSTEM)

    initiates position, orientation
    - adds moveBY, rotateBy
    """

    def __init__(self, position, angle, axis):
        # fundamental (unit)-orientation/rotation is [0,0,0,1]
        assert any(
            ax != 0 for ax in axis), "Invalid Axis input for Sensor (0,0,0)"
        assert all(
            isinstance(ax, int) or isinstance(ax, float) for ax in axis), "Invalid Axis input for Sensor" + str(axis)

        self.position = array(position, dtype=float64, copy=False)
        try:
            self.angle = float(angle)
        except ValueError:
            sys.exit('Bad angle input')
        self.axis = array(axis, dtype=float64, copy=False)

        # check input format
        if any(isnan(self.position)) or len(self.position) != 3:
            sys.exit('Bad pos input')
        if any(isnan(self.axis)) or len(self.axis) != 3:
            sys.exit('Bad axis input')

    def setPosition(self, newPos):
        """
        This method moves the source to the position given by the argument 
        vector `newPos`. Vector input format can be either list, tuple or array
        of any data type (float, int)

        Parameters
        ----------
        newPos : vec3 [mm]
            Set new position of the source.

        Returns
        -------
        None

        Example
        -------
        >>> from magpylib import source
        >>> pm = source.magnet.Sphere(mag=[0,0,1000],dim=1)
        >>> print(pm.position)
            [0. 0. 0.]
        >>> pm.setPosition([5,5,5])
        >>> print(pm.position)
            [5. 5. 5.]
        """
        self.position = array(newPos, dtype=float64, copy=False)
        if any(isnan(self.position)) or len(self.position) != 3:
            sys.exit('Bad pos input')

    def move(self, displacement):
        """
        This method moves the source by the argument vector `displacement`. 
        Vector input format can be either list, tuple or array of any data
        type (float, int).

        Parameters
        ----------
        displacement : vec3 [mm]
            Set displacement vector

        Returns
        -------
        None

        Example
        -------
        >>> from magpylib import source
        >>> pm = source.magnet.Sphere(mag=[0,0,1000],dim=1,pos=[1,2,3])
        >>> print(pm.position)
            [1. 2. 3.]
        >>> pm.move([3,2,1])
        >>> print(pm.position)
            [4. 4. 4.]
        """
        mV = array(displacement, dtype=float64, copy=False)
        if any(isnan(mV)) or len(mV) != 3:
            sys.exit('Bad move vector input')
        self.position = self.position + mV

    def setOrientation(self, angle, axis):
        """
        This method sets a new source orientation given by `angle` and `axis`.
        Scalar input is either integer or float. Vector input format can be
        either list, tuple or array of any data type (float, int).

        Parameters
        ----------
        angle  : scalar [deg]
            Set new angle of source orientation.

        axis : vec3 []
            Set new axis of source orientation.

        Returns
        -------
        None            

        Example
        -------
        >>> from magpylib import source
        >>> pm = source.magnet.Sphere(mag=[0,0,1000],dim=1)
        >>> print([pm.angle,pm.axis])
            [0.0, array([0., 0., 1.])]
        >>> pm.setOrientation(45,[0,1,0])
        >>> print([pm.angle,pm.axis])
            [45.0, array([0., 1., 0.])]
        """
        try:
            self.angle = float(angle)
        except ValueError:
            sys.exit('Bad angle input')
        self.axis = array(axis, dtype=float64, copy=False)
        if any(isnan(self.axis)) or len(self.axis) != 3:
            sys.exit('Bad axis input')

    def rotate(self, angle, axis, anchor='self.position'):
        """
        This method rotates the source about `axis` by `angle`. The axis passes
        through the center of rotation anchor. Scalar input is either integer or
        float. Vector input format can be either list, tuple or array of any
        data type (float, int).

        Parameters
        ----------
        angle  : scalar [deg]
            Set angle of rotation in units of [deg]
        axis : vec3 []
            Set axis of rotation
        anchor : vec3 [mm]
            Specify the Center of rotation which defines the position of the
            axis of rotation. If not specified the source will rotate about its
            own center.

        Returns
        -------
        None

        Example
        -------
        >>> from magpylib import source
        >>> pm = source.magnet.Sphere(mag=[0,0,1000], dim=1)
        >>> print(pm.position, pm.angle, pm.axis)
          [0. 0. 0.] 0.0 [0. 0. 1.]
        >>> pm.rotate(90, [0,1,0], anchor=[1,0,0])
        >>> print(pm.position, pm.angle, pm.axis)
          [1., 0., 1.] 90.0 [0., 1., 0.]
        """
        # secure type
        ax = array(axis, dtype=float64, copy=False)

        try:
            ang = float(angle)
        except ValueError:
            sys.exit('Bad angle input')

        if str(anchor) == 'self.position':
            anchor = self.position
        else:
            anchor = array(anchor, dtype=float64, copy=False)

        # check input
        if any(isnan(ax)) or len(ax) != 3:
            sys.exit('Bad axis input')
        if fastSum3D(ax**2) == 0:
            sys.exit('Bad axis input')
        if any(isnan(anchor)) or len(anchor) != 3:
            sys.exit('Bad anchor input')

        # determine Rotation Quaternion Q from self.axis-angle
        Q = getRotQuat(self.angle, self.axis)

        # determine rotation Quaternion P from rot input
        P = getRotQuat(ang, ax)

        # determine new orientation quaternion which follows from P.Q v (P.Q)*
        R = Qmult(P, Q)

        # reconstruct new axis-angle from new orientation quaternion
        ang3 = arccosSTABLE(R[0])*180/pi*2

        ax3 = R[1:]  # konstanter mult faktor ist wurscht für ax3
        self.angle = ang3
        if ang3 == 0:  # avoid returning a [0,0,0] axis
            self.axis = array([0, 0, 1])
        else:
            Lax3 = fastNorm3D(ax3)
            self.axis = array(ax3)/Lax3

        # set new position using P.v.P*
        posOld = self.position-anchor
        Vold = [0] + [p for p in posOld]
        Vnew = Qmult(P, Qmult(Vold, Qconj(P)))
        self.position = array(Vnew[1:])+anchor


# %% HOMOGENEOUS MAGNETIZATION CLASS
#       - initiates magnetization

class HomoMag(RCS, FieldSampler):

    def __init__(self, position, angle, axis, magnetization):

        # inherit class RCS
        RCS.__init__(self, position, angle, axis)
        assert all(
            a == 0 for a in magnetization) is False, "Bad mag input, all values are zero"

        # secure input type and check input format of mag
        self.magnetization = array(magnetization, dtype=float64, copy=False)
        assert (not any(isnan(self.magnetization)) and len(
            self.magnetization) == 3), "Bad mag input, invalid vector dimension"


# %% LINE CURRENT CLASS
#       - initiates current

class LineCurrent(RCS, FieldSampler):

    def __init__(self, position, angle, axis, current):

        # inherit class RCS
        RCS.__init__(self, position, angle, axis)

        # secure input types and check input format
        try:
            self.current = float(current)
        except ValueError:
            sys.exit('Bad current input')


class MagMoment(RCS, FieldSampler):

    def __init__(self, moment=(Mx, My, Mz),
                 pos=(0.0, 0.0, 0.0),
                 angle=0.0, axis=(0.0, 0.0, 1.0)):

        # inherit class RCS
        RCS.__init__(self, pos, angle, axis)

        # secure input type and check input format of moment
        self.moment = checkDimensions(3, moment, "Bad moment input")

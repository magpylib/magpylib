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

# tool-tip / intellisense helpers -----------------------------------------------
# Class initialization is done purely by kwargs. While some # of these can be 
# set to zero by default other MUST be given to make any sense 
# (e.g. magnetization). To improve tool tips and intellisense we inizilize them
# with names, e.g. mag=(Mx, My, Mz). This looks good, but it requires that
# these names are pre-initialzed:
Auto = 0 # Maximum cores, for multicore
        # function. if 0 find max.
numpyArray = 0
constant = None
Mx=My=Mz=0.0 # Zero Moment


# -------------------------------------------------------------------------------
from numpy import array, float64, pi, isnan, array
from magpylib._lib.mathLib import Qmult, Qconj, getRotQuat, arccosSTABLE, fastSum3D, fastNorm3D
from magpylib._lib.utility import checkDimensions, unit_prefix
import sys


# -------------------------------------------------------------------------------
# FUNDAMENTAL CLASS - RCS (RELATIVE COORDINATE SYSTEM)
#       - initiates position, orientation
#       - adds moveBY, rotateBy

class RCS:
    """
    base class RCS(RELATIVE COORDINATE SYSTEM)

    initiates position, orientation (angle, axis)
    adds methods setPosition, move, setOrientation, rotate
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

    def __repr__(self):
        name = getattr(self,'name',None)
        str_repr = [f"type: {type(self).__name__}"]
        if name is not None:
            str_repr.append(f"name: {name}")
        str_repr.extend([
            "position: x={}m, y={}m, z={}m".format(*(unit_prefix(pos/1000) for pos in self.position)),
            "angle: {}Degrees".format(unit_prefix(self.angle)),
            "axis: x={:.2f}, y={:.2f}, z={:.2f}".format(*(ax for ax in self.axis))
        ])
        return '\n '.join(str_repr)

    def _repr_html_(self):
        table = [[s for s in rs.strip().split(':')] for rs in self.__repr__().split('\n')]
        head = '\n'.join(f'''<th>{s}</th>\n''' for s in table[0])
        body = '\n'.join(
            '<tr>\n'+''.join(
                f'''<td>{''.join(
                    '<font color="{}">{}</font>'.format(color,ss) if i!=0 else ss for color,ss in zip(('red','green','blue'),s2.split(','))
                )}</td>\n''' for i,s2 in enumerate(s)
            )+'</tr>' for s in table[1:])
        return f'''<table border="1" class="dataframe">
        <thead>
            <tr style="text-align: right;">
            {head}
            </tr>
        </thead>
        <tbody>
            {body}
        </tbody>
        </table> '''

#------------------------------------------------------------------------------
class FieldSampler:
    """
    Field Sampler Class
    
    This class initiates the getB method and is inherited by all source objects.
    The main reason this is centralized here is that the docstring of the
    getB method is inherited by all sources.
    """

    def getB(self, pos):
        """
        This method returns the magnetic field vector generated by the source
        at the argument position `pos` in units of [mT]

        Parameters
        ----------
        pos : vec3 [mm] Position or list of Positions where magnetic field
            should be determined.


        Returns
        -------
        magnetic field vector : arr3 [mT] Magnetic field at the argument
            position `pos` generated by the source in units of [mT].
        """
        # Return a list of vec3 results
        # This method will be overriden by the classes that inherit it.
        # Throw a warning and return 0s if it somehow isn't.
        # Note: Collection() has its own docstring
        # for getB since it inherits nothing.
        import warnings
        warnings.warn(
            "called getB method is not implemented in this class,"
            "returning [0,0,0]", RuntimeWarning)
        return [0, 0, 0]



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------



# subclass for HOMOGENEOUS MAGNETS --------------------------------------
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


# subclass for LINE CURRENTS ---------------------------------------------
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


# subclass for MOMENTS ----------------------------------------------------
#       - initiates nothing
class MagMoment(RCS, FieldSampler):

    def __init__(self, moment=(Mx, My, Mz),
                 pos=(0.0, 0.0, 0.0),
                 angle=0.0, axis=(0.0, 0.0, 1.0)):

        # inherit class RCS
        RCS.__init__(self, pos, angle, axis)

        # secure input type and check input format of moment
        self.moment = checkDimensions(3, moment, "Bad moment input")

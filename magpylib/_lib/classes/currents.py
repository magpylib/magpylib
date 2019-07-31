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

####### Type hint definitions ########
from typing import List, Tuple, TypeVar
# Magnetization Vector Typehints
x_i = TypeVar('x_i', int, float)
y_i = TypeVar('y_i', int, float)
z_i = TypeVar('z_i', int, float)
listOfPos = List[Tuple[x_i, y_i, z_i]]

I = 0.0  # Default Current
d = 0.0  # Default Diameter
######################################

# %% IMPORTS
from numpy import array, float64, ndarray
from magpylib._lib.mathLibPrivate import angleAxisRotation
from magpylib._lib.fields.Current_Line import Bfield_CurrentLine
from magpylib._lib.fields.Current_CircularLoop import Bfield_CircularCurrentLoop
from magpylib._lib.classes.base import LineCurrent

# %% THE CIRCULAR CLASS
class Circular(LineCurrent):
    """ 
    A circular line current loop with diameter `dim` and a current `curr` flowing
    in positive orientation. In the canonical basis (position=[0,0,0], angle=0.0,
    axis=[0,0,1]) the loop lies in the x-y plane with the origin at its center.
    Scalar input is either integer or float. Vector input format can be
    either list, tuple or array of any data type (float, int).

    Parameters
    ----------

    curr : scalar [A]
        Set current in loop in units of [A]

    dim : float [mm]
        Set diameter of current loop in units of [mm]

    pos=[0,0,0] : vec3 [mm]
        Set position of the center of the current loop in units of [mm].

    angle=0.0 : scalar [deg]
        Set angle of orientation of current loop in units of [deg].

    axis=[0,0,1] : vec3 []
        Set axis of orientation of the current loop.

    Attributes
    ----------

    current : float [A]
        Current in loop in units of [A]

    dimension : float [mm]
        Loop diameter in units of [mm]

    position : arr3 [mm]
        Position of center of loop in units of [mm]

    angle : float [deg]
        Angle of orientation of the current loop.

    axis : arr3 []
        Axis of orientation of the current loop.

    Example
    -------
    >>> from magpylib import source
    >>> cd = source.current.Circular(curr=10,dim=2)
    >>> B = cd.getB([0,0,2])
    >>> print(B)
      [0.         0.         0.56198518]

    Note
    ----
    The following Methods are available to all sources objects.
    """

    def __init__(self, curr=I, dim=d, pos=(0.0, 0.0, 0.0), angle=0.0, axis=(0.0, 0.0, 1.0)):

        # inherit class lineCurrent
        #   - pos, Mrot, MrotInv, curr
        #   - moveBy, rotateBy
        LineCurrent.__init__(self, pos, angle, axis, curr)

        # secure input type and check input format of dim
        assert dim >= 0, 'Bad input dimension'
        self.dimension = float(dim)

    def getB(self, pos):  # Particular Circular current B field calculation. Check RCS for getB() interface
        # secure input type and check input format
        p1 = array(pos, dtype=float64, copy=False)
        # relative position between mag and obs
        posRel = p1 - self.position
        # rotate this vector into the CS of the magnet (inverse rotation)
        rotatedPos = angleAxisRotation(self.angle, -self.axis, posRel) # pylint: disable=invalid-unary-operand-type
        # rotate field vector back
        BCm = angleAxisRotation(self.angle, self.axis, Bfield_CircularCurrentLoop(self.current,self.dimension,rotatedPos))
        # BCm is the obtained magnetic field in Cm
        # the field is well known in the magnet coordinates.
        return BCm

    def __repr__(self):
        """
         This is for the IPython Console
        When you call a defined circular, this method shows you all its components.

        Examples
        --------
        >>> from magpylib import source
        >>> c = source.current.Circular(2.45, 3.1469, [4.4, 5.24, 0.5])
        >>> c
            type: current.Circular 
            current: 2.45  
            dimension: d: 3.1469 
            position: x: 4.4, y: 5.24, z: 0.5
            angle: 0.0 
            axis: x: 0.0, y: 0.0, z: 1.0
        """
        return "type: {} \n current: {}  \n dimension: d: {} \n position: x: {}, y: {}, z: {} \n angle: {}  \n axis: x: {}, y: {}, z: {}".format("current.Circular", self.current, self.dimension, *self.position, self.angle, *self.axis)

# %% THE CIRCUAR CL CLASS


class Line(LineCurrent):
    """ 

    A line current flowing along linear segments from vertex to vertex given by
    a list of positions `vertices` in the canonical basis (position=[0,0,0], angle=0.0,
    axis=[0,0,1]). Scalar input is either integer or float. Vector input format
    can be either list, tuple or array of any data type (float, int).


    Parameters
    ----------

    curr : scalar [A]
        Set current in loop in units of [A]

    vertices : vecNx3 [mm]
        N positions given in units of [mm] that make up N-1 linear segments
        along which the current `curr` flows, starting from the first position
        and ending with the last one.
        [[x,y,z], [x,y,z], ...]
        "[pos1,pos2,...]"
    pos=[0,0,0] : vec3 [mm]
        Set reference position of the current distribution in units of [mm].

    angle=0.0 : scalar [deg]
        Set angle of orientation of current distribution in units of [deg].

    axis=[0,0,1] : vec3 []
        Set axis of orientation of the current distribution.

    Attributes
    ----------

    current : float [A]
        Current flowing along line in units of [A].

    vertices : arrNx3 [mm]
        Positions of line current vertices in units of [mm].

    position : arr3 [mm]
        Reference position of line current in units of [mm].

    angle : float [deg]
        Angle of orientation of line current in units of [deg].

    axis : arr3 []
        Axis of orientation of the line current.

    Examples
    --------
    >>> from magpylib import source
    >>> from numpy import sin,cos,pi,linspace
    >>> vertices = [[cos(phi),sin(phi),0] for phi in linspace(0,2*pi,36)]
    >>> cd = source.current.Line(curr=10,vertices=vertices)
    >>> B = cd.getB([0,0,2])
    >>> print(B)
      [-6.24500451e-17  1.73472348e-18  5.59871233e-01]


    Note
    ----
    The following Methods are available to all sources objects.
    """

    def __init__(self, curr=I, vertices=listOfPos, pos=(0.0, 0.0, 0.0), angle=0.0, axis=(0.0, 0.0, 1.0)):

        # inherit class lineCurrent
        #   - pos, Mrot, MrotInv, curr
        #   - moveBy, rotateBy
        LineCurrent.__init__(self, pos, angle, axis, curr)

        # secure input type and check input format of dim
        assert isinstance(vertices, list) or isinstance(
            vertices, ndarray), 'Line Current: enter a list of position vertices - Ex: Line(vertices=[(1,2,3),(3,2,1)])'
        assert all(isinstance(pos, tuple) or isinstance(pos, list)
                   or isinstance(pos, ndarray) for pos in vertices), 'Line-current: Input position (3D) tuples or lists within the list - Ex: Line(vertices=[(1,2,3),(3,2,1)])'
        assert all(len(
            d) == 3 for d in vertices), 'Line-current: Bad input dimension, vectors in list must be 3D'
        self.vertices = array(vertices, dtype=float64, copy=False)

    def getB(self, pos):  # Particular Line current B field calculation. Check RCS for getB() interface
        # secure input type and check input format
        p1 = array(pos, dtype=float64, copy=False)
        # relative position between mag and obs
        posRel = p1 - self.position
        # rotate this vector into the CS of the magnet (inverse rotation)
        rotatedPos = angleAxisRotation(self.angle, -self.axis, posRel) # pylint: disable=invalid-unary-operand-type
        # rotate field vector back
        BCm = angleAxisRotation(self.angle, self.axis, Bfield_CurrentLine(rotatedPos, self.vertices, self.current))
        # BCm is the obtained magnetic field in Cm
        # the field is well known in the magnet coordinates.
        return BCm

    def __repr__(self):
        """
        This is for the IPython Console
        When you call a defined line, this method shows you all its components.

        Examples
        --------
        >>> from magpylib import source
        >>> l = source.current.Line(2.45, [[2, .35, 2], [10, 2, -4], [4, 2, 1], [102, 2, 7]], [4.4, 5.24, 0.5])
        >>> l
            type: current.Line 
            current: 2.45 
            dimensions: vertices
            position: x: 4.4, y: 5.24, z: 0.5
            angle: 0.0 
            axis: x: 0.0, y: 0.0, z: 1.0
        """
        return "type: {} \n current: {} \n dimensions: vertices \n position: x: {}, y: {}, z: {} \n angle: {}  \n axis: x: {}, y: {}, z: {}".format("current.Line", self.current, *self.position, self.angle, *self.axis)

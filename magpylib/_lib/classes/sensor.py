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

from magpylib._lib.classes.base import RCS
from magpylib._lib.utility import addUniqueSource, addListToCollection, isSource
from magpylib._lib.mathLib import rotatePosition

class Sensor(RCS):
    """
    Create a rotation-enabled sensor to extract B-fields from individual Sources and Source Collections.
    It may be displayed with :class:`~magpylib.Collection`'s :meth:`~magpylib.Collection.displaySystem` using the sensors kwarg.

    Parameters
    ----------
    position : vec3
        Cartesian position of where the sensor is.

    angle : scalar
        Angle of rotation

    axis : vec3
        Rotation axis information (x,y,z)

    Example
    -------
        >>> from magpylib import source, Sensor
        >>> sensor = Sensor([0,0,0],90,(0,0,1)) # This sensor is rotated in respect to space
        >>> cyl = source.magnet.Cylinder([1,2,300],[1,2])
        >>> absoluteReading = cyl.getB([0,0,0])
        >>> print(absoluteReading)
            [  0.552   1.105  268.328 ]
        >>> relativeReading = sensor.getB(cyl)
        >>> print(relativeReading)
            [  1.105  -0.552  268.328 ]
    """

    def __init__(self, pos=[0, 0, 0], angle=0, axis=[0, 0, 1]):
        RCS.__init__(self, pos, angle, axis)

    def __repr__(self):
        return f"\n name: Sensor"\
               f"\n position x: {self.position[0]} mm  n y: {self.position[1]}mm z: {self.position[2]}mm"\
               f"\n angle: {self.angle} Degrees"\
               f"\n axis: x: {self.axis[0]}   n y: {self.axis[1]} z: {self.axis[2]}"

    def getB(self, *sources, dupWarning=True):
        """Extract the magnetic field based on the Sensor orientation

        Parameters
        ----------
        dupWarning : Check if there are any duplicate sources, optional.
            This will prevent duplicates and throw a warning, by default True.

        Returns
        -------
        [vec3]
            B-Field as perceived by the sensor

        Example
        -------
        >>> from magpylib import source, Sensor
        >>> sensor = Sensor([0,0,0],90,(0,0,1)) # This sensor is rotated in respect to space
        >>> cyl = source.magnet.Cylinder([1,2,300],[1,2])
        >>> absoluteReading = cyl.getB([0,0,0])
        >>> print(absoluteReading)
            [  0.552   1.105  268.328 ]
        >>> relativeReading = sensor.getB(cyl)
        >>> print(relativeReading)
            [  1.105  -0.552  268.328 ]
        """
        # Check input, add Collection list
        sourcesList = []
        for s in sources:
            try:
                addListToCollection(sourcesList, s.sources, dupWarning)
            except AttributeError:
                if isinstance(s, list) or isinstance(s, tuple):
                    addListToCollection(sourcesList, s, dupWarning)
                else:
                    assert isSource(s), "Argument " + str(s) + \
                        " in addSource is not a valid source for Collection"
                    if dupWarning is True:
                        addUniqueSource(s, sourcesList)
                    else:
                        sourcesList += [s]

        # Read the field from all nominated sources
        Btotal = sum([s.getB(self.position) for s in sources])
        return rotatePosition(Btotal,
                              -self.angle,  # Rotate in the opposite direction
                              self.axis,
                              [0, 0, 0])

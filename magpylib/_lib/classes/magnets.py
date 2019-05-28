# -------------------------------------------------------------------------------
# magpylib -- A Python 3 toolbox for calculating magnetic fields from
# permanent magnets and current distributions.
# Copyright (C) 2019  Michael Ortner <magpylib@gmail.com>
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
#
# For contact information, reach out over at <magpylib@gmail.com> or our issues
# page at https://www.github.com/OrtnerMichael/magpylib/issues.
# -------------------------------------------------------------------------------
######### Type hint definitions ########
# These aren't type hints, but look good 
# in Spyder IDE. Pycharm recognizes it.
from typing import Tuple
Mx=My=Mz=0.0 # Def.Magnetization Vector
a=b=c=0.0 #Default Cuboid dimensions
d=0.0 # Default Diameter 
h=0.0 # Default Height 
Max=0 # Multicore flag
#######################################


# %% IMPORTS
from numpy import float64, isnan, array
from magpylib._lib.mathLibPrivate import angleAxisRotation
import sys
from magpylib._lib.fields.PM_Box import Bfield_Box
from magpylib._lib.fields.PM_Cylinder import Bfield_Cylinder
from magpylib._lib.fields.PM_Sphere import Bfield_Sphere
from magpylib._lib.classes.base import HomoMag
from magpylib._lib.utility import checkDimensions, getBField, rotateToCS

# %% THE CUBE CLASS


class Box(HomoMag):
    """ 
    A homogeneously magnetized cuboid magnet. In 
    the canonical basis (position=[0,0,0], angle=0.0, axis=[0,0,1]) the magnet
    has the origin at its geometric center and the sides of the box are parallel
    to the basis vectors. Scalar input is either integer or float. 
    Vector input format can be either list, tuple or array of any data type (float, int).


    Parameters
    ----------

    mag : vec3 [mT]
        Set magnetization vector of magnet in units of [mT].

    dim : vec3 [mm]
        Set the size of the box. dim=[a,b,c] which anchorresponds to the three
        side lenghts of the box in units of [mm].

    pos=[0,0,0] : vec3 [mm]
        Set position of the center of the magnet in units of [mm].

    angle=0.0 : scalar [deg]
        Set angle of orientation of magnet in units of [deg].

    axis=[0,0,1] : vec3 []
        Set axis of orientation of the magnet.

    Attributes
    ----------
    magnetization : arr3 [mT]
        Magnetization vector of box in units of [mT].

    dimension : arr3 [mm]
        Magnet dimension=[a,b,c] which anchorrespond to the three side lenghts
        of the box in units of [mm] in x-,y- and z-direction respectively
        in the canonical basis.

    position : arr3 [mm]
        Position of the center of the magnet in units of [mm].

    angle : float [deg]
        Angle of orientation of the magnet in units of [deg].

    axis : arr3 []
        Axis of orientation of the magnet.

    Example
    -------
    >>> from magpylib import source
    >>> pm = source.magnet.Box(mag=[0,0,1000],dim=[1,1,1])
    >>> B = pm.getB([1,0,1])
    >>> print(B)
      [4.29223532e+01 1.76697482e-14 1.37461635e+01]

    Note
    ----
    The following Methods are available to all sources objects.
    """

    def __init__(self, mag=(Mx, My, Mz), dim=(a, b, c), pos=(0.0, 0.0, 0.0), angle=0.0, axis=(0.0, 0.0, 1.0)):

        # inherit class HomoMag
        HomoMag.__init__(self, pos, angle, axis, mag)

        # secure input type and check input format of dim
        self.dimension = checkDimensions(3, dim, "Bad dim for box")

    # Private method. This is used by getB for multiprocess reference.
    def getB(self, pos):
        rotatedPos = rotateToCS(pos, self)
        return getBField(Bfield_Box(self.magnetization, rotatedPos, self.dimension),  # The B field
                         self)  # Object Angle/Axis properties


# %% THE CYLINDER CLASS

class Cylinder(HomoMag):
    """ 
    A homogeneously magnetized cylindrical magnet. 
    The magnet is initialized in the canonical basis (position=[0,0,0],
    angle=0.0, axis=[0,0,1]) with the geometric center at the origin and the
    central symmetry axis pointing in z-direction so that the circular bottom
    lies in a plane parallel to the xy-plane. Scalar input is either integer
    or float and reflects a round bottom. 
    Vector input format can be either list, tuple or array of any
    data type (float, int).

    Parameters
    ----------
    mag : vec3 [mT]
        Set magnetization vector of magnet in units of [mT].

    dim : vec2 [mm]
        Set the size of the cylinder. dim=[D,H] which are diameter and height
        of the cylinder in units of [mm] respectively.

    pos=[0,0,0] : vec3 [mm]
        Set position of the center of the magnet in units of [mm].

    angle=0.0 : scalar [deg]
        Set angle of orientation of magnet in units of [deg].

    axis=[0,0,1] : vec3 []
        Set axis of orientation of the magnet.

    iterDia=50 : int []
        Set number of iterations for calculation of B-field from non-axial 
        magnetization. Lower values will make the calculation faster but
        less precise.

    Attributes
    ----------
    magnetization : arr3 [mT]
        Magnetization vector of magnet in units of [mT].

    dimension : arr2 [mm]
        Magnet dimension=[d,h] which anchorrespond to diameter and height of the
        cylinder in units of [mm].

    position : arr3 [mm]
        Position of the center of the magnet in units of [mm].

    angle : float [deg]
        Angle of orientation of the magnet in units of [deg].

    axis : arr3 []
        Axis of orientation of the magnet.

    iterDia : int []
        Number of iterations for calculation of B-field from non-axial
        magnetization. Lower values will make the calculation faster but less
        precise.

    Example
    -------
    >>> from magpylib import source
    >>> pm = source.magnet.Cylinder(mag=[0,0,1000],dim=[1,1])
    >>> B = pm.getB([1,0,1])
    >>> print(B)
      [34.31662243  0.         10.16090915]

    Note
    ----
    The following Methods are available to all sources objects.
    """

    def __init__(self, mag=(Mx, My, Mz), dim=(d, h), pos=(0.0, 0.0, 0.0), angle=0.0, axis=(0.0, 0.0, 1.0), iterDia=50):

        # inherit class homoMag
        #   - pos, Mrot, MrotInv, mag
        #   - moveBy, rotateBy
        HomoMag.__init__(self, pos, angle, axis, mag)

        # secure input type and check input format of dim
        assert type(
            iterDia) == int, 'Bad iterDia input for cylinder, expected <class int> got ' + str(type(iterDia))
        self.dimension = checkDimensions(2, dim, "Bad dim input for cylinder")
        self.iterDia = iterDia

    def getB(self, pos):  # Particular Cylinder B field calculation. Check RCS for getB() interface
        rotatedPos = rotateToCS(pos, self)
        return getBField(Bfield_Cylinder(self.magnetization, rotatedPos, self.dimension, self.iterDia),  # The B field
                         self)  # Object Angle/Axis properties


# %% THE SPHERE CLASS

class Sphere(HomoMag):
    """ 
    A homogeneously magnetized sphere. The magnet
    is initialized in the canonical basis (position=[0,0,0],
    angle=0.0, axis=[0,0,1]) with the center at the origin. Scalar input is
    either integer or float. Vector input format can be either list, tuple
    or array of any data type (float, int).

    Parameters
    ----------

    mag : vec3 [mT]
        Set magnetization vector of magnet in units of [mT].

    dim : float [mm]
        Set diameter of the sphere in units of [mm].

    pos=[0,0,0] : vec3 [mm]
        Set position of the center of the magnet in units of [mm].

    angle=0.0 : scalar [deg]
        Set angle of orientation of magnet in units of [deg].

    axis=[0,0,1] : vec3 []
        Set axis of orientation of the magnet.

    Attributes
    ----------

    magnetization : arr3 [mT]
        Magnetization vector of magnet in units of [mT].

    dimension : float [mm]
        Sphere diameter in units of [mm].

    position : arr3 [mm]
        Position of the center of the magnet in units of [mm].

    angle : float [deg]
        Angle of orientation of the magnet in units of [deg].

    axis : arr3 []
        Axis of orientation of the magnet.

    Example
    -------
    >>> from magpylib import source
    >>> pm = source.magnet.Sphere(mag=[0,0,1000],dim=1)
    >>> B = pm.getB([1,0,1])
    >>> print(B)
      [22.09708691  0.          7.36569564]

    Note
    ----
    The following Methods are available to all sources objects.
    """

    def __init__(self, mag=(Mx, My, Mz), dim=d, pos=(0.0, 0.0, 0.0), angle=0.0, axis=(0.0, 0.0, 1.0)):

        # inherit class homoMag
        #   - pos, Mrot, MrotInv, mag
        #   - moveBy, rotateBy
        HomoMag.__init__(self, pos, angle, axis, mag)

        # secure input type and check input format of dim
        self.dimension = float(dim)
        assert self.dimension > 0, 'Bad dim<=0 for sphere'

    def getB(self, pos):
        rotatedPos = rotateToCS(pos, self)
        return getBField(Bfield_Sphere(self.magnetization, rotatedPos, self.dimension),  # The B Field
                         self)  # Object Angle/Axis properties

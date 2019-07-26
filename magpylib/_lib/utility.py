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
from typing import Tuple
from numpy import float64, isnan, array
# Helper function for validating input dimensions


def checkDimensions(expectedD: int, dim: Tuple[float, float, float], exitMsg: str = "Bad dim input") -> array:
    if type(dim) == int or type(dim) == float:
        dim = [dim]
    assert all(coord == 0 for coord in dim) is False, exitMsg + \
        ", all values are zero"
    dimension = array(dim, dtype=float64, copy=False)
    assert (not any(isnan(dimension)) and len(dimension) == expectedD), exitMsg
    return dimension

# Collection Helpers


def addListToCollection(sourceList, inputList, dupWarning):
    assert all(isSource(a)
               for a in inputList), "Non-source object in Collection initialization"
    if dupWarning is True:  # Skip iterating both lists if warnings are off
        for source in inputList:
            # Checks if source is in list, throw warning
            addUniqueSource(source, sourceList)
    else:
        sourceList.extend(inputList)


def isSource(theObject: any) -> bool:
    """
    Check is an object is a magnetic source.

    Parameter
    ---------
        theObject: any
            Object to be evaluated if it is a source. Update list when new sources are up
    Returns
    -------
        bool
    """
    from magpylib import source
    sourcesList = (
        source.magnet.Box,
        source.magnet.Sphere,
        source.magnet.Cylinder,
        source.current.Line,
        source.current.Circular,
        source.moment.Dipole)
    return any(isinstance(theObject, src) for src in sourcesList)

def isSensor(theObject: any) -> bool:
    from magpylib._lib.classes.sensor import Sensor 
    return isinstance(theObject,Sensor)


def addUniqueSource(source, sourceList):
    import warnings
    if source not in sourceList:
        sourceList += [source]
    else:
        warnings.warn("Source " + str(source) +
                      " already in Collection list; Ignoring", Warning)
####


def drawMagnetizationVector(position, magnetization, angle, axis, color, SYSSIZE, ax):
    """Draw the magnetization vector of a magnet.

    Parameters
    ----------
    position : vec3
        position of the magnet
    magnetization : vec3
        magnetization vector
    angle : float
        angle of rotation
    axis : vec3
        Axis of rotation
    color : matplotlib color
        Color of the axis. No default value specified
    SYSSIZE : float
        Size of the display syste
    ax : [pyploy]
        pyplot canvas to draw on

    """
    from magpylib._lib.mathLibPublic import rotatePosition
    M = rotatePosition(magnetization, angle, axis)
    P = position
    # Get a lil different but unique tone
    c = [color[0]/2, color[1]/2, color[2]/2, color[3]]
    ax.quiver(P[0], P[1], P[2],  # X,Y,Z position
                  M[0], M[1], M[2],  # Components of the Vector
                  normalize=True,
                  length=SYSSIZE,
                  color=c)

def drawSensor(sensor, SYSSIZE, ax):
    """Draw the sensor coordinates

    Parameters
    ----------
    sensor: Sensor
        Sensor to draw
    SYSSIZE : float
        Size of the display system
    ax : [pyplot]
        pyplot canvas to draw on

    """
    from magpylib._lib.mathLibPublic import rotatePosition
    M = rotatePosition([1,0,0],sensor.angle,sensor.axis)
    P = sensor.position
    ax.quiver(P[0], P[1], P[2],  # X position
              M[0], M[1], M[2],  # Components of the Vector
                  normalize=True,
                  length=SYSSIZE/4,
                  color='r')
    ax.text(M[0]+P[0], M[1]+P[1], M[2]+P[2], "x", None)
    
    M = rotatePosition([0,1,0],sensor.angle,sensor.axis)
    ax.quiver(P[0], P[1], P[2],  # Y position
              M[0], M[1], M[2],  # Components of the Vector
                  normalize=True,
                  length=SYSSIZE/4,
                  color='g')
    ax.text(M[0]+P[0], M[1]+P[1], M[2]+P[2], "y", None)
    
    M = rotatePosition([0,0,1],sensor.angle,sensor.axis)
    ax.quiver(P[0], P[1], P[2],  # Z position
              M[0], M[1], M[2],  # Components of the Vector
                  normalize=True,
                  length=SYSSIZE/4,
                  color='b')
    ax.text(M[0]+P[0], M[1]+P[1], M[2]+P[2], "z", None)

def drawMagAxis(magnetList, SYSSIZE, ax):
    """
    Draws the magnetization vectors of magnet objects in a list.

    Parameters
    ----------
    magnetList: [list]
        list of magnet objects with a "color" attribute.
        Do source.color = 'k' in the meantime if there isnt any
        before appending it to the list.

    SYSSIZE : [float]
        [Size of the display system]
    pyplot : [pyplot]
        [Pyplot canvas]

    """

    for s in magnetList:
        drawMagnetizationVector(s.position, s.magnetization,
                                s.angle, s.axis, s.color,
                                SYSSIZE, ax)

####


def drawLineArrows(vertices, current, SYSSIZE, ax):
    """
    Helper function for Collection.displaySystem()
    Draw Arrows inside the line to show current orientation

    Parameters
    ----------
    vertices : [list]
            A list of position lists of each vertix.
    current : [float]
            The current. Polarity Inverts the orientation.
    SYSSIZE : [type]
            Size of the System for controlling arrow size.
    pyplot : [pyplot]
            The pyplot instance

    """

    lenli = len(vertices)
    for v in range(0, len(vertices)-1):
        # Get last position if current is position
        x = vertices[(-(v+1), v)[current <= 0]]
        y = vertices[(-((v+2) % lenli), (v+1) % lenli)
                     [current <= 0]]  # Get second to last
        ax.quiver((x[0]+y[0])/2, (x[1]+y[1])/2, (x[2]+y[2])/2,  # Mid point in line
                      # Components of the Vector
                      x[0]-y[0], x[1]-y[1], x[2]-y[2],
                      normalize=True,
                      length=SYSSIZE/12,
                      color='k')

        ax.quiver(y[0], y[1], y[2],  # Arrow at start
                      # Components of the Vector
                      x[0]-y[0], x[1]-y[1], x[2]-y[2],
                      normalize=True,
                      length=SYSSIZE/12,
                      color='k')


def drawCurrentArrows(currentList, SYSSIZE, ax):
    for s in currentList:
        drawLineArrows(s.vertices, s.current, SYSSIZE, ax)

###


def drawDipole(position, moment, angle, axis, SYSSIZE, ax):
    """
    Draw a dipole moment arrow.

    Parameters
    ----------
    position : vec3
        position of the dipole
    moment : vec3
        orientation vector of the dipole
    SYSSIZE : float
        size of the display
    pyplot : pyplot
        canvas to draw on

    """
    from magpylib._lib.mathLibPublic import rotatePosition
    P = rotatePosition(position, angle, axis)
    M = rotatePosition(moment, angle, axis)
    
    ax.quiver(P[0], P[1], P[2],  # X,Y,Z position
                  M[0], M[1], M[2],  # Components of the Vector
                  normalize=True,
                  length=SYSSIZE/12,
                  color='k')


# Source package helpers


def recoordinateAndGetB(source_ref, args):
    ## Used in base.RCS.getBDisplacement(),
    # Take an object, a sample position to place the object in
    # and magnet orientation arguments.
    # Apply the new position, orient it, and return the B field value from position Bpos.
    Bpos = args[0]
    Mpos = args[1]
    MOrient = args[2]
    angle = MOrient[0]
    axis = MOrient[1]

    assert isPosVector(Mpos)
    assert isPosVector(Bpos)
    assert isPosVector(axis)
    assert isinstance(angle, float) or isinstance(angle, int)

    source_ref.setPosition(Mpos)
    # if len(MOrient)==3:
    #     anchor = MOrient[3]
    #     assert isPosVector(anchor)
    #     source_ref.rotate(  angle,
    #                         axis,
    #                         anchor)
    # else:
    source_ref.setOrientation(angle,
                              axis)

    return source_ref.getB(Bpos)


def isPosVector(object_ref):
    # Return true if the object reference is that of
    # a position array.
    from numpy import array, ndarray
    try:
        if (isinstance(object_ref, list) or isinstance(object_ref, tuple) or isinstance(object_ref, ndarray) or isinstance(object_ref, array)):
            if len(object_ref) == 3:
                return all(isinstance(int(coordinate), int) for coordinate in object_ref)
    except Exception:
        return False


def initializeMulticorePool(processes):
    # Helper for setting up Multicore pools.
    from multiprocessing import Pool, cpu_count
    if processes == 0:
        # Identify how many workers the host machine can take.
        processes = cpu_count() - 1
        # Using all cores is USUALLY a bad idea.
    assert processes > 0, "Could not identify multiple cores for getB. This machine may not support multiprocessing."
    return Pool(processes=processes)


def isDisplayMarker(object_ref):
    m = object_ref
    if len(m) == 3:  # Check if it's [numeric,numeric,numeric]
        return all(isinstance(p, int) or isinstance(p, float) for p in m)
    if len(m) == 4:  # Check if it's [numeric,numeric,numeric,"label"]
        return all(isinstance(p, int) or isinstance(p, float) for p in m[:2]) and isinstance(m[3], str)



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

from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy import array, amax, linspace, pi, sin, cos, finfo
from magpylib._lib.classes.magnets import Box, Cylinder, Sphere
from magpylib._lib.classes.currents import Line, Circular
from magpylib._lib.classes.moments import Dipole
from magpylib._lib.classes.sensor import Sensor
from magpylib._lib.classes.base import FieldSampler
from magpylib._lib.utility import drawCurrentArrows, drawMagAxis, drawDipole, isDisplayMarker
from magpylib._lib.utility import drawSensor, isSensor
from magpylib._lib.mathLib import angleAxisRotation_priv
from magpylib._lib.mathLib import angleAxisRotation
from magpylib import Collection


# tool-tip / intellisense helpers -----------------------------------------------
# Class initialization is done purely by kwargs. While some # of these can be 
# set to zero by default other MUST be given to make any sense 
# (e.g. magnetization). To improve tool tips and intellisense we inizilize them
# with names, e.g. mag=(Mx, My, Mz). This looks good, but it requires that
# these names are pre-initialzed:
x=y=z=0.0 # Position Vector
sensor1=sensor2="sensor type"
numpyArray=[[x,y,z]] # List of Positions
listOfPos=[[x,y,z]] # List of Positions
listOfSensors=[sensor1,sensor2] # List of Sensors



# -------------------------------------------------------------------------------
def displaySystem(sources, markers=listOfPos, subplotAx=None,
                        sensors=listOfSensors, suppress=False, direc=False):
    """
    Shows the collection system in an interactive pyplot and returns a matplotlib figure identifier.

    WARNING
    -------
    As a result of an inherent problem in matplotlib the 
    Poly3DCollections z-ordering fails when bounding boxes intersect.


    Parameters
    ----------
    markers : list[scalar,scalar,scalar,[label]]
        List of position vectors to add visual markers to the display, optional label.
        Default: [[0,0,0]]

    Example
    -------
    >>> from magpylib import Collection, source
    >>> c=source.current.Circular(3,7)
    >>> x = Collection(c)
    >>> marker0 = [0,0,0,"Neutral Position"]
    >>> marker1 = [10,10,10]
    >>> x.displaySystem(markers=[ marker0,
    ...                           marker1])

    Parameters
    ----------
    sensors : list[sensor]
        List of :class:`~magpylib.Sensor` objects to add the display.
        Default: None

    Example
    -------
    >>> from magpylib import Collection, source
    >>> c=source.current.Circular(3,7)
    >>> x = Collection(c)
    >>> sensor0 = Sensor()
    >>> sensor1 = Sensor(pos=[1,2,3], angle=180)
    >>> x.displaySystem(sensors=[ sensor0,
    ...                           sensor1])


    Parameters
    ----------
    suppress : bool
        If True, only return Figure information, do not show. Interactive mode must be off.
        Default: False.


    Example
    -------
    >>> ## Suppress matplotlib.pyplot.show() 
    >>> ## and returning figure from showing up
    >>> from matplotlib import pyplot 
    >>> pyplot.ioff()
    >>> figureData = Collection.displayFigure(suppress=True)

    Parameters
    ----------
    direc : bool
        Set to True to show current directions and magnetization vectors.
        Default: False


    Return    
    ------
    matplotlib Figure object
        graphics object is displayed through plt.show()

    Example
    -------
    >>> from magpylib import source, Collection
    >>> pm1 = source.magnet.Box(mag=[0,0,1000],dim=[1,1,1],pos=[-1,-1,-1],angle=45,axis=[0,0,1])
    >>> pm2 = source.magnet.Cylinder(mag=[0,0,1000],dim=[2,2],pos=[0,-1,1],angle=45,axis=[1,0,0])
    >>> pm3 = source.magnet.Sphere(mag=[0,0,1000],dim=3,pos=[-2,1,2],angle=45,axis=[1,0,0])
    >>> C1 = source.current.Circular(curr=100,dim=6)
    >>> col = Collection(pm1,pm2,pm3,C1)
    >>> col.displaySystem()
    
    Parameters
    ----------
    subplotAx : matplotlib subplot axe instance
        Use an existing matplotlib subplot instance to draw the 3D system plot into.
        Default: None
    
    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from magpylib.source.magnet import Box
    >>> from magpylib import Collection
    >>> #create collection of one magnet
    >>> s1 = Box(mag=[ 500,0, 500], dim=[3,3,3], pos=[ 0,0, 3], angle=45, axis=[0,1,0])
    >>> c = Collection(s1)
    >>> #create positions
    >>> xs = np.linspace(-8,8,100)
    >>> zs = np.linspace(-6,6,100)
    >>> posis = [[x,0,z] for z in zs for x in xs]
    >>> #calculate fields
    >>> Bs = c.getBsweep(posis)
    >>> #reshape array and calculate amplitude
    >>> Bs = np.array(Bs).reshape([100,100,3])
    >>> Bamp = np.linalg.norm(Bs,axis=2)
    >>> X,Z = np.meshgrid(xs,zs)
    >>> # Define figure
    >>> fig = plt.figure()
    >>> ## Define ax for 2D
    >>> ax1 = fig.add_subplot(1, 2, 1, axisbelow=True)
    >>> ## Define ax for 3D displaySystem
    >>> ax2 = fig.add_subplot(1, 2, 2, axisbelow=True,projection='3d')
    >>> ## field plot 2D
    >>> ax1.contourf(X,Z,Bamp,100,cmap='rainbow')
    >>> U,V = Bs[:,:,0], Bs[:,:,2]
    >>> ax1.streamplot(X, Z, U, V, color='k', density=2)
    >>> ## plot Collection system in 3D ax subplot
    >>> c.displaySystem(subplotAx=ax2)
    
    Raises
    ------
    AssertionError
        If Marker position list is poorly defined. i.e. listOfPos=(x,y,z) instead of lisOfPos=[(x,y,z)]
    """
    
    collection = Collection(sources)

    if subplotAx is None:
        fig = plt.figure(dpi=80, figsize=(8, 8))
        ax = fig.gca(projection='3d')
    else:
        ax = subplotAx

    # count magnets
    Nm = 0
    for s in collection.sources:
        if type(s) is Box or type(s) is Cylinder or type(s) is Sphere:
            Nm += 1
    
    cm = plt.cm.hsv  # Linter complains about this but it is working pylint: disable=no-member
    # select colors
    colors = [cm(x) for x in linspace(0, 1, Nm+1)]

    ii = -1
    SYSSIZE = finfo(float).eps  # Machine Epsilon for moment
    dipolesList = []
    magnetsList = []
    sensorsList = []
    currentsList = []
    markersList = []

    # Check input and Add markers to the Markers list before plotting
    for m in markers:
        assert isDisplayMarker(m), "Invalid marker definition in displaySystem:" + str(
            m) + ". Needs to be [vec3] or [vec3,string]"
        markersList += [m]
    
    for s in sensors:    
        if s == sensor1:
            continue
        else:
            assert isSensor(s), "Invalid sensor definition in displaySystem:" + str(
            s) 
            sensorsList.append(s)

    for s in collection.sources:
        if type(s) is Box:
            ii += 1  # increase color counter
            P = s.position
            D = s.dimension/2
            # create vertices in canonical basis
            v0 = array([D, D*array([1, 1, -1]), D*array([1, -1, -1]), D*array([1, -1, 1]),
                        D*array([-1, 1, 1]), D*array([-1, 1, -1]), -D, D*array([-1, -1, 1])])
            # rotate vertices + displace
            v = array([angleAxisRotation_priv(s.angle, s.axis, d)+P for d in v0])
            # create faces
            faces = [[v[0], v[1], v[2], v[3]],
                        [v[0], v[1], v[5], v[4]],
                        [v[4], v[5], v[6], v[7]],
                        [v[2], v[3], v[7], v[6]],
                        [v[0], v[3], v[7], v[4]],
                        [v[1], v[2], v[6], v[5]]]
            # plot
            boxf = Poly3DCollection(
                faces, facecolors=colors[ii], linewidths=0.5, edgecolors='k', alpha=1)
            ax.add_collection3d(boxf)
            # check system size
            maxSize = amax(abs(v))
            if maxSize > SYSSIZE:
                SYSSIZE = maxSize

            if direc is True:
                s.color = colors[ii]
                magnetsList.append(s)
        elif type(s) is Cylinder:
            ii += 1  # increase color counter
            P = s.position
            R, H = s.dimension/2

            resolution = 20

            # vertices
            phis = linspace(0, 2*pi, resolution)
            vertB0 = array([[R*cos(p), R*sin(p), -H] for p in phis])
            vertT0 = array([[R*cos(p), R*sin(p), H] for p in phis])
            # rotate vertices+displacement
            vB = array(
                [angleAxisRotation_priv(s.angle, s.axis, d)+P for d in vertB0])
            vT = array(
                [angleAxisRotation_priv(s.angle, s.axis, d)+P for d in vertT0])
            # faces
            faces = [[vT[i], vB[i], vB[i+1], vT[i+1]]
                        for i in range(resolution-1)]
            faces += [vT, vB]
            # plot
            coll = Poly3DCollection(
                faces, facecolors=colors[ii], linewidths=0.5, edgecolors='k', alpha=1)
            ax.add_collection3d(coll)
            # check system size
            maxSize = max([amax(abs(vB)), amax(abs(vT))])
            if maxSize > SYSSIZE:
                SYSSIZE = maxSize

            if direc is True:
                s.color = colors[ii]
                magnetsList.append(s)

        elif type(s) is Sphere:
            ii += 1  # increase color counter
            P = s.position
            R = s.dimension/2

            resolution = 12

            # vertices
            phis = linspace(0, 2*pi, resolution)
            thetas = linspace(0, pi, resolution)
            vs0 = [[[R*cos(phi)*sin(th), R*sin(phi)*sin(th), R*cos(th)]
                    for phi in phis] for th in thetas]
            # rotate vertices + displacement
            vs = array(
                [[angleAxisRotation_priv(s.angle, s.axis, v)+P for v in vss] for vss in vs0])
            # faces
            faces = []
            for j in range(resolution-1):
                faces += [[vs[i, j], vs[i+1, j], vs[i+1, j+1], vs[i, j+1]]
                            for i in range(resolution-1)]
            # plot
            boxf = Poly3DCollection(
                faces, facecolors=colors[ii], linewidths=0.5, edgecolors='k', alpha=1)
            ax.add_collection3d(boxf)
            # check system size
            maxSize = amax(abs(vs))
            if maxSize > SYSSIZE:
                SYSSIZE = maxSize

            if direc is True:
                s.color = colors[ii]
                magnetsList.append(s)

        elif type(s) is Line:
            P = s.position
            vs0 = s.vertices
            # rotate vertices + displacement
            vs = array(
                [angleAxisRotation_priv(s.angle, s.axis, v)+P for v in vs0])
            # plot
            ax.plot(vs[:, 0], vs[:, 1], vs[:, 2], lw=1, color='k')
            # check system size
            maxSize = amax(abs(vs))
            if maxSize > SYSSIZE:
                SYSSIZE = maxSize

            if direc is True:
                # These don't move in the original object,
                sCopyWithVertices = deepcopy(s)
                sCopyWithVertices.vertices = vs  # We just draw the frame rotation, discard changes
                currentsList.append(sCopyWithVertices)

        elif type(s) is Circular:
            P = s.position
            R = s.dimension/2

            resolution = 20

            # vertices
            phis = linspace(0, 2*pi, resolution)
            vs0 = array([[R*cos(p), R*sin(p), 0] for p in phis])
            # rotate vertices + displacement
            vs = array(
                [angleAxisRotation_priv(s.angle, s.axis, v)+P for v in vs0])
            # plot
            ax.plot(vs[:, 0], vs[:, 1], vs[:, 2], lw=1, color='k')
            # check system size
            maxSize = amax(abs(vs))
            if maxSize > SYSSIZE:
                SYSSIZE = maxSize

            if direc is True:
                # Send the Circular vertice information
                sCopyWithVertices = deepcopy(s)
                sCopyWithVertices.vertices = vs  # to the object drawing list
                currentsList.append(sCopyWithVertices)

        elif type(s) is Dipole:
            P = angleAxisRotation(s.position, s.angle, s.axis)
            maxSize = amax(abs(P))
            if maxSize > SYSSIZE:
                SYSSIZE = maxSize

            dipolesList.append(s)

    for m in markersList:  # Draw Markers
        ax.scatter(m[0], m[1], m[2], s=20, marker='x')
        if(len(m) > 3):
            zdir = None
            ax.text(m[0], m[1], m[2], m[3], zdir)
        # Goes up to 3rd Position
        maxSize = max([abs(pos) for pos in m[:3]])
        if maxSize > SYSSIZE:
            SYSSIZE = maxSize

    for s in sensorsList: # Draw Sensors
        maxSize = max([abs(pos) for pos in s.position])
        if maxSize > SYSSIZE:
            SYSSIZE = maxSize
        drawSensor(s,SYSSIZE,ax)

    for d in dipolesList:
        drawDipole(d.position, d.moment,
                    d.angle, d.axis,
                    SYSSIZE, ax)

    if direc is True:  # Draw the Magnetization axes and current directions
        drawCurrentArrows(currentsList, SYSSIZE, ax)
        drawMagAxis(magnetsList, SYSSIZE, ax)

    #for tick in ax.xaxis.get_ticklabels()+ax.yaxis.get_ticklabels()+ax.zaxis.get_ticklabels():
    #    tick.set_fontsize(12)
    ax.set_xlabel('x[mm]')#, fontsize=12)
    ax.set_ylabel('y[mm]')#, fontsize=12)   #change font size through rc parameters
    ax.set_zlabel('z[mm]')#, fontsize=12)
    ax.set(
        xlim=(-SYSSIZE, SYSSIZE),
        ylim=(-SYSSIZE, SYSSIZE),
        zlim=(-SYSSIZE, SYSSIZE),
    )
    
    plt.tight_layout()

    if suppress == True:
        return plt.gcf()
    else:
        plt.show()
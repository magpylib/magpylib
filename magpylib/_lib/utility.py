from typing import Tuple
from numpy import float64, isnan, array
## Helper function for validating input dimensions
def checkDimensions(expectedD: int, dim: Tuple[float,float,float], exitMsg: str="Bad dim input") -> array:
    if type(dim)==int or type(dim)==float:
        dim=[dim]
    assert all(coord == 0 for coord in dim) is False, exitMsg + ", all values are zero"
    dimension = array(dim, dtype=float64, copy=False) 
    assert (not any(isnan(dimension))  and  len(dimension) == expectedD), exitMsg
    return dimension

##### Collection Helpers

def isSource(theObject : any) -> bool:
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
    return any(type(theObject) == src for src in sourcesList)

def addUniqueSource(source,sourceList):
    import warnings
    if source not in sourceList:
        sourceList += [source]
    else:
        warnings.warn("Source " + str(source) + " already in Collection list; Ignoring", Warning)
####
def drawMagnetizationVector(position,magnetization,angle,axis,color,SYSSIZE,pyplot):
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
    pyplot : [pyploy]
        pyplot canvas to draw on
    
    """
    from magpylib._lib.mathLibPublic import rotatePosition
    M = rotatePosition(magnetization,angle,axis) 
    P=position
    c=[color[0]/2,color[1]/2,color[2]/2,color[3]] ## Get a lil different but unique tone
    pyplot.quiver(P[0],P[1],P[2], # X,Y,Z position
                    M[0],M[1],M[2], # Components of the Vector
                    normalize=True,
                    length=SYSSIZE,
                    color=c)

def drawMagAxis(magnetList,SYSSIZE,pyplot):
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
            drawMagnetizationVector(s.position,s.magnetization,
                                    s.angle,s.axis,s.color,
                                    SYSSIZE,pyplot)

####

def drawLineArrows(vertices,current,SYSSIZE,pyplot):
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
    for v in range(0,len(vertices)-1):
                    x = vertices[(-(v+1),v)[current<=0]] #Get last position if current is position
                    y = vertices[(-((v+2)%lenli),(v+1)%lenli)[current<=0]] #Get second to last 
                    pyplot.quiver((x[0]+y[0])/2,(x[1]+y[1])/2,(x[2]+y[2])/2, # Mid point in line
                               x[0]-y[0],x[1]-y[1],x[2]-y[2], # Components of the Vector
                               normalize=True,
                               length=SYSSIZE/12,
                               color='k')
                    
                    pyplot.quiver(y[0],y[1],y[2], # Arrow at start
                               x[0]-y[0],x[1]-y[1],x[2]-y[2], # Components of the Vector
                               normalize=True,
                               length=SYSSIZE/12,
                               color='k')

def drawCurrentArrows(currentList,SYSSIZE,pyplot):
    for s in currentList:
            drawLineArrows(s.vertices,s.current,SYSSIZE,pyplot)

###

def drawDipole(position,moment,angle,axis,SYSSIZE,pyplot):
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
    P = rotatePosition(position,angle,axis)
    M = rotatePosition(moment,angle,axis) 
    
    pyplot.quiver(P[0],P[1],P[2], # X,Y,Z position
                M[0],M[1],M[2], # Components of the Vector
                normalize=True,
                length=SYSSIZE/12,
                color='k')


### Source package helpers

def rotateToCS(pos,source_ref):
        from magpylib._lib.mathLibPrivate import angleAxisRotation
        #secure input type and check input format   
        p1 = array(pos, dtype=float64, copy=False)
        
        #relative position between mag and obs
        posRel = p1 - source_ref.position

        #rotate this vector into the CS of the magnet (inverse rotation)
        p21newCm = angleAxisRotation(source_ref.angle,-source_ref.axis,posRel) # Leave this alone for now pylint: disable=invalid-unary-operand-type

        return p21newCm


def getBField(BCm,source_ref):
        # BCm is the obtained magnetic field in Cm
        #the field is well known in the magnet coordinates
        from magpylib._lib.mathLibPrivate import angleAxisRotation
        #rotate field vector back
        B = angleAxisRotation(source_ref.angle,source_ref.axis,BCm)
        
        return B

def recoordinateAndGetB(source_ref,newPos=[0,0,0],rotationArgs=(0,(0,0,1)),Bpos=[0,0,0]):
        source_ref.setPosition(newPos)
        source_ref.rotate(rotationArgs[0],rotationArgs[1])
        return source_ref.getB(Bpos)

def initializeMulticorePool(processes):
        from multiprocessing import Pool, cpu_count
        if processes == 0:
            processes = cpu_count() - 1 ## Identify how many workers the host machine can take. 
                                        ## Using all cores is USUALLY a bad idea.
        assert processes > 0, "Could not identify multiple cores for getB. This machine may not support multiprocessing."
        return Pool(processes=processes) 
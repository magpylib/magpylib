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
    return any(isinstance(theObject,src) for src in sourcesList)

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
        ## Used in all getB()
        from magpylib._lib.mathLibPrivate import angleAxisRotation
        #secure input type and check input format   
        p1 = array(pos, dtype=float64, copy=False)
        
        #relative position between mag and obs
        posRel = p1 - source_ref.position

        #rotate this vector into the CS of the magnet (inverse rotation)
        p21newCm = angleAxisRotation(source_ref.angle,-source_ref.axis,posRel) # Leave this alone for now pylint: disable=invalid-unary-operand-type

        return p21newCm


def getBField(BCm,source_ref):
        ## Used in all getB()
        # BCm is the obtained magnetic field in Cm
        #the field is well known in the magnet coordinates
        from magpylib._lib.mathLibPrivate import angleAxisRotation
        #rotate field vector back
        B = angleAxisRotation(source_ref.angle,source_ref.axis,BCm)
        
        return B


def recoordinateAndGetB(source_ref,args):
        ## Used in base.RCS.getBDisplacement(),

        # Take an object, a position to place the object in and magnet rotation arguments.
        # Apply the new position, rotate it, and return the B field value from position Bpos.
        Bpos = args[0]
        Mpos = args[1]
        MOrient = args[2]
        angle = MOrient[0]
        axis = MOrient[1]
        
        assert isPosVector(Mpos)
        assert isPosVector(Bpos)
        assert isPosVector(axis)
        assert isinstance(angle,float) or isinstance(angle,int)
        
        source_ref.setPosition(Mpos)
        if len(MOrient)==3:
            anchor = MOrient[3]
            assert isPosVector(anchor)
            source_ref.setOrientation(  angle,
                                        axis,
                                        anchor)    
        else:
            source_ref.setOrientation(  angle,
                                        axis)

        return source_ref.getB(Bpos)

def isPosVector(object_ref):
    # Return true if the object reference is that of 
    # a position array.
    from numpy import array, ndarray
    try:
        if ( isinstance(object_ref,list) or isinstance(object_ref,tuple) or isinstance(object_ref,ndarray) or isinstance(object_ref,array) ):
            if len(object_ref) == 3:
                return all(isinstance(int(coordinate),int) for coordinate in object_ref)
    except Exception:
        return False


def initializeMulticorePool(processes):
    # Helper for setting up Multicore pools.
    from multiprocessing import Pool, cpu_count
    if processes == 0:
        processes = cpu_count() - 1 ## Identify how many workers the host machine can take. 
                                    ## Using all cores is USUALLY a bad idea.
    assert processes > 0, "Could not identify multiple cores for getB. This machine may not support multiprocessing."
    return Pool(processes=processes) 

def posVectorFinder(dArray,positionsList):
    # Explore an array and append all the indexed values 
    # that are position vectors to the given list.
    for index in range(len(dArray)):
        if isPosVector(dArray[index]):
            positionsList.append(dArray[index])
        else:
            posVectorFinder(dArray[index],positionsList) # Recursively call itself to explore all dimensions

def equalizeListOfPos(listOfPos,listOfRotations,neutralPos=[0,0,0]):
    ERR_REDUNDANT = "Both list of positions and Rotations are uninitizalized for getBdisplacement, so function call is redundant. Use getB for a single position"
    ERR_UNEVENLISTS = "List of Positions is of different size than list of rotations. Enter repeating values or neutral values for matching Position and Rotation"
    # Check if either list is omitted, 
    # if only one is omitted then fill the other with neutral elements so they are equalized.
    assert listOfPos is not None or listOfRotations is not None, ERR_REDUNDANT
    if listOfPos == None:
        listOfPos = [neutralPos for n in range(len(listOfRotations))]
    else:
        if listOfRotations == None:
            listOfRotations = [(0,(0,0,1)) for n in range(len(listOfPos))]
    
    assert len(listOfPos)==len(listOfRotations), ERR_UNEVENLISTS
    return (listOfPos,listOfRotations)
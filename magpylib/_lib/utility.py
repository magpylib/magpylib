from typing import Tuple
from numpy import float64, isnan, array
## Helper function for validating input dimensions
def checkDimensions(expectedD: int, dim: Tuple[float,float,float], exitMsg: str="Bad dim input") -> array:
    assert all(coord == 0 for coord in dim) is False, exitMsg + ", all values are zero"
    dimension = array(dim, dtype=float64, copy=False) 
    assert (not any(isnan(dimension))  and  len(dimension) == expectedD), exitMsg
    return dimension

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
                    x = vertices[(-(v+1),v)[current>0]] #Get last position if current is position
                    y = vertices[(-((v+2)%lenli),(v+1)%lenli)[current>0]] #Get second to last 
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
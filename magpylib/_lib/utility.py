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

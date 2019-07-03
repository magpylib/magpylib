from magpylib._lib.classes.base import RCS 
from magpylib._lib.classes.collection import Collection 
from magpylib._lib.utility import addListToCollection, addUniqueSource, isSource, sensorRotate
from itertools import repeat

class Sensor(RCS):
    """
    Create a rotation-enabled sensor to extract B-fields from Source and Source Collections.

    Parameters
    ----------
    position : vec3
        Cartesian position of where the sensor is.
    
    angle : scalar
        Angle of rotation

    Attributes
    ----------
    sources : list of source objects
        List of all sources that have been added to the collection.

    Example
    -------
        >>> from magpylib import Collection, Sensor
        >>> pm = source.magnet.Box(mag=[0,0,1000],dim=[1,1,1])
        >>> B = pm.getB([1,0,1])
        >>> print(B)
        [ 0.53614741 1.87571635 2.8930498 ]
    """
    def __init__(self,position, angle = 0, axis = (0,0,1), anchor = (0,0,0)):
        self.position = position
        self.angle = angle
        self.anchor = anchor
        self.axis = axis
    
    def getB(self,*sources, dupWarning=True):
        Btotal = sum([s.getB(self.position) for s in sources])
        return sensorRotate(self,Btotal)
        

from magpylib._lib.classes.base import RCS 
from magpylib._lib.utility import sensorRotate, isSource
from itertools import repeat
from numpy import float64, angle, array, isnan
import sys
class Sensor(RCS):
    """
    Create a rotation-enabled sensor to extract B-fields from individual Sources and Source Collections.

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
    
    def __init__(self, position = [0, 0, 0], angle = 0, axis = [0, 0, 1]):
        # fundamental (unit)-orientation/rotation is [0,0,0,1]

        self.position = array(position, dtype=float64, copy=False)
        try:
            self.angle = float(angle)
        except ValueError:
            sys.exit('Bad angle input')
        assert any(ax!=0 for ax in axis), "Invalid Axis input for Sensor (0,0,0)"
        self.axis = array(axis, dtype=float64, copy=False)

        # check input format
        if any(isnan(self.position)) or len(self.position) != 3:
            sys.exit('Bad pos input')
        if any(isnan(self.axis)) or len(self.axis) != 3:
            sys.exit('Bad axis input')

    def __repr__(self):
        return f"\n name: Sensor"\
               f"\n position x: {self.position[0]} mm  n y: {self.position[1]}mm z: {self.position[2]}mm"\
               f"\n angle: {self.angle} Degrees"\
               f"\n axis: x: {self.axis[0]}   n y: {self.axis[1]} z: {self.axis[2]}"

    def getB(self,*sources, dupWarning=True):
        assert all(isSource(source) for source in sources)
        Btotal = sum([s.getB(self.position) for s in sources])
        return sensorRotate(self,Btotal)
        

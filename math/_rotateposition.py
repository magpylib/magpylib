
# -*- coding: utf-8 -*-
import numpy as np
from numpy import cos, sin, array, arccos, float64, pi
from magPyLib.math._mathLibPrivate import getPhi, fastNorm3D, angleAxisRotation

def rotatePosition(position,angle,axis,CoR=[0,0,0]):
    """
    This function uses angle-axis rotation to rotate the `position` vector by
    the `angle` argument about an axis defined by the `axis` vector which passes
    through the center of rotation `CoR` vector. Scalar input is either integer
    or float.Vector input format can be either list, tuple or array of any data
    type (float, int).
    
    Parameters:
    ----------
    position : vec3
        Input position to be rotated.
        
    angle : scalar [deg]
        Angle of rotation in untis of [deg]
    
    axis : vec3
        Axis of rotation

    CoR : vec3
        The Center of rotation which defines the position of the axis of rotation

    Returns:    
    --------
    newPosition : arr3
        Rotated position
        
    Example:
    --------
    >>> import magPyLib as magPy
    >>> from numpy import pi
    >>> position0 = [1,1,0]
    >>> angle = -90
    >>> axis = [0,0,1]
    >>> centerOfRotation = [1,0,0]
    >>> positionNew = magPy.math.rotatePosition(position0,angle,axis,CoR=centerOfRotation)
    >>> print(positionNew)
      [2. 0. 0.]
    """

    pos = array(position, dtype=float64, copy=False)
    ang = float(angle)
    ax = array(axis, dtype=float64, copy=False)
    cor = array(CoR, dtype=float64, copy=False)
    
    pos12 = pos-cor  
    pos12Rot = angleAxisRotation(ang,ax,pos12)
    posRot = pos12Rot+cor
    
    return posRot

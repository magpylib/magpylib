
# -*- coding: utf-8 -*-

import numpy as np
from numpy import cos, sin, array, arccos, float64, pi
from magPyLib.math._mathLibPrivate import getPhi, fastNorm3D, angleAxisRotation

def anglesFromAxis(axis):
    """
    This function takes an arbitrary `axis` (3-vector) and returns the orientation
    given by the `angles` = [phi,th] that are defined as in spherical coordinates. 
    phi = azimuth angle, th = polar angle. Vector input format can be either 
    list, tuple or array of any data type (float, int).
       
    Parameters:
    ----------
    axis : vec3
        Arbitrary input axis that defines an orientation.
        
    Returns:    
    --------
    angles : arr2 [deg]
        The angles [phi,th], azimuth and polar, that correspond to the orientation 
        given by the input axis.
        
    Example:
    --------
    >>> import magPyLib as magPy
    >>> axis = [1,1,0]
    >>> angles = magPy.math.anglesFromAxis(axis)
    >>> print(angles)
      [45. 90.]
    """
    ax = array(axis, dtype=float64, copy=False)
    
    Lax = fastNorm3D(ax)
    Uax = ax/Lax
    
    TH = arccos(Uax[2])/pi*180
    PHI = getPhi(Uax[0],Uax[1])/pi*180
    return array([PHI,TH])

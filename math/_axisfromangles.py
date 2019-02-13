# -*- coding: utf-8 -*-

import numpy as np
from numpy import cos, sin, array, arccos, float64, pi

def axisFromAngles(angles):
    """
    This function generates an `axis` (3-vector of length 1) from two `angles` = [phi,th]
    that are defined as in spherical coordinates. phi = azimuth angle, th = polar angle.
    Vector input format can be either list, tuple or array of any data type (float, int).
       
    Parameters:
    ----------
    angles : vec2 [deg]
        The two angels [phi,th], azimuth and polar, in units of deg.
        
    Returns:    
    --------
    axis : arr3
        An axis of length that is oriented as given by the input angles.
        
    Example:
    --------
    >>> import magPyLib as magPy
    >>> angles = [90,90]
    >>> ax = magPy.math.axisFromAngles(angles)
    >>> print(ax)
      [0.0  1.0  0.0]
    """
    phi, th = angles #phi in [0,2pi], th in [0,pi]
    phi = phi/180*pi
    th = th/180*pi
    return array([cos(phi)*sin(th),sin(phi)*sin(th),cos(th)])
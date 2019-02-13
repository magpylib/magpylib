
# -*- coding: utf-8 -*-
import numpy as np
from numpy import cos, sin, array, arccos, float64, pi
from magPyLib.math._mathLibPrivate import getPhi, fastNorm3D, angleAxisRotation

def randomAxis():
    """
    This function generates a random `axis` (3-vector of length 1) from an equal
    angular distribution using a MonteCarlo scheme.
       
    Parameters:
    ----------
    none
        
    Returns:    
    --------
    axis : arr3
        A random axis from an equal angular distribution of length 1
        
    Example:
    --------
    >>> import magPyLib as magPy
    >>> ax = magPy.math.randomAxis()
    >>> print(ax)
      [-0.24834468  0.96858637  0.01285925]
    
    """
    while True: 
        r = np.random.rand(3)*2-1       #create random axis
        Lr2 = sum(r**2)                 #get length
        if Lr2 <= 1:                    #is axis within sphere?
            Lr = np.sqrt(Lr2)           #normalize
            return r/Lr
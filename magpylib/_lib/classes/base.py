#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Define base classes here on which the magnetic source objects are built on

    1. RCS class: the underlying relative coordintate system initiates position
                    and orientation and provides object movement and rotation
                    functionalities.
    2. HomoMag class: initializes the homogeneous magnetization for all homogeneous
                    magnet classes
    3. LineCurrent class: initializes a current for all line current classes

'''

#%% IMPORTS
from numpy import array,float64,pi,isnan
from magpylib._lib.mathLibPrivate import Qmult, Qconj, getRotQuat, arccosSTABLE, fastSum3D, fastNorm3D
import sys

        
#%% FUNDAMENTAL CLASS - RCS (RELATIVE COORDINATE SYSTEM)
#       - initiates position, orientation
#       - adds moveBY, rotateBy

class RCS:

    def __init__(self, position, angle, axis):
        # fundamental (unit)-orientation/rotation is [0,0,0,1]
        
        self.position = array(position, dtype=float64, copy=False)
        self.angle = float(angle)
        self.axis = array(axis, dtype=float64, copy=False)
        
        #check input format
        if any(isnan(self.position))  or  len(self.position)!= 3:
            sys.exit('Bad pos input')
        if any(isnan(self.axis))  or  len(self.position)!= 3:
            sys.exit('Bad axis input')
   
        
    def setPosition(self,newPos):
        """
        This method moves the source to the position given by the argument 
        vector `newPos`. Vector input format can be either list, tuple or array
        of any data type (float, int)
        
        Parameters:
        ----------
        newPos : vec3 [mm]
            Set new position of the source.
            
        Returns:    
        --------
        None
            
        Example:
        --------
        >>> magpylib as magPy
        >>> pm = magPy.magnet.Sphere(mag=[0,0,1000],dim=1)
        >>> print(pm.position)
            [0. 0. 0.]
        >>> pm.setPosition([5,5,5])
        >>> print(pm.position)
            [5. 5. 5.]
        """
        self.position = array(newPos, dtype=float64, copy=False)
        if any(isnan(self.position))  or  len(self.position)!= 3:
            sys.exit('Bad pos input')


    def move(self,displacement):
        """
        This method moves the source by the argument vector `displacement`. 
        Vector input format can be either list, tuple or array of any data
        type (float, int).
        
        Parameters:
        ----------
        displacement : vec3 [mm]
            Set displacement vector
            
        Returns:    
        --------
        None
            
        Example:
        --------
        >>> magpylib as magPy
        >>> pm = magPy.magnet.Sphere(mag=[0,0,1000],dim=1,pos=[1,2,3])
        >>> print(pm.position)
            [1. 2. 3.]
        >>> pm.move([3,2,1])
        >>> print(pm.position)
            [4. 4. 4.]
        """
        mV = array(displacement, dtype=float64, copy=False)
        if any(isnan(mV))  or  len(mV)!= 3:
            sys.exit('Bad move vector input')
        self.position = self.position + mV
        
            
    def setOrientation(self,angle,axis):
        """
        This method sets a new source orientation given by `angle` and `axis`.
        Scalar input is either integer or float. Vector input format can be
        either list, tuple or array of any data type (float, int).
        
        Parameters:
        ----------
        angle  : scalar [deg]
            Set new angle of source orientation.
        
        axis : vec3 []
            Set new axis of source orientation.
            
        Returns:    
        --------
        None            
        
        Example:
        --------
        >>> magpylib as magPy
        >>> pm = magPy.magnet.Sphere(mag=[0,0,1000],dim=1)
        >>> print([pm.angle,pm.axis])
            [0.0, array([0., 0., 1.])]
        >>> pm.setOrientation(45,[0,1,0])
        >>> print([pm.angle,pm.axis])
            [45.0, array([0., 1., 0.])]
        """
        self.angle = float(angle)
        self.axis = array(axis, dtype=float64, copy=False)
        if any(isnan(self.axis))  or  len(self.position)!= 3:
            sys.exit('Bad axis input')
        
        
    def rotate(self, angle, axis, CoR='self.position'):
        """
        This method rotates the source about `axis` by `angle`. The axis passes
        through the center of rotation CoR. Scalar input is either integer or
        float. Vector input format can be either list, tuple or array of any
        data type (float, int).
        
        Parameters:
        ----------
        angle  : scalar [deg]
            Set angle of rotation in units of [deg]
        axis : vec3 []
            Set axis of rotation
        CoR : vec3 [mm]
            Specify the Center of rotation which defines the position of the
            axis of rotation. If not specified the source will rotate about its
            own center.
        
        Returns:    
        --------
        None
        
        Example:
        --------
        >>> magpylib as magPy
        >>> pm = magPy.magnet.Sphere(mag=[0,0,1000], dim=1)
        >>> print(pm.position, pm.angle, pm.axis)
          [0. 0. 0.] 0.0 [0. 0. 1.]
        >>> pm.rotate(90, [0,1,0], CoR=[1,0,0])
        >>> print([pm.position, pm.angle, pm.axis])
          [1., 0., 1.] 90.0 [0., 1., 0.]
        """
        #secure type
        ax = array(axis, dtype=float64, copy=False)
        ang = float(angle)
        if str(CoR) == 'self.position':
            cor = self.position
        else:
            cor = array(CoR, dtype=float64, copy=False)
        
        #check input
        if any(isnan(ax)) or len(ax)!= 3:
            sys.exit('Bad axis input')
        if fastSum3D(ax**2) == 0:
            sys.exit('Bad axis input')
        if type(ang) != float:
            sys.exit('Bad angle input')
        if any(isnan(cor)) or len(cor)!= 3:
            sys.exit('Bad CoR input')
        
        # determine Rotation Quaternion Q from self.axis-angle
        Q = getRotQuat(self.angle,self.axis)
        
        # determine rotation Quaternion P from rot input
        P = getRotQuat(ang,ax)
        
        # determine new orientation quaternion which follows from P.Q v (P.Q)*
        R = Qmult(P,Q)
        
        #reconstruct new axis-angle from new orientation quaternion
        ang3 = arccosSTABLE(R[0])*180/pi*2
        
        ax3 = R[1:] #konstanter mult faktor ist wurscht für ax3
        self.angle = ang3
        if ang3 == 0: #avoid returning a [0,0,0] axis
            self.axis = array([0,0,1])
        else:
            Lax3 = fastNorm3D(ax3)
            self.axis = array(ax3)/Lax3
        
        # set new position using P.v.P*
        posOld = self.position-cor
        Vold = [0] + [p for p in posOld]
        Vnew = Qmult(P,Qmult(Vold,Qconj(P)))
        self.position = array(Vnew[1:])+cor




#%% HOMOGENEOUS MAGNETIZATION CLASS
#       - initiates magnetization

class HomoMag(RCS):
    
    def __init__(self, position, angle, axis, magnetization):
        
        #inherit class RCS
        RCS.__init__(self,position,angle,axis)
        
        #secure input type and check input format of mag
        self.magnetization = array(magnetization, dtype=float64, copy=False)
        if any(isnan(self.magnetization))  or  len(self.magnetization)!= 3:
            sys.exit('Bad mag input')




#%% LINE CURRENT CLASS
#       - initiates current

class LineCurrent(RCS):
    
    def __init__(self, position, angle, axis, current):
        
        #inherit class RCS
        RCS.__init__(self,position,angle,axis)
        
        #secure input types and check input format
        self.current = float(current)
        if type(self.current) != float:
            sys.exit('Bad current input')
            
            
            
            
            
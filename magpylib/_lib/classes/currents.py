####### Type hint definitions ########
from typing import List, Tuple,TypeVar 
## Magnetization Vector Typehints
x_i=TypeVar('x_i',int,float) 
y_i=TypeVar('y_i',int,float)
z_i=TypeVar('z_i',int,float)
listOfPos = List[Tuple[x_i,y_i,z_i]]

I=0.0 ## Default Current
d=0.0 ## Default Diameter
######################################

#%% IMPORTS
from numpy import array,float64
import sys
from magpylib._lib.mathLibPrivate import angleAxisRotation
from magpylib._lib.classes.base import LineCurrent
from magpylib._lib.fields.Current_CircularLoop import Bfield_CircularCurrentLoop
from magpylib._lib.fields.Current_Line import Bfield_CurrentLine



#%% THE CIRCULAR CLASS
class Circular(LineCurrent):
    """ 
    A circular line current loop with diameter `dim` and a current `curr` flowing
    in positive orientation. In the canonical basis (position=[0,0,0], angle=0.0,
    axis=[0,0,1]) the loop lies in the x-y plane with the origin at its center.
    Scalar input is either integer or float. Vector input format can be
    either list, tuple or array of any data type (float, int).
    
    Parameters
    ----------
    
    curr : scalar [A]
        Set current in loop in units of [A]
        
    dim : float [mm]
        Set diameter of current loop in units of [mm]
        
    pos=[0,0,0] : vec3 [mm]
        Set position of the center of the current loop in units of [mm].
    
    angle=0.0 : scalar [deg]
        Set angle of orientation of current loop in units of [deg].
    
    axis=[0,0,1] : vec3 []
        Set axis of orientation of the current loop.
    
    Attributes
    ----------
    
    current : float [A]
        Current in loop in units of [A]
        
    dimension : float [mm]
        Loop diameter in units of [mm]
    
    position : arr3 [mm]
        Position of center of loop in units of [mm]
    
    angle : float [deg]
        Angle of orientation of the current loop.
        
    axis : arr3 []
        Axis of orientation of the current loop.
        
    Example
    -------
    >>> from magpylib import source
    >>> cd = source.current.Circular(curr=10,dim=2)
    >>> B = cd.getB([0,0,2])
    >>> print(B)
      [0.         0.         0.56198518]

    Note
    ----
    The following Methods are available to all sources objects.
    """  
      
    def __init__(self, curr=I, dim=d, pos=(0.0,0.0,0.0), angle=0.0, axis=(0.0,0.0,1.0)):
        
        #inherit class lineCurrent
        #   - pos, Mrot, MrotInv, curr
        #   - moveBy, rotateBy
        LineCurrent.__init__(self,pos,angle,axis,curr)
        
        #secure input type and check input format of dim
        assert dim >= 0, 'Bad input dimension'
        self.dimension = float(dim)
        
        
    def getB(self,pos):
        #secure input type and check input format
        p1 = array(pos, dtype=float64, copy=False)
        
        #relative position between mag and obs
        posRel = p1 - self.position
        
        #rotate this vector into the CS of the magnet (inverse rotation)
        p21newCm = angleAxisRotation(self.angle,-self.axis,posRel) # Leave this alone for now pylint: disable=invalid-unary-operand-type
        
        #the field is well known in the magnet coordinates
        BCm = Bfield_CircularCurrentLoop(self.current,self.dimension,p21newCm)  # obtain magnetic field in Cm
        
        #rotate field vector back
        B = angleAxisRotation(self.angle,self.axis,BCm)
        
        return B
        
    

#%% THE CIRCUAR CL CLASS

class Line(LineCurrent):
    """ 
    
    A line current flowing along linear segments from vertex to vertex given by
    a list of positions `vertices` in the canonical basis (position=[0,0,0], angle=0.0,
    axis=[0,0,1]). Scalar input is either integer or float. Vector input format
    can be either list, tuple or array of any data type (float, int).
    
    
    Parameters
    ----------
    
    curr : scalar [A]
        Set current in loop in units of [A]
        
    vertices : vecNx3 [mm]
        N positions given in units of [mm] that make up N-1 linear segments
        along which the current `curr` flows, starting from the first position
        and ending with the last one.
        [[x,y,z], [x,y,z], ...]
        "[pos1,pos2,...]"
    pos=[0,0,0] : vec3 [mm]
        Set reference position of the current distribution in units of [mm].
    
    angle=0.0 : scalar [deg]
        Set angle of orientation of current distribution in units of [deg].
    
    axis=[0,0,1] : vec3 []
        Set axis of orientation of the current distribution.
    
    Attributes
    ----------
    
    current : float [A]
        Current flowing along line in units of [A].
        
    vertices : arrNx3 [mm]
        Positions of line current vertices in units of [mm].
    
    position : arr3 [mm]
        Reference position of line current in units of [mm].
    
    angle : float [deg]
        Angle of orientation of line current in units of [deg].
        
    axis : arr3 []
        Axis of orientation of the line current.
        
    Examples
    --------
    >>> from magpylib import source
    >>> from numpy import sin,cos,pi,linspace
    >>> vertices = [[cos(phi),sin(phi),0] for phi in linspace(0,2*pi,36)]
    >>> cd = source.current.Line(curr=10,vertices=vertices)
    >>> B = cd.getB([0,0,2])
    >>> print(B)
      [-6.24500451e-17  1.73472348e-18  5.59871233e-01]

    
    Note
    ----
    The following Methods are available to all sources objects.
    """    
    def __init__(self, curr=I, vertices=listOfPos, pos=(0.0,0.0,0.0), angle=0.0, axis=(0.0,0.0,1.0)):
        
        #inherit class lineCurrent
        #   - pos, Mrot, MrotInv, curr
        #   - moveBy, rotateBy
        LineCurrent.__init__(self,pos,angle,axis,curr)
        
        #secure input type and check input format of dim
        assert type(vertices) != type(listOfPos), 'Line Current: enter a list of position vertices - Ex: Line(vertices=[(1,2,3),(3,2,1)])'
        assert all(type(pos)==tuple or type(pos)==list for pos in vertices), 'Line-current: Input position (3D) tuples or lists within the list - Ex: Line(vertices=[(1,2,3),(3,2,1)])'
        assert all(len(d)==3 for d in vertices), 'Line-current: Bad input dimension, tuple vectors in list must be 3D' 
        self.vertices = array(vertices, dtype=float64,copy=False)
        
        
                
        
        
    def getB(self,pos):
        #secure input type and check input format
        p1 = array(pos, dtype=float64, copy=False)
        
        #relative position between mag and obs
        posRel = p1 - self.position
        
        #rotate this vector into the CS of the magnet (inverse rotation)
        p21newCm = angleAxisRotation(self.angle,-self.axis,posRel) # Leave this alone for now pylint: disable=invalid-unary-operand-type
        
        #the field is well known in the magnet coordinates
        BCm = Bfield_CurrentLine(p21newCm,self.vertices,self.current)  # obtain magnetic field in Cm
        
        #rotate field vector back
        B = angleAxisRotation(self.angle,self.axis,BCm)
        
        return B
    
    

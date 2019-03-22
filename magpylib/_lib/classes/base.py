'''
Base Functions
==============
Define base classes here on which the magnetic source objects are built on

    1. RCS class: the underlying relative coordintate system initiates position
                    and orientation and provides object movement and rotation
                    functionalities.
    2. HomoMag class: initializes the homogeneous magnetization for all homogeneous
                    magnet classes
    3. LineCurrent class: initializes a current for all line current classes

'''

######### Type hint definitions ########
# These aren't type hints, but look good 
# in Spyder IDE. Pycharm recognizes it.
from typing import Tuple
Auto = 0 # Maximum cores, for multicore
        # function. if 0 find max.
numpyArray = 0
constant = None
#######################################

#%% IMPORTS
from itertools import product, repeat
from numpy import array,float64,pi,isnan,array,ndarray
from magpylib._lib.mathLibPrivate import Qmult, Qconj, getRotQuat, arccosSTABLE, fastSum3D, fastNorm3D
from magpylib._lib.utility import checkDimensions, initializeMulticorePool, recoordinateAndGetB, equalizeListOfPos
from magpylib._lib.utility import posVectorFinder
from multiprocessing import Pool,cpu_count
import sys

        
#%% FUNDAMENTAL CLASS - RCS (RELATIVE COORDINATE SYSTEM)
#       - initiates position, orientation
#       - adds moveBY, rotateBy

class RCS:
    """
    FUNDAMENTAL CLASS - RCS (RELATIVE COORDINATE SYSTEM)

    initiates position, orientation
    - adds moveBY, rotateBy
    """
   
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
        
        Parameters
        ----------
        newPos : vec3 [mm]
            Set new position of the source.
            
        Returns
        -------
        None
            
        Example
        -------
        >>> from magpylib import source
        >>> pm = source.magnet.Sphere(mag=[0,0,1000],dim=1)
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
        
        Parameters
        ----------
        displacement : vec3 [mm]
            Set displacement vector
            
        Returns
        -------
        None
            
        Example
        -------
        >>> from magpylib import source
        >>> pm = source.magnet.Sphere(mag=[0,0,1000],dim=1,pos=[1,2,3])
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
        
        Parameters
        ----------
        angle  : scalar [deg]
            Set new angle of source orientation.
        
        axis : vec3 []
            Set new axis of source orientation.
            
        Returns
        -------
        None            
        
        Example
        -------
        >>> from magpylib import source
        >>> pm = source.magnet.Sphere(mag=[0,0,1000],dim=1)
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
        
        
    def rotate(self, angle, axis, anchor='self.position'):
        """
        This method rotates the source about `axis` by `angle`. The axis passes
        through the center of rotation anchor. Scalar input is either integer or
        float. Vector input format can be either list, tuple or array of any
        data type (float, int).
        
        Parameters
        ----------
        angle  : scalar [deg]
            Set angle of rotation in units of [deg]
        axis : vec3 []
            Set axis of rotation
        anchor : vec3 [mm]
            Specify the Center of rotation which defines the position of the
            axis of rotation. If not specified the source will rotate about its
            own center.
        
        Returns
        -------
        None
        
        Example
        -------
        >>> from magpylib import source
        >>> pm = source.magnet.Sphere(mag=[0,0,1000], dim=1)
        >>> print(pm.position, pm.angle, pm.axis)
          [0. 0. 0.] 0.0 [0. 0. 1.]
        >>> pm.rotate(90, [0,1,0], anchor=[1,0,0])
        >>> print(pm.position, pm.angle, pm.axis)
          [1., 0., 1.] 90.0 [0., 1., 0.]
        """
        #secure type
        ax = array(axis, dtype=float64, copy=False)
        ang = float(angle)
        if str(anchor) == 'self.position':
            anchor = self.position
        else:
            anchor = array(anchor, dtype=float64, copy=False)
        
        #check input
        if any(isnan(ax)) or len(ax)!= 3:
            sys.exit('Bad axis input')
        if fastSum3D(ax**2) == 0:
            sys.exit('Bad axis input')
        if type(ang) != float:
            sys.exit('Bad angle input')
        if any(isnan(anchor)) or len(anchor)!= 3:
            sys.exit('Bad anchor input')
        
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
        posOld = self.position-anchor
        Vold = [0] + [p for p in posOld]
        Vnew = Qmult(P,Qmult(Vold,Qconj(P)))
        self.position = array(Vnew[1:])+anchor

    def getBMulticore(self,pos=numpyArray,processes=Auto):
        """
        Take a numpy array/matrix of positions, set up with any dimension,
        then calculate the Bfield in parallel using a process in each core
        and return a matrix of the same format.
        
        Parameters
        ----------
        pos : [numpyArray]
            Numpy array (the default is numpyArray, which [default_description])
        processes : [type], optional
            Number of worker processes to multicore. (the default is Auto, 
            which is all visible cores minus 1)
        
        Returns
        -------
        [NumpyArray]
            A matrix array with the shame shape as pos, 
            containing instead values of B fields for each input coordinate position.
        
        Example
        -------
        >>> from magpylib import source
        >>> from numpy import array
        >>> pm = source.magnet.Box(mag=[6,7,8],dim=[10,10,10],pos=[2,2,2])
        >>> ## Positions list
        >>> P1=(.5,.5,5)
        >>> P2=[30,20,10]
        >>> P3=[1,.2,60]

        >>> arrayOfPos = array( [   [P1,P2,P3],])
        >>> result = pm.getBMulticore(arrayOfPos)
            [[[ 3.99074612e+00  4.67238469e+00  4.22419432e+00]
            [ 3.90057773e-02  1.88083191e-02 -1.34111687e-03]
            [-2.60347051e-03 -3.13961826e-03  6.10885894e-03]]]
        """

        results = []
        positionsList = []
        posVectorFinder(pos,positionsList) # Put all position vectors in a list reference
        pool = initializeMulticorePool(processes)


        results = pool.map(self.getB, positionsList) # Map the concurrent function pointer to a list of 
                                                    # arguments to run as parameters for each parallelized instance.
        pool.close()
        pool.join()
        
        results = array(results).reshape(pos.shape)
        return results
    
    def getBDisplacement(self,Bpos,listOfPos=constant,listOfRotations=constant,processes=Auto): 
        """
        In a parallelized environment, calculates the B field in a position for every pair of source position
        and absolute rotation within the lists.

        Will make a copy of the original object, so make sure rotations are absolute.

        
        Parameters
        ----------
        listOfPos : List [vec3]
            Repositions of the target magnet. Needs to be the same size as rotations list.
        listOfRotations : List [angle, axisVec,[anchor]]
            Angle and axis vector for the rotation after reposition. Needs to be the same size as positions list.
        Bpos : [vec3]
            Position Vector for calculating the magnetic field.
        processes : [type], optional
            Number of workers for parallel processing (the default is Auto, which calculates with all cores minus 1)
        
        Returns
        -------
        Array of position vectors for B field for each pairing of Position/Rotation.

        Example
        -------
        >>> # Input
        >>> mag=[1,2,3]
        >>> dim=[1,2,3]
        >>> pos=[0,0,0]
        >>> listOfDisplacement=[[0,0,1],
        >>>                    [0,1,0],
        >>>                    [1,0,0]]
        >>> #(angle,axisVector,anchorPos) // anchor is optional
        >>> listOfRotations = [ (180,(0,1,0)),
        >>>                    (90,(1,0,0)),
        >>>                    (255,(0,1,0))]
        >>> Bpos = [1,2,3]
        >>> # Run
        >>> from magpylib.source import magnet
        >>> pm = magnet.Box(mag,dim,pos)
        >>> result = pm.getBDisplacement(Bpos,
        >>>                               listOfPos=listOfDisplacement,
        >>>                               listOfRotations=listOfRotations)
            [   array([ 0.00453617, -0.07055326,  0.03153698]), 
                array([0.00488989, 0.04731373, 0.02416068]), 
                array([0.0249435 , 0.00106315, 0.02894469]) ]
        """
        results = []
        ## Assert lists are of equal size before proceeding. Equalize when possible.
        posVectors, rotArguments = equalizeListOfPos(   listOfPos,
                                                        listOfRotations,
                                                        neutralPos=self.position )
        ## Start pooling arguments for a reposition+rotate -> getB helper function.
        ## Feed positions and rotations from each list in pairs.
        ## Same getB position for all instances.
        ## Since this is parallelized, Positions and Rotations need to be absolute 
        ## against the initial coordinates of the object. 
        pool = initializeMulticorePool(processes)
        results = pool.starmap(recoordinateAndGetB, 
                                                    zip(repeat(self,times=len(posVectors)),
                                                        posVectors,
                                                        rotArguments,
                                                        repeat(Bpos,times=len(posVectors))))
        ## Close the pooled processes and wrap up before returning.
        pool.close()
        pool.join()
        return results

    def getB(self,pos): 
        """
        This method returns the magnetic field vector generated by the source 
        at the argument position `pos` in units of [mT]
        
        Parameters
        ----------
        pos : vec3 [mm]
            Position or list of Positions where magnetic field should be determined.
        

        Returns
        -------
        magnetic field vector : arr3 [mT]
            Magnetic field at the argument position `pos` generated by the
            source in units of [mT].
        """
        #Return a list of vec3 results   
        # This method will be overriden by the classes that inherit it.
        # Throw a warning and return 0s if it somehow isn't.
        ## Note: Collection() has its own docstring 
        ## for getB since it inherits nothing.
        import warnings
        warnings.warn("called getB method is not implemented in this class, returning [0,0,0]", RuntimeWarning)
        return [0,0,0]



#%% HOMOGENEOUS MAGNETIZATION CLASS
#       - initiates magnetization

class HomoMag(RCS):
    
    def __init__(self, position, angle, axis, magnetization):
        
        #inherit class RCS
        RCS.__init__(self,position,angle,axis)
        assert all(a == 0 for a in magnetization) is False, "Bad mag input, all values are zero"

        #secure input type and check input format of mag
        self.magnetization = array(magnetization, dtype=float64, copy=False)
        assert (not any(isnan(self.magnetization))  and  len(self.magnetization)== 3), "Bad mag input, invalid vector dimension"
    



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
            
            
            
            
            
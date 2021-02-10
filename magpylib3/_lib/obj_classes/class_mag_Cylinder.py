"""Magnet Cylinder class code"""

import numpy as np
from magpylib3._lib.obj_classes.class_BaseGeo import BaseGeo
from magpylib3._lib.obj_classes.class_Collection import Collection
from magpylib3._lib.fields.field_BHwrapper import getB, getH
    
class Cylinder(BaseGeo):
    """ Homogeneous Cylinder magnet.

    init_state: the geometric center is in the CS origin, the 
        cylinder axis coincides with the z-direction
    
    ### Properties
    - mag (vec3): Magnetization vector (remanence field) of magnet
        in units of mT.
    - dim (vec3): Dimension/Size of the Cylinder with diameter and
        height (d,h) in units of mm.
    - pos (vec3): Position of the geometric center of the magnet 
        in units of mm. Defaults to (0,0,0).
    - rot (rotation input): Relative rotation of magnet to init_state.
        Input can either be a pair (angle, axis) with angle a scalar 
        given in deg and axis an arbitrary 3-vector, or a 
        scipy..Rotation object. Defaults to (0,0,0) rotation vector.

    ### Methods
    - move: move magnet by argument vector
    - rotate: rotate magnet
    - display: graphically display magnet
    - getB: compute B-field of magnet
    - getH: compute H-field of magnet

    ### Returns:
    - (Cylinder source object)

    ### Info
    Addition of sources returns a Collection.

    Computation of the Cylinder field has a iter=50 kwarg to set the
        iteration in the computation of the diametral component.
    """
    
    def __init__(self,
        mag = '(mx,my,mz)', 
        dim = '(d,h)', 
        pos = (0,0,0),
        rot = None
        ):

        # inherit base_geo class
        BaseGeo.__init__(self, pos, rot)

        # set mag and dim attributes
        self.mag = mag
        self.dim = dim
    
    # properties ----------------------------------------------------
    @property
    def mag(self):
        """ magnet magnetization in mT
        """
        return self._mag

    @mag.setter
    def mag(self, value):
        """ set magnetization vector, vec3, mT
        """
        self._mag = np.array(value,dtype=np.float64)
    

    @property
    def dim(self):
        """ cylinder dimension (d,h) in mm
        """
        return self._dim

    @dim.setter
    def dim(self, value):
        """ set box dimension (a,b,c), vec3, mm
        """
        self._dim = np.array(value,dtype=np.float64)


    # dunders -------------------------------------------------------
    def __add__(self, sources):
        """ sources add up to a Collection object
        """
        return Collection(self,sources)


    # methods -------------------------------------------------------
    def getB(self, pos_obs, niter=50):
        """ Compute B-field of magnet at observer positions.

        ### Args:
        - pos_obs (N1 x N2 x ... x 3 vec): single position or set of 
            observer positions in units of mm.
        - niter (int): Number of iterations in the computation of the
            diametral component of the field

        ### Returns:
        - (N1 x N2 x ... x 3 ndarray): B-field at observer positions
            in units of mT.
        """
        B = getB(self, pos_obs, niter=niter)
        return B
    

    def getH(self, pos_obs, niter=50):
        """ Compute H-field of magnet at observer positions.

        ### Args:
        - pos_obs (N1 x N2 x ... x 3 vec): single position or set of 
            observer positions in units of mm.

        ### Returns:
        - (N1 x N2 x ... x 3 ndarray): H-field at observer positions
            in units of kA/m.
        """
        H = getH(self, pos_obs, niter=niter)
        return H
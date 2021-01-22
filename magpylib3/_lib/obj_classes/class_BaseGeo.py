"""BaseGeo class code"""

import numpy as np
from magpylib3._lib.math.utility import rotobj_from_rot_input
from magpylib3._lib.graphics import display


class BaseGeo:
    """ initializes basic geometric properties 'position' and 'orientation'
    
    position is simple arr3 in Euclidean space

    orientation/rotation is a scipy.spatial.transformation.Rotation 
    object that gives the relative rotation to the init_state. The 
    init_state is defined by how the fields are implemented (e.g. 
    cyl upright in xy-plane)

    ### Properties
    - pos
    - rot

    ### Methods
    - display
    - move
    - rotate

    ### Returns:
    - BaseGeo object
    """

    def __init__(self, pos, rot):
        # set pos and orient attributes
        self.pos = pos
        self.rot = rot


    # properties ----------------------------------------------------
    @property
    def pos(self):
        """ return object position, (arr3)
        """
        return self._pos

    @pos.setter
    def pos(self, value):
        """ set object position, input vec3
        """
        self._pos = np.array(value, dtype=np.float64)


    @property
    def rot(self):
        """ returns rotation relative to init_state (rot object)
        """
        return self._rot

    @rot.setter
    def rot(self, value):
        """ set relative rotation
        """
        self._rot = rotobj_from_rot_input(value)


    # methods -------------------------------------------------------
    def display(self,**kwargs):
        """ 
        display object graphically. kwargs of top level display() function.
        """
        display(self,**kwargs)


    def move(self, displacement):
        """ 
        Translate object by the argument vector.

        ### Args:
        - displacement (vec3): displacement vector in units of mm.

        ### Returns:
        - self
        """
        
        displ = np.array(displacement, dtype=np.float)
        self._pos = self._pos + displ

        return self # for chaining


    def rotate(self, rot, anchor=None):
        """ 
        Rotate object such that the axis of rotation passes through
        the anchor.

        ### Args:
        - rot (rotation input): Can either be a pair (angle, axis) with 
            angle a scalar given in [deg] and axis an arbitrary 3-vector, 
            or a scipy..Rotation object.
        - anchor (vec3): The axis of rotation passes through the anchor point. 
            By default (anchor=None) the object will rotate about its own center.

        ### Returns:
        - self
        """
        
        # rotation object from input
        rot = rotobj_from_rot_input(rot)
        
        # adjust relative object rotation
        self._rot = rot*self._rot

        # set new position when rotating about an anchor
        if anchor is not None:
            anch = np.array(anchor, dtype=np.float64) # secure type
            pos_old = self._pos-anch                  # relative pos to anchor
            pos_new = rot.apply(pos_old)              # rotate about anchor
            self._pos = pos_new + anch                # set new pos

        return self # for chaining
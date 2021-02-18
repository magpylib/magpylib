"""BaseGeo class code"""

import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib3._lib.math_utility.utility import rotobj_from_angax
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
        """ object position-path
        
        Returns
        -------
        object position-path: np.array, shape (3,) or (N,3)
        """
        if len(self._pos) == 1:     # single path position - reduce dimension
            return self._pos[0]
        else:                       # return full path
            return self._pos

    @pos.setter
    def pos(self, input):
        """ set object position-path
        
        input: array_like, shape (3,) or (N,3)
            set position-path of object
        """
        input = np.array(input, dtype=np.float)     # secure input 
        if input.ndim == 1:                         # single position - increase dimension to (1,3)
            self._pos = np.tile(input,(1,1))
        elif input.ndim == 2:                       # multi position
            self._pos = input
        else:
            print('ERROR setting src.pos: bad input shape')
            sys.exit()

    @property
    def rot(self):
        """ object rotation-path relative to init_state

        Returns
        -------
        object rotation-path: scipy Rotation object, shape (1,) or (N,)
            set rotation-path of object
        """
        if len(self._rot)==1:           # single path rotation - reduce dimension
            return self._rot[0]
        else:                           # return full path
            return self._rot

    @rot.setter
    def rot(self, input: R):
        """ set object rotation-path

        input: scipy Rotation object, shape (1,) or (N,), default=None
            set rotation-path of object. None generates a unit rotation
            for every path step.
        """
        if input is None:                                           # None input generates unit rotation
            path_length = len(self._pos)
            self._rot = R.from_quat([(0,0,0,1)]*path_length, normalized=True)
        else:                                                       # single rotation - increase dimension to (1,3)
            val = input.as_quat()
            if val.ndim == 1:
                self._rot = R.from_quat([val], normalized=True)
            else:                                                   # multi rotation
                self._rot = input


    # methods -------------------------------------------------------
    def display(self,**kwargs):
        """ display object graphically. kwargs of top level display() function.
        
        Parameters
        ----------
        markers: array_like, shape (N,3), default=[(0,0,0)]
            Mark positions in graphic output. Puts a marker in the origin.
            by default.
        
        subplotAx: pyplot.axis, default=None
            Display graphical output in a given pyplot axis (must be 3D). 
        
        direc: bool, default=False
            Set True to plot magnetization and current directions
        
        show_path: bool/string, default=False
            Set True to plot object paths. Set 'all' to plot an object 
            represenation at each path position.
        
        Returns
        -------
        None
        """
        display(self,**kwargs)


    def move_by(self, displacement, steps=0):
        """ Linear displacement of object

        Parameters
        ----------
        displacement: array_like, shape (3,)
            displacement vector in units of mm.
        
        steps: int, optional, default=0
            If steps=0: path[-1] will be displaced
            If steps>0: add to existing path, linear steps to displaced
                position starting from path[-1]
            If steps<0: superpose existing path with linear motion from 
                path[steps-1] to displaced position.

        Returns:
        --------
        self : object with position and orientation properties.

        """
        displ = np.array(displacement, dtype=float)     # secure input
        path_pos = self._pos                            # load position path
        pos_tgt = path_pos[-1] + displ                  # target position

        # steps=0: last path pos becomes target position, no rot
        if steps == 0:             
            self._pos[-1] = pos_tgt
            return self
        # determine start position
        if steps >= 1:             # steps >0: add new positions to path
            pos_start = path_pos[-1]
        else:                      # steps <0: apply operation to last steps 
            if len(path_pos)<=abs(steps):
                print('ERROR: .move_by() bad input, |-steps| must be smaller than path_length')
                sys.exit()
            pos_start = (0,0,0)
            pos_tgt = displ

        # generate additional pos path
        x0, y0, z0 = pos_start
        x1, y1, z1 = pos_tgt
        xs = np.linspace(x0, x1, abs(steps)+1)
        ys = np.linspace(y0, y1, abs(steps)+1)
        zs = np.linspace(z0, z1, abs(steps)+1)
        addpath_pos = np.c_[xs,ys,zs][1:]

        # apply to existing path
        if steps >=1 :
            # load rotation path and tile last entry
            path_rot = self._rot.as_quat()
            addpath_rot = np.tile(path_rot[-1],(steps,1))
            # set new path
            self.pos = np.r_[path_pos, addpath_pos]
            self.rot = R(np.r_[path_rot, addpath_rot], normalized=True)
        else:
            # apply operation on top of state
            self._pos[steps:] = self._pos[steps:] + addpath_pos
        
        return self


    def move_to(self, pos_target, steps=0):
        """ Object translation to target position.

        Parameters
        ----------
        pos_target: array_like, shape (3,)
            target position vector in units of mm.
        
        steps: int, optional, default=0
            If steps=0: path[-1] will be set to target position
            If steps>0: add to path, linear steps to target position 
                starting from path[-1].
            If steps<0: superpose existing path with linear motion from 
                path[steps-1] to target position.

        Returns:
        --------
        self : object with position and orientation properties.

        """
        # secure input
        pos_tgt = np.array(pos_target, dtype=float)

        # determine input for .move_to() and hand over
        if steps>=0:
            self.move_by(pos_tgt - self._pos[-1], steps)
        else:
            if len(self._pos)<=abs(steps):
                print('ERROR: .move_to() bad input, |-steps| must be smaller than path_length')
                sys.exit()
            self.move_by(pos_tgt - self._pos[steps-1], steps)

        return self


    def rotate(self, rot:R, anchor=None, steps=0):    
        """ Object rotation

        Parameters
        ----------
        rot: scipy Rotation object
        
        anchor: None or array_like, shape (3,), default=None
            The axis of rotation passes through the anchor point. For anchor=None
            the object will rotate about its own center.

        steps: int, optional, default=0
            If steps=0: rot is applied to path[-1]
            If steps>0: linear rotation steps from 0 to rot starting with 0 at
                path[-1] are added to the existing path.
            If steps<0: apply linear rotation steps from 0 to rot to existing
                path starting with 0 at path[steps-1].

        Returns:
        --------
        self : object with position and orientation properties.

        """
        # load current path
        path_pos = self._pos
        path_rot = self._rot.as_quat()

        # generate rotations
        stepss = np.tile(np.linspace(0,1,abs(steps)+1),(3,1)).T
        if steps==0:
            stepss += 1
        rots = R.from_rotvec(rot.as_rotvec()*stepss)
        
        if steps >=0:
            rot_new = rots*self._rot[-1]                  # apply rot to path[-1]
            self.rot = R(np.r_[path_rot[:-1], rot_new.as_quat()], normalized=True)
            if anchor is not None:
                anch = np.array(anchor, dtype=float)      # secure type
                pos_old = path_pos[-1] - anch             # relative pos to anchor
                pos_new = rots.apply(pos_old) + anch      # rotate about anchor
            else:
                pos_new = np.tile(path_pos[-1],(steps,1)) # tile positions
            self.pos = np.r_[self._pos[:-1], pos_new]
        
        else:
            rot_new = rots*self.rot[steps-1:]              # apply to path[steps:]
            self.rot = R(np.r_[path_rot[:steps-1], rot_new.as_quat()], normalized=True)
            if anchor is not None:
                anch = np.array(anchor, dtype=float)      # secure type
                pos_old = path_pos[steps-1:] - anch       # relative pos to anchor
                pos_new = rots.apply(pos_old) + anch      # rotate about anchor
                self._pos[steps-1:] = pos_new

        return self


    def rotate_from_angax(self, angle, axis, anchor=None, steps=0, degree=True):
        """ Object rotation from angle-axis combination

        Parameters
        ----------
        angle: float
            Angle of rotation (in [deg] by default).
        
        axis: array_like, shape (3,)
            The axis of rotation [dimensionless]

        anchor: None or array_like, shape (3,), default=None
            The axis of rotation passes through the anchor point. By default 
            anchor=None the object will rotate about its own center.
        
        degree: bool, default=True
            If True, Angle is given in [deg]. If False, angle is given in [rad].
                    
        steps: int, optional, default=0
            If steps=0: rot is applied to path[-1]
            If steps>0: linear rotation steps from 0 to rot starting with 0 at
                path[-1] are added to the existing path.
            If steps<0: apply linear rotation steps from 0 to rot to existing
                path starting with 0 at path[steps-1].

        Returns:
        --------
        self : object with position and orientation properties.
        """
        
        # degree to rad
        if degree:
            angle = angle/180*np.pi

        # secure input type
        axis = np.array(axis, dtype=np.float64)

        # rotations beyond pi
        n_rot = int(abs(angle)/np.pi)    # number of pi-rotations
        
        if n_rot>0:
            sign_angle = np.sign(angle)
            rot_pi = rotobj_from_angax(sign_angle*np.pi, axis) # rotate n_rot times
            rot_rest = rotobj_from_angax(angle%np.pi, axis)    # rest rotation
            
            # apply rest rotation
            self.rotate(rot_rest, anchor, steps)
            # on top of rest rotation apply pi-rotations
            for _ in range(n_rot):
                self.rotate(rot_pi, anchor, -abs(steps))
        
        return self
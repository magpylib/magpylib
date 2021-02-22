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
            return np.array(self._pos[0])
        else:                       # return full path
            return np.array(self._pos)

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
    def display(self,
        markers=[(0,0,0)], 
        subplotAx=None,
        direc=False,
        show_path=False):
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
        display(self, 
            markers=markers,
            subplotAx=subplotAx, 
            direc=direc, 
            show_path=show_path)


    def move_by(self, displacement, steps=-1):
        """ Linear displacement of object

        Parameters
        ----------
        displacement: array_like, shape (3,)
            displacement vector in units of mm.
        
        steps: int, optional, default=-1
            If steps < 0: apply a linear motion from 0 to displ on top 
                of existing path[steps:]. Specifically, steps=-1 will just
                displace path[-1].
            If steps > 0: add linear displacement to existing path starting
                at path[-1].

        Returns:
        --------
        self : object with position and orientation properties.

        """
        # secure input
        displ = np.array(displacement, dtype=float)
        
        # load current path
        path_pos = self._pos
        path_len = len(path_pos)

        # bad steps input, set to max size
        if steps < -path_len:
            steps = -path_len # path[0] sees 0 displ
            print('WARNING: .move_by(), steps<path_len, setting to -len(path)')

        # generate additional pos path
        ts = np.linspace(0, 1, abs(steps)+1)[1:]
        addpath_pos = displ * np.tile(ts,(3,1)).T

        # apply to existing path
        if steps > 0:
            # load rotation path and tile last entry
            path_rot = self._rot.as_quat()
            addpath_rot = np.tile(path_rot[-1],(steps,1))
            # set new path
            self.pos = np.r_[path_pos, addpath_pos + path_pos[-1]]
            self.rot = R(np.r_[path_rot, addpath_rot], normalized=True)
        else:
            # apply operation on top of path[steps:]
            self._pos[steps:] = path_pos[steps:] + addpath_pos
        
        return self


    def rotate(self, rot:R, anchor=None, steps=-1):    
        """ Object rotation

        Parameters
        ----------
        rot: scipy Rotation object
        
        anchor: None or array_like, shape (3,), default=None
            The axis of rotation passes through the anchor point. When anchor=None
            the object will rotate about its own center.

        steps: int, optional, default=-1
            If steps < 0: apply linear rotation steps from 0 to rot on top 
                of existing path[steps:]. Specifically, steps=-1 will just
                rotate path[-1].
            If steps > 0: add linear rotation steps from 0 to rot to existing
                path starting at path[-1].

        Returns:
        --------
        self : object with position and orientation properties.

        """
        # secure input type
        if anchor is not None:
            anchor = np.array(anchor, dtype=float)      # if None

        # load current path
        path_pos = self._pos
        path_rot = self._rot.as_quat()
        path_len = len(path_rot)
        
        # bad steps input, set to max size
        if steps < -path_len:
            steps = -path_len + 1  # path[0] sees 0 rot
            print('WARNING: .rotate(), steps<path_len, setting to max-1')

        # generate rotations
        stepping = np.linspace(0,1,abs(steps)+1)[1:]
        rots = R.from_rotvec([rot.as_rotvec()*s for s in stepping])
        
        if steps > 0:
            # apply rot to path[-1] and add resulting vector to path
            rot_new = rots*self._rot[-1]                  
            self.rot = R(np.r_[path_rot, rot_new.as_quat()], normalized=True)
            # compute positions and add to path
            if anchor is not None:
                pos_new = rots.apply(path_pos[-1]-anchor) + anchor
            else:
                pos_new = np.tile(path_pos[-1],(steps,1))
            self.pos = np.r_[self._pos, pos_new]
        
        else:
            # apply rotation to path[steps:] and apply result to path[steps:]
            rot_new = rots*self.rot[steps:]
            self.rot = R(np.r_[path_rot[:steps], rot_new.as_quat()], normalized=True)
            if anchor is not None:
                pos_new = rots.apply(path_pos[steps:]-anchor) + anchor      # rotate about anchor
                self._pos[steps:] = pos_new

        return self


    def rotate_from_angax(self, angle, axis, anchor=None, steps=-1, degree=True):
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
                    
        steps: int, optional, default=-1
            If steps < 0: apply linear rotation steps from 0 to rot on top 
                of existing path[steps:]. Specifically, steps=-1 will just
                rotate path[-1].
            If steps > 0: add linear rotation steps from 0 to rot to existing
                path starting at path[-1].

        Returns:
        --------
        self : object with position and orientation properties.
        
        """
        # degree to rad
        if degree:
            angle = angle/180*np.pi

        # secure input type
        if isinstance(axis,str):
            if axis=='x':
                axis=np.array((1,0,0))
            elif axis=='y':
                axis=np.array((0,1,0))
            elif axis=='z':
                axis=np.array((0,0,1))
            else:
                print('ERROR: .rotate_from_angax(), bad axis input')
                sys.exit()
        else:
            axis = np.array(axis, dtype=np.float64)

        # Split up rotation into pi-rotation and rest-rotation as
        #   the scipy.Rotation module is limited to express rotations
        #   only within the interval [-pi,pi]
        # pi-rotation includes all multiples of pi
        # rest-rotation includes the rest

        # apply rest-rotation (within [-pi,pi])
        ang_sign = np.sign(angle)
        ang_rest = abs(angle) % np.pi
        rot_rest = rotobj_from_angax(ang_sign*ang_rest, axis)
        self.rotate(rot_rest, anchor, steps)
        
        # apply rotations beyond pi (on top of rest-rotation)
        n_rot = int(abs(angle)/np.pi)    # number of pi-rotations
        if n_rot>0:
            rot_pi = rotobj_from_angax(ang_sign*np.pi, axis) # rotate n_rot times
            for _ in range(n_rot):
                self.rotate(rot_pi, anchor, -abs(steps))
        
        return self
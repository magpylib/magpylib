"""BaseGeo class code"""

import sys
from contextlib import contextmanager
import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib3._lib.math_utility import rotobj_from_angax
from magpylib3._lib.display import display
from magpylib3._lib.obj_classes.class_Collection import Collection


class BaseGeo:
    """ Initializes position and rotation (=orientation) properties
    of an object in a global CS.

    Position is a ndarray with shape (3,).

    Rotation is a scipy.spatial.transformation.Rotation
    object that gives the relative rotation to the init_state. The
    init_state is defined by how the fields are implemented (e.g.
    cyl upright in xy-plane)

    Both attributes _pos and _rot.as_rotvec() are of shape (N,3),
    and describe a path of length N. (N=1 if there is only one
    object position).

    Properties
    ----------
    pos: array_like, shape (N,3)
        Position path

    rot: scipy.Rotation, shape (N,)
        Rotation path

    Methods
    -------

    - display
    - move_by
    - move_to
    - rotate
    - rotate_from_angax

    """

    def __init__(self, pos, rot):
        # set pos and orient attributes
        self.pos = pos
        self.rot = rot
        self._mm = False        # motion_merge
        self._mm_steps = None   # motion_merge steps
        self._mm_first = True   # motion_merge first operation

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
        return self._pos            # return full path


    @pos.setter
    def pos(self, inp):
        """ set object position-path

        inp: array_like, shape (3,) or (N,3)
            set position-path of object
        """

        inp = np.array(inp, dtype=np.float)       # secure input
        if inp.ndim == 1:                         # single position - increase dimension to (1,3)
            self._pos = np.array([inp])
        elif inp.ndim == 2:                       # multi position
            self._pos = inp
        else:
            sys.exit('ERROR: .pos() - a path must be of shape (3,) or (N,3)')


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
        return self._rot                # return full path


    @rot.setter
    def rot(self, inp: R):
        """ set object rotation-path

        inp: scipy Rotation object, shape (1,) or (N,), default=None
            set rotation-path of object. None generates a unit rotation
            for every path step.
        """

        if inp is None:                            # None inp generates unit rotation
            path_length = len(self._pos)
            self._rot = R.from_quat([(0,0,0,1)]*path_length, normalized=True)
        else:                                      # single rotation - increase dimension to (1,3)
            val = inp.as_quat()
            if val.ndim == 1:
                self._rot = R.from_quat([val], normalized=True)
            else:                                  # multi rotation
                self._rot = inp


    # methods -------------------------------------------------------
    def display(
            self,
            markers=[(0,0,0)],
            axis=None,
            direc=False,
            show_path=False):
        """
        Display object graphically. kwargs of top level display() function.

        Parameters
        ----------
        markers: array_like, shape (N,3), default=[(0,0,0)]
            Mark positions in graphic output. Puts a marker in the origin.
            by default.

        axis: pyplot.axis, default=None
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
        #pylint: disable=dangerous-default-value

        display(
            self,
            markers=markers,
            axis=axis,
            direc=direc,
            show_path=show_path)


    def move_by(self, displacement, steps=None):
        """ Linear displacement of object

        Parameters
        ----------
        displacement: array_like, shape (3,)
            displacement vector in units of mm.

        steps: int or None, default=None
            If steps=None: Object will simply be moved without generating a
                path. Specifically, path[-1] of object is set anew. This is
                similar to having steps=-1.
            If steps < 0: apply a linear motion from 0 to displ on top
                of existing path[steps:]. Specifically, steps=-1 will just
                displace path[-1].
            If steps > 0: add linear displacement to existing path starting
                at path[-1].

        Returns:
        --------
        self : object with position and orientation properties.
        """

        # steps
        steps = get_steps(steps, self)

        # secure input
        displ = np.array(displacement, dtype=float)

        # load current path
        path_pos = self._pos

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


    def move_to(self, target_pos, steps=None):
        """ Linear motion of object to target position

        Parameters
        ----------
        target_pos: array_like, shape (3,)
            target position vector in units of mm.

        steps: int, optional, default=-1
            If steps=None: Object will simply be moved without generating a
                path. Specifically, path[-1] of object is set anew. This is
                similar to having steps=-1.
            If steps < 0: apply a linear motion from path[steps] to
                target_pos on top of existing path[steps:]. Specifically,
                steps=-1 will just displace path[-1].
            If steps > 0: add linear motion to target_pos to existing
                path starting at path[-1].

        Returns:
        --------
        self : object with position and orientation properties.
        """

        # steps
        steps = get_steps(steps, self)

        # avoid mm-motion calling get_steps again in
        #   .move_by() call, mm_first problem
        mm_flag = self._mm
        if mm_flag:
            self._mm = False

        # secure input
        tgt_pos = np.array(target_pos, dtype=float)

        # load current path
        path_pos = self._pos

        # call move_by
        if steps>0:
            displ = tgt_pos - path_pos[-1]
        if steps<0:
            displ = tgt_pos - path_pos[steps]
        print(displ)

        self.move_by(displ, steps=steps)

        # after motion application turn _mm on again
        if mm_flag:
            self._mm = True

        return self


    def rotate(self, rot:R, anchor=None, steps=None):
        """ Object rotation

        Parameters
        ----------
        rot: scipy Rotation object

        anchor: None or array_like, shape (3,), default=None
            The axis of rotation passes through the anchor point. When anchor=None
            the object will rotate about its own center.

        steps: int, optional, default=-1
            If steps=None: Object will simply be rotated without generating a
                path. Specifically, path[-1] of object is set anew. This is
                similar to having steps=-1.
            If steps < 0: apply linear rotation steps from 0 to rot on top
                of existing path[steps:]. Specifically, steps=-1 will just
                rotate path[-1].
            If steps > 0: add linear rotation steps from 0 to rot to existing
                path starting at path[-1].

        Returns:
        --------
        self : object with position and orientation properties.
        """

        # steps
        steps = get_steps(steps, self)

        # secure input type
        if anchor is not None:
            anchor = np.array(anchor, dtype=float)      # if None

        # load current path
        path_pos = self._pos
        path_rot = self._rot.as_quat()

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


    def rotate_from_angax(self, angle, axis, anchor=None, steps=None, degree=True):
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
            If steps=None: Object will simply be rotated without generating a
                path. Specifically, path[-1] of object is set anew. This is
                similar to having steps=-1.
            If steps < 0: apply linear rotation steps from 0 to rot on top
                of existing path[steps:]. Specifically, steps=-1 will just
                rotate path[-1].
            If steps > 0: add linear rotation steps from 0 to rot to existing
                path starting at path[-1].

        Returns:
        --------
        self : object with position and orientation properties.
        """

        # steps
        steps = get_steps(steps, self)

        # avoid mm-motion calling get_steps again
        #   in .rotate() call, mm_first problem
        mm_flag = self._mm
        if mm_flag:
            self._mm = False

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
                sys.exit('ERROR: src.rotate_from_angax() - bad axis input')
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

        # after motion application turn _mm on again
        if mm_flag:
            self._mm = True

        return self


def get_steps(steps:int, bg:BaseGeo):
    """ compute correct steps

    Parameters
    ----------
    steps: int
        number of steps

    bg: object that inherits BaseGeo
        object for which move operations are applied

    Returns:
    --------
        steps: int
    """
    # pylint: disable=protected-access

    # motion_merge
    if bg._mm:
        if bg._mm_first:
            steps = bg._mm_steps
            assert steps>0, 'ERROR: motion_merge - steps must be larger than 0'
            bg._mm_first = False
        else:
            steps = -bg._mm_steps

    # normal_motion
    else:
        # set default value
        if steps is None:
            steps = -1

        # bad steps input, set to max size
        path_len = len(bg._pos)
        if steps < -path_len:
            steps = -path_len + 1  # path[0] sees 0 rot
            print('WARNING: src.motion() - steps<path_len, setting to max-1')

    return steps


@contextmanager
def motion_merge(obj,steps):
    """combine object motions

    Parameters:
    -----------
    obj: moveable object
        object or collection to which a combined motion
        should be applied

    steps: int
        Number of steps of the combined motion
    """
    # pylint: disable=protected-access

    # enter
    if isinstance(obj, Collection):
        for src in obj:
            src._mm = True
            src._mm_steps = steps
    else:
        obj._mm = True
        obj._mm_steps = steps

    yield obj

    # exit
    if isinstance(obj, Collection):
        for src in obj:
            src._mm = False
            src._mm_steps = None
            src._mm_first = True
    else:
        obj._mm = False
        obj._mm_steps = None
        obj._mm_first = True

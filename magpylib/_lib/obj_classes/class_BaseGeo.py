"""BaseGeo class code"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib._lib.obj_classes.class_Collection import Collection
from magpylib._lib.exceptions import MagpylibBadUserInput, MagpylibBadInputShape
from magpylib._lib.config import Config


# ALL METHODS ON INTERFACE
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

    # properties ----------------------------------------------------
    @property
    def pos(self):
        """ Object position-path.

        Returns
        -------
        object position-path: np.array, shape (3,) or (N,3)
        """
        return np.squeeze(self._pos)


    @pos.setter
    def pos(self, position):
        """ Set object position-path.

        position: array_like, shape (3,) or (N,3)
            Position-path of object.
        """
        # secure input
        position = np.array(position, dtype=float)

        # input check
        if Config.CHECK_INPUTS:
            if position.shape[-1] != 3:
                raise MagpylibBadInputShape('Bad position input shape. Must be (3,) or (N,3)')
            #if position.ndim > 2:
            #    raise MagpylibBadInputShape('Bad position input shape. Must be (3,) or (N,3)')

        # create correct shape of _pos
        if position.ndim == 1:           # single position (3,) -> increase shape to (1,3)
            self._pos = np.expand_dims(position,0)
        else:                            # multi position (N,3)
            self._pos = position

    @property
    def rot(self):
        """ Object rotation-path relative to init_state.

        Returns
        -------
        object rotation-path: scipy Rotation object, shape (1,) or (N,)
            Set rotation-path of object.
        """

        if len(self._rot)==1:           # single path rotation - reduce dimension
            return self._rot[0]
        return self._rot                # return full path


    @rot.setter
    def rot(self, inp: R):
        """ Set object rotation-path.

        inp: scipy Rotation object, shape (1,) or (N,), default=None
            Set rotation-path of object. None generates a unit rotation
            for every path step.
        """
        # None inp generates unit rotation
        if inp is None:
            path_length = len(self._pos)
            self._rot = R.from_quat([(0,0,0,1)]*path_length)

        # create correct shape of _rot
        else:                        # single rotation - increase dimension to (1,3)
            val = inp.as_quat()
            if val.ndim == 1:
                self._rot = R.from_quat([val])
            else:                    # multi rotation
                self._rot = inp


    # dunders -------------------------------------------------------
    def __add__(self, source):
        """ sources add up to a Collection object
        """
        return Collection(self,source)


    # methods -------------------------------------------------------
    def reset_path(self):
        """
        Set object.pos to (0,0,0) and object.rot to unit rotation.
        """
        self.pos = (0,0,0)
        self.rot = R.from_quat((0,0,0,1))


    def move(self, displacement, start=-1, increment=False):
        """
        Translates the object by the input displacement (can be a path).

        Parameters
        ----------
        displacement: array_like, shape (3,) or (N,3), units [mm]
            Displacement vector (3,) or path (N,3) in units of [mm].

        start: int or str, default=-1
            Choose at which index of the original object path, the input path will begin.
            If `start=-1`, inp_path will start at the last old_path position.
            If `start=0`, inp_path will start with the beginning of the old_path.
            If `start=len(old_path)` or `start='attach'`, inp_path will be attached to
            the old_path.

        increment: bool, default=False
            If `increment=False`, input displacements are absolute.
            If `increment=True`, input displacements are interpreted as increments of each other.
            For example, an incremental input displacement of `[(2,0,0), (2,0,0), (2,0,0)]`
            corresponds to an absolute input displacement of `[(2,0,0), (4,0,0), (6,0,0)]`.

        Note:
        -----
        'move' simply uses vector addition to merge inp_path and old_path. It keeps the old
        orientation. If inp_path extends beyond the old_path, the old_path will be padded
        by its last entry before paths are added up.

        Returns:
        --------
        self: Object with pos and rot properties
        """

        # input type check
        # if Config.CHECK_INPUTS:
        #     if not isinstance(displacement, (list, tuple, np.ndarray)):
        #         msg = 'Bad input type. Displacement must be list, tuple or ndarray.'
        #         raise MagpylibBadUserInput(msg)
        #     if not isinstance(start_pos, int):
        #         msg = 'Bad input type. start_pos must be int.'
        #         raise MagpylibBadUserInput(msg)

        # secure type and format
        inpath = np.array(displacement, dtype=float)
        if inpath.ndim == 1:
            inpath = np.expand_dims(inpath, 0)

        # load old path
        old_ppath = self._pos
        old_opath = self._rot.as_quat()

        lenop = len(old_ppath)
        lenin = len(inpath)

        # change start to positive values in [0, lenop]
        if start == 'attach':
            start = lenop
        if start<0:
            start += lenop
        start = 0 if start<0 else lenop if start>lenop else start

        # # input format check
        # if Config.CHECK_INPUTS:
        #     if displ.shape != (3,):
        #         raise MagpylibBadInputShape('Bad displacement input shape.')
        #     if abs(start) > lenop:
        #         print('Warning: Given |start| > len(old_path)! setting to len(old_path).')

        # incremental input -> absolute input
        if increment:
            for i,d in enumerate(inpath[:-1]):
                inpath[i+1] = inpath[i+1] + d

        end = start + lenin # end position of new_path

        til = end - lenop
        if til > 0: # case inpos extends beyond old_path -> tile up old_path
            old_ppath = np.pad(old_ppath, ((0,til),(0,0)), 'edge')
            old_opath = np.pad(old_opath, ((0,til),(0,0)), 'edge')
            self.rot = R.from_quat(old_opath)

        # add new_ppath to old_ppath
        old_ppath[start:end] += inpath
        self.pos = old_ppath

        return self


    def rotate(self, rot, anchor=None, start=-1, increment=False):
        """
        Rotates the object by a given rotation input (can be a path).

        Parameters
        ----------
        rot: scipy Rotation object
            Rotation to be applied. The rotation object can feature a single rotation
            of shape (3,) or a set of rotations of shape (N,3) that correspond to a path.

        anchor: None or array_like, shape (3,), default=None, unit [mm]
            The axis of rotation passes through the anchor point given in units of [mm].
            By default (`anchor=None`) the object will rotate about its own center.

        start: int or str, default=-1
            Choose at which index of the original object path, the input path will begin.
            If `start=-1`, inp_path will start at the last old_path position.
            If `start=0`, inp_path will start with the beginning of the old_path.
            If `start=len(old_path)` or `start='attach'`, inp_path will be attached to
            the old_path.

        increment: bool, default=False
            If `increment=False`, input rotations are absolute.
            If `increment=True`, input rotations are interpreted as increments of each other.

        Returns:
        --------
        self: Object with pos and rot properties

        Notes:
        ------
        'rotate' applies given rotations to the original orientation. If inp_path extends beyond
        the old_path, the old_path will be padded by its last entry before paths are added up.

        Thanks to Benjamin for pointing this natural functionality out.
        """

        # Config.CHECK_INPUTS type checks

        # secure input and format
        inrotQ = rot.as_quat()
        if inrotQ.ndim==1:
            inrotQ = np.expand_dims(inrotQ, 0)
            rot = R.from_quat(inrotQ)

        if anchor is not None:
            anchor = np.array(anchor, dtype=float)

        # Config.CHECK_INPUTS format checks (after securing array types)

        # load old path
        old_ppath = self._pos
        old_opath = self._rot.as_quat()

        lenop = len(old_ppath)
        lenin = len(inrotQ)

        # change start to positive values in [0, lenop]
        if start == 'attach':
            start = lenop
        if start<0:
            start += lenop
        start = 0 if start<0 else lenop if start>lenop else start

        # Config.CHECK_INPUTS - warning
        #     if abs(start) > lenop:
        #         print('Warning: Given |start| > len(old_path)! setting to len(old_path).')

        # incremental input -> absolute input
        #   missing Rotation object item assign to improve this code
        if increment:
            rot1 = rot[0]
            for i,r in enumerate(rot[1:]):
                rot1 = rot1*r
                inrotQ[i+1] = rot1.as_quat()
            rot = R.from_quat(inrotQ)

        end = start + lenin  # end position of new_path

        # allocate new paths
        til = end - lenop
        if til <= 0: # case inpos completely inside of existing path
            new_ppath = old_ppath
            new_opath = old_opath
        else: # case inpos extends beyond old_path -> tile up old_path
            new_ppath = np.pad(old_ppath, ((0,til),(0,0)), 'edge')
            new_opath = np.pad(old_opath, ((0,til),(0,0)), 'edge')

        # position change when there is an anchor
        if anchor is not None:
            new_ppath[start:end] -= anchor
            new_ppath[start:end] = rot.apply(new_ppath[start:end])
            new_ppath[start:end] += anchor

        # set new rotation
        oldrot = R.from_quat(new_opath[start:end])
        new_opath[start:end] = (rot*oldrot).as_quat()

        # store new position and orientation
        self.rot = R.from_quat(new_opath)
        self.pos = new_ppath

        return self


    def rotate_from_angax(self, angle, axis, anchor=None, start=-1, increment=False, degree=True):
        """
        Object rotation from angle-axis-anchor input.

        Parameters
        ----------
        angle: int/float or array_like with shape (n,) unit [deg] (by default)
            Angle of rotation, or a vector of n angles defining a rotation path in units
            of [deg] (by default).

        axis: str or array_like, shape (3,)
            The direction of the axis of rotation. Input can be a vector of shape (3,)
            or a string 'x', 'y' or 'z' to denote respective directions.

        anchor: None or array_like, shape (3,), default=None, unit [mm]
            The axis of rotation passes through the anchor point given in units of [mm].
            By default (`anchor=None`) the object will rotate about its own center.

        start: int or str, default=-1
            Choose at which index of the original object path, the input path will begin.
            If `start=-1`, inp_path will start at the last old_path position.
            If `start=0`, inp_path will start with the beginning of the old_path.
            If `start=len(old_path)` or `start='attach'`, inp_path will be attached to
            the old_path.

        increment: bool, default=False
            If `increment=False`, input rotations are absolute.
            If `increment=True`, input rotations are interpreted as increments of each other.
            For example, the incremental angles [1,1,1,2,2] correspond to the absolute angles
            [1,2,3,5,7].

        degree: bool, default=True
            By default angle is given in units of [deg]. If degree=False, angle is given
            in units of [rad].

        Notes:
        ------
        'rotate' applies given rotations to the original orientation. If inp_path extends beyond
        the old_path, the old_path will be padded by its last entry before paths are added up.

        Returns:
        --------
        self : object with position and orientation properties
        """

        # Config.CHECK_INPUTS type checks
        angle = np.array(angle)

        # secure input
        if isinstance(axis,str):                    # generate axis from string
            ax = np.array((1,0,0)) if axis=='x'\
                else np.array((0,1,0)) if axis=='y'\
                else np.array((0,0,1)) if axis=='z'\
                else MagpylibBadUserInput(f'Bad axis string input \"{axis}\"')
        else:                                       # generate axis from vector
            ax = np.array(axis, dtype=float)
            ax = axis/np.linalg.norm(axis)

        # Config.CHECK_INPUTS format checks (after type secure)
            # axis.shape != (3,)
            # axis must not be (0,0,0)

        # degree to rad
        if degree:
            angle = angle/180*np.pi

        # apply rotation
        angle = np.tile(angle, (3,1)).T
        rot = R.from_rotvec(ax*angle)
        self.rotate(rot, anchor, start, increment)

        return self

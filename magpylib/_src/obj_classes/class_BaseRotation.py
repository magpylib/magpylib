"""BaseRotation class code"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib._src.default_classes import default_settings as Config
from magpylib._src.input_checks import (
    check_start_type,
    check_increment_type,
    check_rot_type,
    check_anchor_type,
    check_anchor_format,
    check_angle_type,
    check_axis_type,
    check_degree_type,
    check_angle_format,
    check_axis_format,
)
from magpylib._src.utility import adjust_start


class BaseRotation:
    """Defines the Rotation class for magpylib objects"""

    def __init__(self, parent_class):
        self._parent_class = parent_class
        self._target_class = self._parent_class

    def _rotate(self, rotation, anchor=None, start=-1, increment=False):
        """Rotates the object in the global coordinate system by a given rotation input"""
        # pylint: disable=protected-access
        if self._parent_class._object_type == 'Collection':
            for obj in self._parent_class:
                self._target_class = obj
                self(rotation, anchor, start, increment)
            return self._parent_class
        else:
            return self(rotation, anchor, start, increment)

    def __call__(self, rotation, anchor=None, start=-1, increment=False):
        """
        Rotates the object in the global coordinate system by a given rotation input
        (can be a path).

        This method applies given rotations to the original orientation. If the input path
        extends beyond the existing path, the old path will be padded by its last entry
        before paths are added up.

        Parameters
        ----------
        rotation: scipy Rotation object
            Rotation to be applied. The rotation object can feature a single rotation
            of shape (3,) or a set of rotations of shape (N,3) that correspond to a path.

        anchor: None, 0 or array_like, shape (3,), default=None
            The axis of rotation passes through the anchor point given in units of [mm].
            By default (`anchor=None`) the object will rotate about its own center.
            `anchor=0` rotates the object about the origin (0,0,0).

        start: int or str, default=-1
            Choose at which index of the original object path, the input path will begin.
            If `start=-1`, inp_path will start at the last old_path position.
            If `start=0`, inp_path will start with the beginning of the old_path.
            If `start=len(old_path)` or `start='append'`, inp_path will be attached to
            the old_path.

        increment: bool, default=False
            If `increment=False`, input rotations are absolute.
            If `increment=True`, input rotations are interpreted as increments of each other.

        Returns
        -------
        self: Magpylib object

        Examples
        --------

        With the ``rotate`` method Magpylib objects can be rotated about their local coordinate
        system center:

        >>> import magpylib as magpy
        >>> from scipy.spatial.transform import Rotation as R
        >>> sensor = magpy.Sensor()
        >>> print(sensor.position)
        [0. 0. 0.]
        >>> print(sensor.orientation.as_euler('xyz'))
        [0. 0. 0.]
        >>> rotation_object = R.from_euler('x', 45, degrees=True)
        >>> sensor.rotate(rotation_object)
        >>> print(sensor.position)
        [0. 0. 0.]
        >>> print(sensor.orientation.as_euler('xyz', degrees=True))
        [45.  0.  0.]

        With the ``anchor`` keyword the object rotates about a designated axis that passes
        through the given anchor point:

        >>> import magpylib as magpy
        >>> from scipy.spatial.transform import Rotation as R
        >>> sensor = magpy.Sensor()
        >>> rotation_object = R.from_euler('x', 90, degrees=True)
        >>> sensor.rotate(rotation_object, anchor=(0,1,0))
        >>> print(sensor.position)
        [ 0.  1. -1.]
        >>> print(sensor.orientation.as_euler('xyz', degrees=True))
        [90.  0.  0.]

        The method can also be used to generate paths, making use of scipy.Rotation object
        vector input:

        >>> import magpylib as magpy
        >>> from scipy.spatial.transform import Rotation as R
        >>> sensor = magpy.Sensor()
        >>> rotation_object = R.from_euler('x', 90, degrees=True)
        >>> sensor.rotate(rotation_object, anchor=(0,1,0), start='append')
        >>> print(sensor.position)
        [[ 0.  0.  0.]
         [ 0.  1. -1.]]
        >>> print(sensor.orientation.as_euler('xyz', degrees=True))
        [[ 0.  0.  0.]
         [90.  0.  0.]]
        >>> rotation_object = R.from_euler('x', [10,20,30], degrees=True)
        >>> sensor.rotate(rotation_object, anchor=(0,1,0), start='append')
        >>> print(sensor.position)
        [[ 0.          0.          0.        ]
         [ 0.          1.         -1.        ]
         [ 0.          1.17364818 -0.98480775]
         [ 0.          1.34202014 -0.93969262]
         [ 0.          1.5        -0.8660254 ]]
        >>> print(sensor.orientation.as_euler('xyz', degrees=True))
        [[  0.   0.   0.]
         [ 90.   0.   0.]
         [100.   0.   0.]
         [110.   0.   0.]
         [120.   0.   0.]]

        Complex paths can be generated by making use of the ``increment`` keyword
        and the superposition of subsequent paths:

        >>> import magpylib as magpy
        >>> from scipy.spatial.transform import Rotation as R
        >>> sensor = magpy.Sensor()
        >>> rotation_object = R.from_euler('x', [10]*3, degrees=True)
        >>> sensor.rotate(rotation_object, anchor=(0,1,0), start='append', increment=True)
        >>> print(sensor.position)
        [[ 0.          0.          0.        ]
         [ 0.          0.01519225 -0.17364818]
         [ 0.          0.06030738 -0.34202014]
         [ 0.          0.1339746  -0.5       ]]
        >>> print(sensor.orientation.as_euler('xyz', degrees=True))
        [[ 0.  0.  0.]
         [10.  0.  0.]
         [20.  0.  0.]
         [30.  0.  0.]]
        >>> rotation_object = R.from_euler('z', [5]*4, degrees=True)
        >>> sensor.rotate(rotation_object, anchor=0, start=0, increment=True)
        >>> print(sensor.position)
        [[ 0.          0.          0.        ]
         [-0.00263811  0.01496144 -0.17364818]
         [-0.0156087   0.05825246 -0.34202014]
         [-0.04582201  0.12589494 -0.5       ]]
        >>> print(sensor.orientation.as_euler('xyz', degrees=True))
        [[ 0.  0.  5.]
         [10.  0. 10.]
         [20.  0. 15.]
         [30.  0. 20.]]

        """

        # check input types
        if Config.checkinputs:
            check_rot_type(rotation)
            check_anchor_type(anchor)
            check_start_type(start)
            check_increment_type(increment)

        # input anchor -> ndarray type
        if anchor is not None:
            anchor = np.array(anchor, dtype=float)

        # check format
        if Config.checkinputs:
            check_anchor_format(anchor)
            # Non need for Rotation check. R.as_quat() can only be of shape (4,) or (N,4)

        # expand rot.as_quat() to shape (1,4)
        rot = rotation
        inrotQ = rot.as_quat()
        if inrotQ.ndim == 1:
            inrotQ = np.expand_dims(inrotQ, 0)
            rot = R.from_quat(inrotQ)

        # load old path
        # pylint: disable=protected-access
        old_ppath = self._target_class._position
        old_opath = self._target_class._orientation.as_quat()

        lenop = len(old_ppath)
        lenin = len(inrotQ)

        # change start to positive values in [0, lenop]
        start = adjust_start(start, lenop)

        # incremental input -> absolute input
        #   missing Rotation object item assign to improve this code
        if increment:
            rot1 = rot[0]
            for i, r in enumerate(rot[1:]):
                rot1 = r * rot1
                inrotQ[i + 1] = rot1.as_quat()
            rot = R.from_quat(inrotQ)

        end = start + lenin  # end position of new_path

        # allocate new paths
        til = end - lenop
        if til <= 0:  # case inpos completely inside of existing path
            new_ppath = old_ppath
            new_opath = old_opath
        else:  # case inpos extends beyond old_path -> tile up old_path
            new_ppath = np.pad(old_ppath, ((0, til), (0, 0)), "edge")
            new_opath = np.pad(old_opath, ((0, til), (0, 0)), "edge")

        # position change when there is an anchor
        if anchor is not None:
            new_ppath[start:end] -= anchor
            new_ppath[start:end] = rot.apply(new_ppath[start:end])
            new_ppath[start:end] += anchor

        # set new rotation
        oldrot = R.from_quat(new_opath[start:end])
        new_opath[start:end] = (rot * oldrot).as_quat()

        # store new position and orientation
        self._target_class.orientation = R.from_quat(new_opath)
        self._target_class.position = new_ppath

        return self._target_class

    def from_angax(
        self, angle, axis, anchor=None, start=-1, increment=False, degrees=True
    ):
        """
        Object rotation in the global coordinate system from angle-axis input.

        This method applies given rotations to the original orientation. If the input path
        extends beyond the existingp path, the oldpath will be padded by its last entry before paths
        are added up.

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
            `anchor=0` rotates the object about the origin (0,0,0).

        start: int or str, default=-1
            Choose at which index of the original object path, the input path will begin.
            If `start=-1`, inp_path will start at the last old_path position.
            If `start=0`, inp_path will start with the beginning of the old_path.
            If `start=len(old_path)` or `start='append'`, inp_path will be attached to
            the old_path.

        increment: bool, default=False
            If `increment=False`, input rotations are absolute.
            If `increment=True`, input rotations are interpreted as increments of each other.
            For example, the incremental angles [1,1,1,2,2] correspond to the absolute angles
            [1,2,3,5,7].

        degrees: bool, default=True
            By default angle is given in units of [deg]. If degrees=False, angle is given
            in units of [rad].

        Returns
        -------
        self: Magpylib object

        Examples
        --------
        With the ``rotate.from_angax`` method Magpylib objects can be rotated about their local
        coordinte system center:

        >>> import magpylib as magpy
        >>> sensor = magpy.Sensor()
        >>> print(sensor.position)
        [0. 0. 0.]
        >>> print(sensor.orientation.as_euler('xyz'))
        [0. 0. 0.]
        >>> sensor.rotate.from_angax(angle=45, axis='x')
        >>> print(sensor.position)
        [0. 0. 0.]
        >>> print(sensor.orientation.as_euler('xyz', degrees=True))
        [45.  0.  0.]

        With the ``anchor`` keyword the object rotates about a designated axis
        that passes through the given anchor point:

        >>> import magpylib as magpy
        >>> sensor = magpy.Sensor()
        >>> sensor.rotate.from_angax(angle=90, axis=(1,0,0), anchor=(0,1,0))
        >>> print(sensor.position)
        [ 0.  1. -1.]
        >>> print(sensor.orientation.as_euler('xyz', degrees=True))
        [90.  0.  0.]

        The method can also be used to generate paths, making use of scipy.Rotation
        object vector input:

        >>> import magpylib as magpy
        >>> sensor = magpy.Sensor()
        >>> sensor.rotate.from_angax(angle=90, axis='x', anchor=(0,1,0), start='append')
        >>> print(sensor.position)
        [[ 0.  0.  0.]
         [ 0.  1. -1.]]
        >>> print(sensor.orientation.as_euler('xyz', degrees=True))
        [[ 0.  0.  0.]
         [90.  0.  0.]]
        >>> sensor.rotate.from_angax(angle=[10,20,30], axis='x', anchor=(0,1,0), start='append')
        >>> print(sensor.position)
        [[ 0.          0.          0.        ]
         [ 0.          1.         -1.        ]
         [ 0.          1.17364818 -0.98480775]
         [ 0.          1.34202014 -0.93969262]
         [ 0.          1.5        -0.8660254 ]]
        >>> print(sensor.orientation.as_euler('xyz', degrees=True))
        [[  0.   0.   0.]
         [ 90.   0.   0.]
         [100.   0.   0.]
         [110.   0.   0.]
         [120.   0.   0.]]

        Complex paths can be generated by making use of the ``increment`` keyword
        and the superposition of subsequent paths:

        >>> import magpylib as magpy
        >>> sensor = magpy.Sensor()
        >>> sensor.rotate.from_angax([10]*3, 'x', (0,1,0), start=1, increment=True)
        >>> print(sensor.position)
        [[ 0.          0.          0.        ]
         [ 0.          0.01519225 -0.17364818]
         [ 0.          0.06030738 -0.34202014]
         [ 0.          0.1339746  -0.5       ]]
        >>> print(sensor.orientation.as_euler('xyz', degrees=True))
        [[ 0.  0.  0.]
         [10.  0.  0.]
         [20.  0.  0.]
         [30.  0.  0.]]
        >>> sensor.rotate.from_angax(angle=[5]*4, axis='z', anchor=0, start=0, increment=True)
        >>> print(sensor.position)
        [[ 0.          0.          0.        ]
         [-0.00263811  0.01496144 -0.17364818]
         [-0.0156087   0.05825246 -0.34202014]
         [-0.04582201  0.12589494 -0.5       ]]
        >>> print(sensor.orientation.as_euler('xyz', degrees=True))
        [[ 0.  0.  5.]
         [10.  0. 10.]
         [20.  0. 15.]
         [30.  0. 20.]]

        """

        # check input types
        if Config.checkinputs:
            check_angle_type(angle)
            check_axis_type(axis)
            check_anchor_type(anchor)
            check_start_type(start)
            check_increment_type(increment)
            check_degree_type(degrees)

        # generate axis from string
        if isinstance(axis, str):
            axis = (
                (1, 0, 0)
                if axis == "x"
                else (0, 1, 0)
                if axis == "y"
                else (0, 0, 1)
                if axis == "z"
                else MagpylibBadUserInput(f'Bad axis string input "{axis}"')
            )

        # input expand and ->ndarray
        if isinstance(angle, (int, float)):
            angle = (angle,)
        angle = np.array(angle, dtype=float)
        axis = np.array(axis, dtype=float)

        # format checks
        if Config.checkinputs:
            check_angle_format(angle)
            check_axis_format(axis)
            # anchor check in .rotate()

        # Config.checkinputs format checks (after type secure)
        # axis.shape != (3,)
        # axis must not be (0,0,0)

        # degree to rad
        if degrees:
            angle = angle / 180 * np.pi

        # apply rotation
        angle = np.tile(angle, (3, 1)).T
        axis = axis / np.linalg.norm(axis)
        rot = R.from_rotvec(axis * angle)
        self._rotate(rot, anchor, start, increment)

        return self._parent_class

    def from_rotvec(
        self, rotvec, anchor=None, start=-1, increment=False, degrees=False
    ):
        """
        Initialize from rotation vectors.

        A rotation vector is a 3 dimensional vector which is co-directional to
        the axis of rotation and whose norm gives the angle of rotation [1]_.

        Parameters
        ----------
        rotvec : array_like, shape (N, 3) or (3,)
            A single vector or a stack of vectors, where `rot_vec[i]` gives
            the ith rotation vector.
        degrees : bool, optional
            If True, then the given magnitudes are assumed to be in degrees.
            Default is False.
        """
        rot = R.from_rotvec(rotvec, degrees=degrees)
        return self._rotate(rot, anchor=anchor, start=start, increment=increment)

    def from_euler(
        self, seq, angles, anchor=None, start=-1, increment=False, degrees=False
    ):
        """Initialize from Euler angles.

        Rotations in 3-D can be represented by a sequence of 3
        rotations around a sequence of axes. In theory, any three axes spanning
        the 3-D Euclidean space are enough. In practice, the axes of rotation are
        chosen to be the basis vectors.

        The three rotations can either be in a global frame of reference
        (extrinsic) or in a body centred frame of reference (intrinsic), which
        is attached to, and moves with, the object under rotation [1]_.

        Parameters
        ----------
        seq : string
            Specifies sequence of axes for rotations. Up to 3 characters
            belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations, or
            {'x', 'y', 'z'} for extrinsic rotations. Extrinsic and intrinsic
            rotations cannot be mixed in one function call.
        angles : float or array_like, shape (N,) or (N, [1 or 2 or 3])
            Euler angles specified in radians (`degrees` is False) or degrees
            (`degrees` is True).
            For a single character `seq`, `angles` can be:

            - a single value
            - array_like with shape (N,), where each `angle[i]`
            corresponds to a single rotation
            - array_like with shape (N, 1), where each `angle[i, 0]`
            corresponds to a single rotation

            For 2- and 3-character wide `seq`, `angles` can be:

            - array_like with shape (W,) where `W` is the width of
            `seq`, which corresponds to a single rotation with `W` axes
            - array_like with shape (N, W) where each `angle[i]`
            corresponds to a sequence of Euler angles describing a single
            rotation

        degrees : bool, optional
            If True, then the given angles are assumed to be in degrees.
            Default is False.
        """
        rot = R.from_euler(seq, angles, degrees=degrees)
        return self._rotate(rot, anchor=anchor, start=start, increment=increment)

    def from_matrix(self, matrix, anchor=None, start=-1, increment=False):
        """
        Initialize from rotation matrix.

        Rotations in 3 dimensions can be represented with 3 x 3 proper
        orthogonal matrices [1]_. If the input is not proper orthogonal,
        an approximation is created using the method described in [2]_.

        Parameters
        ----------
        matrix : array_like, shape (N, 3, 3) or (3, 3)
            A single matrix or a stack of matrices, where ``matrix[i]`` is
            the i-th matrix.
        """
        rot = R.from_matrix(matrix)
        return self._rotate(rot, anchor=anchor, start=start, increment=increment)

    def from_mrp(self, mrp, anchor=None, start=-1, increment=False):
        """
        Initialize from Modified Rodrigues Parameters (MRPs).

        MRPs are a 3 dimensional vector co-directional to the axis of rotation and whose
        magnitude is equal to ``tan(theta / 4)``, where ``theta`` is the angle of rotation
        (in radians) [1]_.

        MRPs have a singuarity at 360 degrees which can be avoided by ensuring the angle of
        rotation does not exceed 180 degrees, i.e. switching the direction of the rotation when
        it is past 180 degrees.

        Parameters
        ----------
        mrp : array_like, shape (N, 3) or (3,)
            A single vector or a stack of vectors, where `mrp[i]` gives
            the ith set of MRPs.
        """
        rot = R.from_mrp(mrp)
        return self._rotate(rot, anchor=anchor, start=start, increment=increment)

    def from_quat(self, quat, anchor=None, start=-1, increment=False):
        """
        Initialize from quaternions.

        3D rotations can be represented using unit-norm quaternions [1]_.

        Parameters
        ----------
        quat : array_like, shape (N, 4) or (4,)
            Each row is a (possibly non-unit norm) quaternion in scalar-last
            (x, y, z, w) format. Each quaternion will be normalized to unit
            norm.
        """
        rot = R.from_quat(quat)
        return self._rotate(rot, anchor=anchor, start=start, increment=increment)

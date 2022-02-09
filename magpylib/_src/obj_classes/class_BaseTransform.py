"""BaseTransform class code"""
# pylint: disable=too-many-instance-attributes
# pylint: disable=protected-access

import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib._src.defaults.defaults_classes import default_settings as Config
from magpylib._src.input_checks import (
    check_start_type,
    check_rot_type,
    check_anchor_type,
    check_anchor_format,
    check_angle_type,
    check_axis_type,
    check_degree_type,
    check_angle_format,
    check_axis_format,
    check_vector_type,
    check_path_format,
)


def multi_anchor_behavior(anchor, inrotQ, rotation):
    """define behavior of rotation with given anchor
    if one is longer than the other pad up other
    """
    len_inrotQ = 0 if inrotQ.ndim == 1 else inrotQ.shape[0]
    len_anchor = 0 if anchor.ndim == 1 else anchor.shape[0]

    if len_inrotQ > len_anchor:
        if len_anchor == 0:
            anchor = np.reshape(anchor, (1, 3))
            len_anchor = 1
        anchor = np.pad(anchor, ((0, len_inrotQ - len_anchor), (0, 0)), "edge")
    elif len_inrotQ < len_anchor:
        if len_inrotQ == 0:
            inrotQ = np.reshape(inrotQ, (1, 4))
            len_inrotQ = 1
        inrotQ = np.pad(inrotQ, ((0, len_anchor - len_inrotQ), (0, 0)), "edge")
        rotation = R.from_quat(inrotQ)

    return anchor, inrotQ, rotation


def path_padding_param(scalar_input: bool, lenop: int, lenip: int, start):
    """
    compute path padding parameters

    Example: with start>0 input path exceeds old_path
        old_path:            |abcdefg|
        input_path:              |xzyuvwrst|
        -> padded_old_path:  |abcdefggggggg|

    Parameters:
    -----------
    scalar_input: True if rotation input is scalar, else False
    lenop: length of old_path
    lenip: length of input_path
    start: start index

    Returns:
    --------
    padding: (pad_before, pad_behind)
        how much the old_path must be padded before
    start: modified start value
    """
    # initialize paddings
    pad_before = 0
    pad_behind = 0

    # start='auto': apply to all if scalar, append if vector
    if start == "auto":
        if scalar_input:
            start = 0
        else:
            start = lenop

    # numpy convention with negative start indices
    if start < 0:
        start = lenop + start
        # if start smaller than -old_path_length: pad before
        if start < 0:
            pad_before = -start  # pylint: disable=invalid-unary-operand-type
            start = 0

    # vector: if start+inpath extends beyond oldpath: pad behind
    if start + lenip > lenop + pad_before:
        pad_behind = start + lenip - (lenop + pad_before)

    if pad_before + pad_behind > 0:
        return (pad_before, pad_behind), start
    return [], start


def path_padding(inpath, start, target_object):
    """
    pad path of target_object and compute start- and end-index for apply_move()
    and apply_rotation() functions below so that ppath[start:end] = X... can be
    applied.

    Parameters
    ----------
    inpath: user input as np.ndarray
    start: start index
    target_object: magpylib object with position and orientation attributes

    Returns
    -------
    ppath: padded target_object position path
    opath: padded target_object orientation path
    start: modified start idex
    end: end index
    padded: True if padding was necessary, else False
    """
    # scalar or vector input
    scalar_input = inpath.ndim == 1

    # load old path
    ppath = target_object._position
    opath = target_object._orientation.as_quat()

    lenip = 1 if scalar_input else len(inpath)

    # pad old path depending on input
    padding, start = path_padding_param(scalar_input, len(ppath), lenip, start)
    if padding:
        ppath = np.pad(ppath, (padding, (0, 0)), "edge")
        opath = np.pad(opath, (padding, (0, 0)), "edge")

    # set end-index
    end = len(ppath) if scalar_input else start + lenip

    return ppath, opath, start, end, bool(padding)


def apply_move(target_object, displacement, start="auto"):
    """
    Implementation of the move() functionality.

    Parameters
    ----------
    target_object: object with position and orientation attributes
    displacement: displacement vector/path, array_like, shape (3,) or (n,3).
        If the input is scalar (shape (3,)) the operation is applied to the
        whole path. If the input is a vector (shape (n,3)), it is
        appended/merged with the existing path.
    start: int, str, default='auto'
        start=i applies an operation starting at the i'th path index.
        With start='auto' and scalar input the wole path is moved. With
        start='auto' and vector input the input is appended.

    Returns
    -------
    target_object
    """
    # pylint: disable=protected-access
    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=too-many-branches

    # check input types
    if Config.checkinputs:
        check_vector_type(displacement, "displacement")
        check_start_type(start)

    # displacement vector -> ndarray
    inpath = np.array(displacement, dtype=float)

    # check input format
    if Config.checkinputs:
        check_path_format(inpath, "displacement")

    # pad target_object path and compute start and end-index for rotation application
    ppath, opath, start, end, padded = path_padding(inpath, start, target_object)
    if padded:
        target_object._orientation = R.from_quat(opath)

    # apply move operation
    ppath[start:end] += inpath
    target_object._position = ppath

    return target_object


def apply_rotation(
    target_object, rotation: R, anchor=None, start="auto", parent_path=None
):
    """
    Implementation of the rotate() functionality.

    Parameters
    ----------
    target_object: object with position and orientation attributes
    rotation: a scipy Rotation object
        If the input is scalar (shape (3,)) the operation is applied to the
        whole path. If the input is a vector (shape (n,3)), it is
        appended/merged with the existing path.
    anchor: array_like shape (3,)
        Rotation anchor
    start: int, str, default='auto'
        start=i applies an operation starting at the i'th path index.
        With start='auto' and scalar input the wole path is moved. With
        start='auto' and vector input the input is appended.
    parent_path=None if there is no parent else parent._position

    Returns
    -------
    target_object
    """
    # pylint: disable=protected-access
    # pylint: disable=too-many-branches

    # check input types
    if Config.checkinputs:
        check_rot_type(rotation)
        check_anchor_type(anchor)
        check_start_type(start)

    # input -> quaternion ndarray
    inrotQ = rotation.as_quat()

    # when an anchor is given
    if anchor is not None:
        # 0-anchor -> (0,0,0)
        if np.isscalar(anchor) and anchor == 0:
            anchor = np.array((0.0, 0.0, 0.0))
        else:
            anchor = np.array(anchor, dtype=float)
        # check anchor input format
        if Config.checkinputs:
            check_anchor_format(anchor)
        # apply multi-anchor behavior
        anchor, inrotQ, rotation = multi_anchor_behavior(anchor, inrotQ, rotation)

    # pad target_object path and compute start and end-index for rotation application
    ppath, opath, newstart, end, _ = path_padding(inrotQ, start, target_object)

    # compute anchor when dealing with Compound rotation (target_object is a child
    #   that rotates about its parent). This happens when a rotation with anchor=None
    #   is applied to a child in a Collection. In this case the anchor must be set to
    #   the parent_path.
    if anchor is None and parent_path is not None:
        # target anchor length
        len_anchor = end - newstart
        # pad up parent_path if input requires it
        padding, start = path_padding_param(
            inrotQ.ndim == 1, parent_path.shape[0], len_anchor, start
        )
        if padding:
            parent_path = np.pad(parent_path, (padding, (0, 0)), "edge")
        # slice anchor from padded parent_path
        anchor = parent_path[start : start + len_anchor]

    # position change when there is an anchor
    if anchor is not None:
        ppath[newstart:end] -= anchor
        ppath[newstart:end] = rotation.apply(ppath[newstart:end])
        ppath[newstart:end] += anchor

    # set new rotation
    oldrot = R.from_quat(opath[newstart:end])
    opath[newstart:end] = (rotation * oldrot).as_quat()

    # store new position and orientation
    # pylint: disable=attribute-defined-outside-init
    target_object._orientation = R.from_quat(opath)
    target_object._position = ppath

    return target_object


class BaseTransform:
    """
    Inherit this class to provide rotation() and move() methods.

    All rotate_from_XXX methods simply generate a scipy Rotation object and hand it
    over to the main rotate() method. This then uses the apply_rotation function to
    apply the rotations to all target objects.

    - For Magpylib objects that inherit BaseRotate and BaseGeo (e.g. Cuboid()),
      apply_rotation() is applied only to the object itself.
    - Collections inherit only BaseRotate. In this case apply_rotation() is only
      applied to the Collection children.
    - Compounds are user-defined classes that inherit Collection but also inherit
      BaseGeo. In this case apply_rotation() is applied to the object itself, but also
      to its children.
    """

    def move(self, displacement, start="auto"):
        """

        Input Statement
        If the input is a scalar the operation is applied to the whole path. If the
        input is a vector, it is merged with the existing path.

        Start Statement
        start=i applies an operation starting at the i'th path index. start=None applies
        an operation to the whole path.

        Translates the object by the input displacement (can be a path).

        This method uses vector addition to merge the input path given by displacement and the
        existing old path of an object. It keeps the old orientation. If the input path extends
        beyond the old path, the old path will be padded by its last entry before paths are
        added up.

        Parameters
        ----------
        displacement: array_like, shape (3,) or (N,3)
            Displacement vector shape=(3,) or path shape=(N,3) in units of [mm].

        start: int or str, default=-1
            Choose at which index of the original object path, the input path will begin.
            If `start=-1`, inp_path will start at the last old_path position.
            If `start=0`, inp_path will start with the beginning of the old_path.
            If `start=len(old_path)` or `start='append'`, inp_path will be attached to
            the old_path.

        Returns
        -------
        self: Magpylib object

        Examples
        --------

        With the ``move`` method Magpylib objects can be repositioned in the global coordinate
        system:

        >>> import magpylib as magpy
        >>> sensor = magpy.Sensor()
        >>> print(sensor.position)
        [0. 0. 0.]
        >>> sensor.move((1,1,1))
        >>> print(sensor.position)
        [1. 1. 1.]

        It is also a powerful tool for creating paths:

        >>> import magpylib as magpy
        >>> sensor = magpy.Sensor()
        >>> sensor.move((1,1,1), start='append')
        >>> print(sensor.position)
        [[0. 0. 0.]
         [1. 1. 1.]]
        >>> sensor.move([(.1,.1,.1)]*2, start='append')
        >>> print(sensor.position)
        [[0.  0.  0. ]
         [1.  1.  1. ]
         [1.1 1.1 1.1]
         [1.1 1.1 1.1]]

        Complex paths can be generated with ease, by making use of the ``increment`` keyword
        and superposition of subsequent paths:

        >>> import magpylib as magpy
        >>> sensor = magpy.Sensor()
        >>> sensor.move([(1,1,1)]*4, start='append', increment=True)
        >>> print(sensor.position)
        [[0. 0. 0.]
         [1. 1. 1.]
         [2. 2. 2.]
         [3. 3. 3.]
         [4. 4. 4.]]
        >>> sensor.move([(.1,.1,.1)]*5, start=2)
        >>> print(sensor.position)
        [[0.  0.  0. ]
         [1.  1.  1. ]
         [2.1 2.1 2.1]
         [3.1 3.1 3.1]
         [4.1 4.1 4.1]
         [4.1 4.1 4.1]
         [4.1 4.1 4.1]]
        """

        # Idea: An operation applied to a Collection is individually
        #    applied to its BaseGeo and to each child.

        for child in getattr(self, "children", []):
            apply_move(child, displacement, start=start)

        apply_move(self, displacement, start=start)

        return self

    def rotate(self, rotation: R, anchor=None, start="auto"):
        """
        Rotates object in the global coordinate system using a scipy Rotation object
        as input.

        Parameters
        ----------
        rotation: scipy Rotation object
            Rotation to be applied to existing object orientation. The scipy Rotation
            object can be of shape (3,) or (N,3).

        anchor: None, 0 or array_like, shape (3,), default=None
            The axis of rotation passes through the anchor point given in units of [mm].
            By default (`anchor=None`) the object will rotate about its own center.
            `anchor=0` rotates the object about the origin (0,0,0).

        start: int or str, default=-1
            Choose at which index of the original object path, the input path will begin.
            If `start=-1`, inp_path will overwrite the last old_path position.
            If `start=0`, inp_path will start with the beginning of the old_path.
            If `start=len(old_path)` or `start='append'`, inp_path will be attached to
            the old_path.

        Returns
        -------
        self: Magpylib object

        Examples
        --------
        >>> from scipy.spatial.transform import Rotation as R
        >>> import magpylib as magpy
        >>> s = magpy.Sensor(position=(1,0,0))

        print initial position and orientation

        >>> print(s.position)
        >>> print(s.orientation.as_euler('xyz', degrees=True))
        [1. 0. 0.]
        [0. 0. 0.]

        rotate and print resulting position and orientation

        >>> s.rotate(R.from_euler('z', 45, degrees=True), anchor=0)
        >>> print(s.position)
        >>> print(s.orientation.as_euler('xyz', degrees=True))
        [0.70710678 0.70710678 0.        ]
        [ 0.  0. 45.]
        """

        # pylint: disable=no-member

        # Idea: An operation applied to a Collection is individually
        #    applied to its BaseGeo and to each child.
        #  -> this automatically generates the rotate-Compound behavior

        for child in getattr(self, "children", []):
            apply_rotation(
                child, rotation, anchor=anchor, start=start, parent_path=self._position
            )

        apply_rotation(self, rotation, anchor=anchor, start=start)

        return self

    def rotate_from_angax(self, angle, axis, anchor=None, start="auto", degrees=True):
        """
        Rotates object in the global coordinate system from angle-axis input.

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
            If `start=-1`, inp_path will overwrite the last old_path position.
            If `start=0`, inp_path will start with the beginning of the old_path.
            If `start=len(old_path)` or `start='append'`, inp_path will be attached to
            the old_path.

        degrees: bool, default=True
            By default angle is given in units of [deg]. If degrees=False, angle is given
            in units of [rad].

        Returns
        -------
        self: Magpylib object

        Examples
        --------
        >>> import magpylib as magpy
        >>> s = magpy.Sensor(position=(1,0,0))

        print initial position and orientation

        >>> print(s.position)
        >>> print(s.orientation.as_euler('xyz', degrees=True))
        [1. 0. 0.]
        [0. 0. 0.]

        rotate and print resulting position and orientation

        >>> s.rotate_from_angax(45, (0,0,1), anchor=0)
        >>> print(s.position)
        >>> print(s.orientation.as_euler('xyz', degrees=True))
        [0.70710678 0.70710678 0.        ]
        [ 0.  0. 45.]
        """

        # check input types
        if Config.checkinputs:
            check_angle_type(angle)
            check_axis_type(axis)
            check_anchor_type(anchor)
            check_start_type(start)
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

        # input is scalar or vector
        is_scalar = isinstance(angle, (int, float))

        # secure type - scalar angle will become a float
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
        if is_scalar:
            angle = np.ones(3) * angle
        else:
            angle = np.tile(angle, (3, 1)).T

        # generate rotation object from rotvec
        axis = axis / np.linalg.norm(axis)
        rot = R.from_rotvec(axis * angle)

        return self.rotate(rot, anchor, start)

    def rotate_from_rotvec(self, rotvec, anchor=None, start="auto", degrees=False):
        """
        Rotates object in the global coordinate system from rotation vector input. (vector
        direction is the rotation axis, vector length is the rotation angle in [rad])

        Parameters
        ----------
        rotvec : array_like, shape (N, 3) or (3,)
            A single vector or a stack of vectors, where `rot_vec[i]` gives
            the ith rotation vector.

        anchor: None or array_like, shape (3,), default=None, unit [mm]
            The axis of rotation passes through the anchor point given in units of [mm].
            By default (`anchor=None`) the object will rotate about its own center.
            `anchor=0` rotates the object about the origin (0,0,0).

        start: int or str, default=-1
            Choose at which index of the original object path, the input path will begin.
            If `start=-1`, inp_path will overwrite the last old_path position.
            If `start=0`, inp_path will start with the beginning of the old_path.
            If `start=len(old_path)` or `start='append'`, inp_path will be attached to
            the old_path.

        degrees : bool, default False
            If True, then the given angles are assumed to be in degrees.

        Returns
        -------
        self: Magpylib object

        Examples
        --------
        >>> import magpylib as magpy
        >>> s = magpy.Sensor(position=(1,0,0))

        print initial position and orientation

        >>> print(s.position)
        >>> print(s.orientation.as_rotvec())
        [1. 0. 0.]
        [0. 0. 0.]

        rotate and print resulting position and orientation

        >>> s.rotate_from_rotvec((0,0,1), anchor=0)
        >>> print(s.position)
        >>> print(s.orientation.as_rotvec())
        [0.54030231 0.84147098 0.        ]
        [0. 0. 1.]
        """
        rot = R.from_rotvec(rotvec, degrees=degrees)
        return self.rotate(rot, anchor=anchor, start=start)

    def rotate_from_euler(self, seq, angles, anchor=None, start="auto", degrees=False):
        """
        Rotates object in the global coordinate system from Euler angle input.

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

        anchor: None or array_like, shape (3,), default=None, unit [mm]
            The axis of rotation passes through the anchor point given in units of [mm].
            By default (`anchor=None`) the object will rotate about its own center.
            `anchor=0` rotates the object about the origin (0,0,0).

        start: int or str, default=-1
            Choose at which index of the original object path, the input path will begin.
            If `start=-1`, inp_path will overwrite the last old_path position.
            If `start=0`, inp_path will start with the beginning of the old_path.
            If `start=len(old_path)` or `start='append'`, inp_path will be attached to
            the old_path.

        degrees : bool, default False
            If True, then the given angles are assumed to be in degrees.

        Returns
        -------
        self: Magpylib object

        Examples
        --------
        >>> import magpylib as magpy
        >>> s = magpy.Sensor(position=(1,0,0))

        print initial position and orientation

        >>> print(s.position)
        >>> print(s.orientation.as_euler('xyz', degrees=True))
        [1. 0. 0.]
        [0. 0. 0.]

        rotate and print resulting position and orientation

        >>> s.rotate_from_euler('z', 45, anchor=0, degrees=True)
        >>> print(s.position)
        >>> print(s.orientation.as_euler('xyz', degrees=True))
        [0.70710678 0.70710678 0.        ]
        [ 0.  0. 45.]
        """
        rot = R.from_euler(seq, angles, degrees=degrees)
        return self.rotate(rot, anchor=anchor, start=start)

    def rotate_from_matrix(self, matrix, anchor=None, start="auto"):
        """
        Rotates object in the global coordinate system from matrix input.
        (see scipy rotation package matrix input)

        Parameters
        ----------
        matrix : array_like, shape (N, 3, 3) or (3, 3)
            A single matrix or a stack of matrices, where `matrix[i]` is
            the i-th matrix.

        anchor: None or array_like, shape (3,), default=None, unit [mm]
            The axis of rotation passes through the anchor point given in units of [mm].
            By default (`anchor=None`) the object will rotate about its own center.
            `anchor=0` rotates the object about the origin (0,0,0).

        start: int or str, default=-1
            Choose at which index of the original object path, the input path will begin.
            If `start=-1`, inp_path will overwrite the last old_path position.
            If `start=0`, inp_path will start with the beginning of the old_path.
            If `start=len(old_path)` or `start='append'`, inp_path will be attached to
            the old_path.

        Returns
        -------
        self: Magpylib object

        Examples
        --------
        >>> import magpylib as magpy
        >>> s = magpy.Sensor(position=(1,0,0))

        print initial position and orientation

        >>> print(s.position)
        >>> print(s.orientation.as_matrix())
        [1. 0. 0.]
        [[1. 0. 0.]
         [0. 1. 0.]
         [0. 0. 1.]]

        rotate and print resulting position and orientation

        >>> s.rotate_from_matrix([(0,-1,0),(1,0,0),(0,0,1)], anchor=0)
        >>> print(s.position)
        >>> print(s.orientation.as_matrix())
        [0. 1. 0.]
        [[ 0. -1.  0.]
         [ 1.  0.  0.]
         [ 0.  0.  1.]]
        """
        rot = R.from_matrix(matrix)
        return self.rotate(rot, anchor=anchor, start=start)

    def rotate_from_mrp(self, mrp, anchor=None, start="auto"):
        """
        Rotates object in the global coordinate system from Modified Rodrigues Parameters input.
        (see scipy rotation package Modified Rodrigues Parameters (MRPs))

        Parameters
        ----------
        mrp : array_like, shape (N, 3) or (3,)
            A single vector or a stack of vectors, where `mrp[i]` gives the ith set of MRPs.

        anchor: None or array_like, shape (3,), default=None, unit [mm]
            The axis of rotation passes through the anchor point given in units of [mm].
            By default (`anchor=None`) the object will rotate about its own center.
            `anchor=0` rotates the object about the origin (0,0,0).

        start: int or str, default=-1
            Choose at which index of the original object path, the input path will begin.
            If `start=-1`, inp_path will overwrite the last old_path position.
            If `start=0`, inp_path will start with the beginning of the old_path.
            If `start=len(old_path)` or `start='append'`, inp_path will be attached to
            the old_path.

        Returns
        -------
        self: Magpylib object

        Examples
        --------
        >>> import magpylib as magpy
        >>> s = magpy.Sensor(position=(1,0,0))

        print initial position and orientation

        >>> print(s.position)
        >>> print(s.orientation.as_mrp())
        [1. 0. 0.]
        [0. 0. 0.]

        rotate and print resulting position and orientation

        >>> s.rotate_from_mrp((0,0,1), anchor=0)
        >>> print(s.position)
        >>> print(s.orientation.as_mrp())
        [-1.  0.  0.]
        [0. 0. 1.]
        """
        rot = R.from_mrp(mrp)
        return self.rotate(rot, anchor=anchor, start=start)

    def rotate_from_quat(self, quat, anchor=None, start="auto"):
        """
        Rotates object in the global coordinate system from Quaternion input.

        Parameters
        ----------
        quat : array_like, shape (N, 4) or (4,)
            Each row is a (possibly non-unit norm) quaternion in scalar-last
            (x, y, z, w) format. Each quaternion will be normalized to unit
            norm.

        anchor: None or array_like, shape (3,), default=None, unit [mm]
            The axis of rotation passes through the anchor point given in units of [mm].
            By default (`anchor=None`) the object will rotate about its own center.
            `anchor=0` rotates the object about the origin (0,0,0).

        start: int or str, default=-1
            Choose at which index of the original object path, the input path will begin.
            If `start=-1`, inp_path will overwrite the last old_path position.
            If `start=0`, inp_path will start with the beginning of the old_path.
            If `start=len(old_path)` or `start='append'`, inp_path will be attached to
            the old_path.

        Returns
        -------
        self: Magpylib object

        Examples
        --------
        >>> import magpylib as magpy
        >>> s = magpy.Sensor(position=(1,0,0))

        print initial position and orientation
        >>> print(s.position)
        >>> print(s.orientation.as_quat())
        [1. 0. 0.]
        [0. 0. 0. 1.]

        rotate and print resulting position and orientation
        >>> s.rotate_from_quat((0,0,1,1), anchor=0)
        >>> print(s.position)
        >>> print(s.orientation.as_quat())
        [0. 1. 0.]
        [0.         0.         0.70710678 0.70710678]
        """
        rot = R.from_quat(quat)
        return self.rotate(rot, anchor=anchor, start=start)

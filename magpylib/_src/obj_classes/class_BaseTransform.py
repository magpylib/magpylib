"""BaseTransform class code"""
# pylint: disable=too-many-instance-attributes
# # pylint: disable=protected-access

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
    check_absolute_type,
)


def apply_rotation(target_object, rotation, anchor=None, start="auto"):
    """
    Implementation of the rotate() functionality.

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

    It is difficult to interpret absolute when rotation anchor is not None and
        a rotation with respect to another anchor already exists. This is a feature
        that must be discussed.
    """
    # pylint: disable=protected-access
    # pylint: disable=too-many-branches

    # check input types
    if Config.checkinputs:
        check_rot_type(rotation)
        check_anchor_type(anchor)
        check_start_type(start)

    # input anchor -> ndarray type
    if anchor is not None:
        anchor = np.array(anchor, dtype=float)

    # check format
    if Config.checkinputs:
        check_anchor_format(anchor)
        # Non need for Rotation format check. R.as_quat() can only be of shape (4,) or (N,4)

    # input -> quaternion ndarray
    inrotQ = rotation.as_quat()
    scalar_input = inrotQ.ndim == 1

    # load old path
    ppath = target_object._position
    opath = target_object._orientation.as_quat()

    # path lengths
    lenop = len(ppath)
    lenip = 1 if scalar_input else len(inrotQ)

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

    # vector: if start+inpath extends beyond oldpath: pad behind and merge
    if start + lenip > lenop + pad_before:
        pad_behind = start + lenip - (lenop + pad_before)

    # avoid execution when there is no padding (cost~100ns)
    if pad_before + pad_behind:
        ppath = np.pad(ppath, ((pad_before, pad_behind), (0, 0)), "edge")
        opath = np.pad(opath, ((pad_before, pad_behind), (0, 0)), "edge")

    # set end-index
    if scalar_input:
        end = len(ppath)
    else:
        end = start + lenip

    # position change when there is an anchor
    if anchor is not None:
        ppath[start:end] -= anchor
        ppath[start:end] = rotation.apply(ppath[start:end])
        ppath[start:end] += anchor

    # set new rotation
    oldrot = R.from_quat(opath[start:end])
    opath[start:end] = (rotation * oldrot).as_quat()

    # store new position and orientation
    # pylint: disable=attribute-defined-outside-init
    target_object._orientation = R.from_quat(opath)
    target_object._position = ppath

    return target_object


class BaseRotate:
    """
    Inherit this class to provide rotation() methods.

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

    def rotate(self, rotation, anchor=None, start="auto"):
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

        increment: bool, default=False
            If `increment=False`, input rotations are absolute.
            If `increment=True`, input rotations are interpreted as increments of each other.

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
        # if Collection: apply to children
        clear = False
        if start == "clear":
            start = 0
            clear = True
        targets = []
        if getattr(self, "_object_type", None) == "Collection" and not getattr(
            self, "_freeze_children", False
        ):
            if anchor is None:
                anchor = self._position[-1]
            targets.extend(self.children)
        # if BaseGeo apply to self
        if getattr(self, "position", None) is not None:
            targets.append(self)
        for obj in targets:
            if clear:
                obj._orientation = R.from_quat([[0, 0, 0, 1]] * len(self._position))
            apply_rotation(obj, rotation, anchor, start)
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

        increment: bool, default=False
            If `increment=False`, input rotations are absolute.
            If `increment=True`, input rotations are interpreted as increments of each other.

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

        increment: bool, default=False
            If `increment=False`, input rotations are absolute.
            If `increment=True`, input rotations are interpreted as increments of each other.

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

        increment: bool, default=False
            If `increment=False`, input rotations are absolute.
            If `increment=True`, input rotations are interpreted as increments of each other.

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

        increment: bool, default=False
            If `increment=False`, input rotations are absolute.
            If `increment=True`, input rotations are interpreted as increments of each other.

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

        increment: bool, default=False
            If `increment=False`, input rotations are absolute.
            If `increment=True`, input rotations are interpreted as increments of each other.

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

        increment: bool, default=False
            If `increment=False`, input rotations are absolute.
            If `increment=True`, input rotations are interpreted as increments of each other.

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


def apply_move(target_object, displacement, start="auto", absolute=False):
    """
    Implementation of the move() functionality.

    target_object: object with position and orientation attributes
    displacement: displacement vector/path, array_like, shape (3,) or (n,3).
        If the input is scalar (shape (3,)) the operation is applied to the
        whole path. If the input is a vector (shape (n,3)), it is
        appended/merged with the existing path.
    start: int, str, default='auto'
        start=i applies an operation starting at the i'th path index.
        With start='auto' and scalar input the wole path is moved. With
        start='auto' and vector input the input is appended.
    absolute: bool, default=False
        If absolute=False then transformations are applied on to existing
        positions/orientations. If absolute=True position/orientation are
        set to input values.
    """
    # pylint: disable=protected-access
    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=too-many-branches

    # check input types
    if Config.checkinputs:
        check_vector_type(displacement, "displacement")
        check_start_type(start)
        check_absolute_type(absolute)

    # displacement vector -> ndarray
    inpath = np.array(displacement, dtype=float)
    scalar_input = inpath.ndim == 1

    # check input format
    if Config.checkinputs:
        check_path_format(inpath, "displacement")

    # load old path
    ppath = target_object._position
    opath = target_object._orientation.as_quat()

    # path lengths
    lenop = len(ppath)
    lenip = 1 if scalar_input else len(inpath)

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

    # vector: if start+inpath extends beyond oldpath: pad behind and merge
    if start + lenip > lenop + pad_before:
        pad_behind = start + lenip - (lenop + pad_before)

    # avoid execution when there is no padding (cost~100ns)
    if pad_before + pad_behind:
        ppath = np.pad(ppath, ((pad_before, pad_behind), (0, 0)), "edge")
        opath = np.pad(opath, ((pad_before, pad_behind), (0, 0)), "edge")
        target_object._orientation = R.from_quat(opath)

    # set end-index
    if scalar_input:
        end = len(ppath)
    else:
        end = start + lenip

    # apply move operation
    if absolute:
        ppath[start:end] = inpath
    else:
        ppath[start:end] += inpath
    target_object._position = ppath

    return target_object


class BaseMove:
    """
    Inherit this class to provide move() methods.

    The apply_move function is applied to all target objects:
    - For Magpylib objects that inherit BaseMove and BaseGeo (e.g. Cuboid()),
      apply_move() is applied only to the object itself.
    - Collections inherit BaseGeo and have children with BaseGeo. In this case
    apply_move() is applied to the object itself, but also to the children.
    """

    def move(self, displacement, start="auto", absolute=False):
        """
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

        increment: bool, default=False
            If `increment=False`, input displacements are absolute.
            If `increment=True`, input displacements are interpreted as increments of each other.
            For example, an incremental input displacement of `[(2,0,0), (2,0,0), (2,0,0)]`
            corresponds to an absolute input displacement of `[(2,0,0), (4,0,0), (6,0,0)]`.

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

        clear = False
        if start == "clear":
            start = 0
            clear = True
        # if Collection: apply to children
        targets = []
        if getattr(self, "_object_type", None) == "Collection" and not getattr(
            self, "_freeze_children", False
        ):
            # pylint: disable=no-member
            targets.extend(self.children)
        # if BaseGeo apply to self
        if getattr(self, "position", None) is not None:
            targets.append(self)
        for obj in targets:
            if clear:
                # pylint: disable=no-member
                obj._position -= self._position
                obj._position = obj._position[-1:]
                obj._orientation = obj._orientation[: len(obj._position)]
            apply_move(obj, displacement, start, absolute)
        return self


class BaseTransform(BaseRotate, BaseMove):
    """Base transformation class holding the move and rotate methods"""

"""BaseTransform class code"""
# pylint: disable=too-many-instance-attributes
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


def apply_rotation(target_object, rotation, anchor=None, start=-1, increment=False):
    """
    Implementation of the rotate() functionality.

    target_object: object with position and orientation attributes
    rotation: scipy rotation object
    anchor: rotation anchor
    start: start
    increment: increment
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
    old_ppath = target_object._position
    old_opath = target_object._orientation.as_quat()

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
    # pylint: disable=attribute-defined-outside-init
    target_object.orientation = R.from_quat(new_opath)
    target_object.position = new_ppath

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

    def rotate(self, rotation, anchor=None, start=-1, increment=False):
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

        # if Collection: apply to children
        if getattr(self, "_object_type", None) == "Collection":
            for obj in self.objects:
                apply_rotation(obj, rotation, anchor, start, increment)
            return self

        # if BaseGeo apply to self
        return apply_rotation(self, rotation, anchor, start, increment)


    def rotate_from_angax(
        self, angle, axis, anchor=None, start=-1, increment=False, degrees=True):
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

        return self.rotate(rot, anchor, start, increment)


    def rotate_from_rotvec(
        self, rotvec, anchor=None, start=-1, increment=False, degrees=False):
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
        return self.rotate(rot, anchor=anchor, start=start, increment=increment)


    def rotate_from_euler(
        self, seq, angles, anchor=None, start=-1, increment=False, degrees=False):
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
        return self.rotate(rot, anchor=anchor, start=start, increment=increment)


    def rotate_from_matrix(self, matrix, anchor=None, start=-1, increment=False):
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
        return self.rotate(rot, anchor=anchor, start=start, increment=increment)


    def rotate_from_mrp(self, mrp, anchor=None, start=-1, increment=False):
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
        return self.rotate(rot, anchor=anchor, start=start, increment=increment)


    def rotate_from_quat(self, quat, anchor=None, start=-1, increment=False):
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
        return self.rotate(rot, anchor=anchor, start=start, increment=increment)


# pylint: disable=too-many-instance-attributes
import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib._src.default_classes import default_settings as Config
from magpylib._src.input_checks import (
    check_vector_type,
    check_path_format,
    check_start_type,
    check_increment_type)
from magpylib._src.utility import adjust_start


def apply_move(target_object, displacement, start=-1, increment=False):
    """
    Implementation of the move() functionality.

    target_object: object with position and orientation attributes
    displacement: displacement vector/path
    start: start
    increment: increment
    """
    # pylint: disable=protected-access
    # pylint: disable=attribute-defined-outside-init

    # check input types
    if Config.checkinputs:
        check_vector_type(displacement, "displacement")
        check_start_type(start)
        check_increment_type(increment)

    # displacement vector -> ndarray
    inpath = np.array(displacement, dtype=float)

    # check input format
    if Config.checkinputs:
        check_path_format(inpath, "displacement")

    # expand if input is shape (3,)
    if inpath.ndim == 1:
        inpath = np.expand_dims(inpath, 0)

    # load old path
    old_ppath = target_object._position
    old_opath = target_object._orientation.as_quat()
    lenop = len(old_ppath)
    lenin = len(inpath)

    # change start to positive values in [0, lenop]
    start = adjust_start(start, lenop)

    # incremental input -> absolute input
    if increment:
        for i, d in enumerate(inpath[:-1]):
            inpath[i + 1] = inpath[i + 1] + d

    end = start + lenin  # end position of new_path

    til = end - lenop
    if til > 0:  # case inpos extends beyond old_path -> tile up old_path
        old_ppath = np.pad(old_ppath, ((0, til), (0, 0)), "edge")
        old_opath = np.pad(old_opath, ((0, til), (0, 0)), "edge")
        target_object._orientation = R.from_quat(old_opath)

    # add new_ppath to old_ppath
    old_ppath[start:end] += inpath
    target_object._position = old_ppath

    return target_object


class BaseMove:
    """
    Inherit this class to provide move() methods.

    The apply_move function is applied to all target objects:
    - For Magpylib objects that inherit BaseRotate and BaseGeo (e.g. Cuboid()),
      apply_move() is applied only to the object itself.
    - Collections inherit only BaseMove. In this case apply_move() is only
      applied to the Collection children.
    - Compounds are user-defined classes that inherit Collection but also inherit
      BaseGeo. In this case apply_move() is applied to the object itself, but also
      to its children.
    """

    def move(self, displacement, start=-1, increment=False):
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

        # if Collection: apply to children
        if getattr(self, "_object_type", None) == "Collection":
            for obj in self.objects:
                apply_move(obj, displacement, start, increment)
            return self

        # if BaseGeo apply to self
        return apply_move(self, displacement, start, increment)

class BaseTransform(BaseRotate, BaseMove):
    """Base transformation class holding the move and rotate methods"""
"""BaseGeo class code"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib._src.obj_classes.class_Collection import Collection
from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib._src.default_classes import default_settings as Config
from magpylib._src.input_checks import (
    check_vector_type,
    check_path_format,
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

# ALL METHODS ON INTERFACE
class BaseGeo:
    """Initializes position and rotation (=orientation) properties
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

    def __init__(self, position, orientation, style=None, **kwargs):
        # set pos and orient attributes
        self.position = position
        self.orientation = orientation
        self.style_class = self._get_style_class()
        if style is not None or kwargs:
            if style is None:
                style = {}
            style_kwargs = {}
            for k, v in kwargs.items():
                if k.startswith("style_"):
                    style_kwargs[k[6:]] = v
                else:
                    raise TypeError(
                        f"__init__() got an unexpected keyword argument {k!r}"
                    )
            style.update(**style_kwargs)
            self.style = style

    def _get_style_class(self):
        """returns style class based on object type. If class has no attribute `_object_type` or is
        not found in `MAGPYLIB_FAMILIES` returns `BaseStyle` class."""
        # pylint: disable=import-outside-toplevel
        from magpylib._src.style import get_style_class

        return get_style_class(self)

    # properties ----------------------------------------------------
    @property
    def position(self):
        """Object position attribute getter and setter."""
        return np.squeeze(self._position)

    @position.setter
    def position(self, pos):
        """Set object position-path.

        position: array_like, shape (3,) or (N,3)
            Position-path of object.
        """

        # check input type
        if Config.checkinputs:
            check_vector_type(pos, "position")

        # path vector -> ndarray
        pos = np.array(pos, dtype=float)

        # check input format
        if Config.checkinputs:
            check_path_format(pos, "position")

        # expand if input is shape (3,)
        if pos.ndim == 1:
            pos = np.expand_dims(pos, 0)
        self._position = pos

    @property
    def orientation(self):
        """Object orientation attribute getter and setter."""
        # cannot squeeze (its a Rotation object)
        if len(self._orientation) == 1:  # single path orientation - reduce dimension
            return self._orientation[0]
        return self._orientation  # return full path

    @orientation.setter
    def orientation(self, rot):
        """Set object orientation-path.

        rot: None or scipy Rotation, shape (1,) or (N,), default=None
            Set orientation-path of object. None generates a unit orientation
            for every path step.
        """
        # check input type
        if Config.checkinputs:
            check_rot_type(rot)

        # None input generates unit rotation
        if rot is None:
            self._orientation = R.from_quat([(0, 0, 0, 1)] * len(self._position))

        # expand rot.as_quat() to shape (1,4)
        else:
            val = rot.as_quat()
            if val.ndim == 1:
                self._orientation = R.from_quat([val])
            else:
                self._orientation = rot

    @property
    def style(self):
        """instance of MagpyStyle for display styling options"""
        if not hasattr(self, "_style") or self._style is None:
            self._style = self._validate_style(val=None)
        return self._style

    @style.setter
    def style(self, val):
        self._style = self._validate_style(val)

    def _validate_style(self, val=None):
        if val is None:
            val = {}
        if isinstance(val, dict):
            val = self.style_class(**val)
        if not isinstance(val, self.style_class):
            raise ValueError(f"style must be of type {self.style_class}")
        return val

    # dunders -------------------------------------------------------
    def __add__(self, source):
        """
        Add up sources to a Collection object.

        Returns
        -------
        Collection: Collection
        """
        return Collection(self, source)

    # methods -------------------------------------------------------
    def reset_path(self):
        """
        Reset object path to position = (0,0,0) and orientation = unit rotation.

        Returns
        -------
        self: Magpylib object

        Examples
        --------
        Create an object with non-zero path

        >>> import magpylib as magpy
        >>> obj = magpy.Sensor(position=(1,2,3))
        >>> print(obj.position)
        [1. 2. 3.]
        >>> obj.reset_path()
        >>> print(obj.position)
        [0. 0. 0.]

        """
        self.position = (0, 0, 0)
        self.orientation = R.from_quat((0, 0, 0, 1))

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
        old_ppath = self._position
        old_opath = self._orientation.as_quat()
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
            self.orientation = R.from_quat(old_opath)

        # add new_ppath to old_ppath
        old_ppath[start:end] += inpath
        self.position = old_ppath

        return self

    def rotate(self, rotation, anchor=None, start=-1, increment=False):
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
        old_ppath = self._position
        old_opath = self._orientation.as_quat()

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
        self.orientation = R.from_quat(new_opath)
        self.position = new_ppath

        return self

    def rotate_from_angax(
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
        With the ``rotate_from_angax`` method Magpylib objects can be rotated about their local
        coordinte system center:

        >>> import magpylib as magpy
        >>> sensor = magpy.Sensor()
        >>> print(sensor.position)
        [0. 0. 0.]
        >>> print(sensor.orientation.as_euler('xyz'))
        [0. 0. 0.]
        >>> sensor.rotate_from_angax(angle=45, axis='x')
        >>> print(sensor.position)
        [0. 0. 0.]
        >>> print(sensor.orientation.as_euler('xyz', degrees=True))
        [45.  0.  0.]

        With the ``anchor`` keyword the object rotates about a designated axis
        that passes through the given anchor point:

        >>> import magpylib as magpy
        >>> sensor = magpy.Sensor()
        >>> sensor.rotate_from_angax(angle=90, axis=(1,0,0), anchor=(0,1,0))
        >>> print(sensor.position)
        [ 0.  1. -1.]
        >>> print(sensor.orientation.as_euler('xyz', degrees=True))
        [90.  0.  0.]

        The method can also be used to generate paths, making use of scipy.Rotation
        object vector input:

        >>> import magpylib as magpy
        >>> sensor = magpy.Sensor()
        >>> sensor.rotate_from_angax(angle=90, axis='x', anchor=(0,1,0), start='append')
        >>> print(sensor.position)
        [[ 0.  0.  0.]
         [ 0.  1. -1.]]
        >>> print(sensor.orientation.as_euler('xyz', degrees=True))
        [[ 0.  0.  0.]
         [90.  0.  0.]]
        >>> sensor.rotate_from_angax(angle=[10,20,30], axis='x', anchor=(0,1,0), start='append')
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
        >>> sensor.rotate_from_angax([10]*3, 'x', (0,1,0), start=1, increment=True)
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
        >>> sensor.rotate_from_angax(angle=[5]*4, axis='z', anchor=0, start=0, increment=True)
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
        self.rotate(rot, anchor, start, increment)

        return self


def adjust_start(start, lenop):
    """
    change start to a value inside of [0,lenop], i.e. inside of the
    old path.
    """
    if start == "append":
        start = lenop
    elif start < 0:
        start += lenop

    # fix out-of-bounds start values
    if start < 0:
        start = 0
        if Config.checkinputs:
            print("Warning: start out of path bounds. Setting start=0.")
    elif start > lenop:
        start = lenop
        if Config.checkinputs:
            print(f"Warning: start out of path bounds. Setting start={lenop}.")

    return start

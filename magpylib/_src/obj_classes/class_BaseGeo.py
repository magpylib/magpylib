"""BaseGeo class code"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib._src.obj_classes.class_Collection import Collection
from magpylib._src.obj_classes.class_BaseRotation import BaseRotation
from magpylib._src.default_classes import default_settings as Config
from magpylib._src.input_checks import (
    check_vector_type,
    check_path_format,
    check_start_type,
    check_increment_type,
    check_rot_type,
)
from magpylib._src.utility import adjust_start

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
    position: array_like, shape (N,3)
        Position path

    orientation: scipy.Rotation, shape (N,)
        Rotation path

    Methods
    -------

    - display
    - move
    - rotate

    """

    def __init__(self, position, orientation, style=None, **kwargs):
        # set pos and orient attributes
        self.position = position
        self.orientation = orientation
        self._rotate = BaseRotation(parent_class=self)

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
    def rotate(self):
        """Rotation class for magpylib objects"""
        return self._rotate

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

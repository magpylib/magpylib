"""BaseGeo class code"""

# pylint: disable=cyclic-import
import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib._src.obj_classes.class_BaseTransform import BaseTransform
from magpylib._src.default_classes import default_settings as Config
from magpylib._src.input_checks import (
    check_vector_type,
    check_path_format,
    check_rot_type)


class BaseGeo(BaseTransform):
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

    def __init__(self, position=(0.,0.,0.,), orientation=None, style=None, **kwargs):

        # set pos and orient attributes
        self._position = np.array([[0., 0., 0.]])
        self._orientation = R.from_quat([[0., 0., 0., 1.]])
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
        try:
            self.move(pos, start='clear')
        except AttributeError:
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
            orient = R.from_quat([(0, 0, 0, 1)] * len(self._position))

        # expand rot.as_quat() to shape (1,4)
        else:
            val = rot.as_quat()
            if val.ndim == 1:
                orient = R.from_quat([val])
            else:
                orient = rot
        try:
            self.rotate(orient, start='clear')
        except AttributeError:
            self._orientation = orient

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
        # pylint: disable=import-outside-toplevel
        from magpylib._src.obj_classes.class_Collection import Collection
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

        # if Collection: apply to children
        targets = []
        if getattr(self, "_object_type", None) == "Collection":
            # pylint: disable=no-member
            targets.extend(self.children)
        # if BaseGeo apply to self
        if getattr(self, "position", None) is not None:
            targets.append(self)
        for obj in targets:
            # pylint: disable=protected-access
            obj._position = np.array([[0., 0., 0.]])
            obj._orientation = R.from_quat([[0., 0., 0., 1.]])

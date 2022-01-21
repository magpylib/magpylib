"""BaseGeo class code"""

# pylint: disable=cyclic-import
# pylint: disable=too-many-instance-attributes
# pylint: disable=protected-access

import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib._src.obj_classes.class_BaseTransform import BaseTransform
from magpylib._src.defaults.defaults_classes import default_settings as Config
from magpylib._src.input_checks import (
    check_vector_type,
    check_path_format,
    check_rot_type)

def position_input_check(pos):
    """
    checks input type and format end returns an ndarray of shape (N,3).
    This function is used for setter and init only -> (1,3) and (3,) input
    creates same behavior.
    """
    # check input type
    if Config.checkinputs:
        check_vector_type(pos, "position")
    # path vector -> ndarray
    pos_array = np.array(pos, dtype=float)
    # check input format
    if Config.checkinputs:
        check_path_format(pos_array, "position")
    # tile to format (N,3) and return
    return pos_array.reshape(-1,3)

def orientation_input_check(ori):
    """
    checks input type and format end returns an ndarray of shape (N,4).
    This function is used for setter and init only -> (1,4) and (4,) input
    creates same behavior.
    """
    # check input type
    if Config.checkinputs:
        check_rot_type(ori)
    # None input generates unit rotation
    ori_array = np.array([(0, 0, 0, 1)]) if ori is None else ori.as_quat()
    # tile to format (N,4) and return
    return ori_array.reshape(-1,4)

class BaseGeo(BaseTransform):
    """Initializes position and orientation properties
    of an object in a global CS.

    position is a ndarray with shape (3,).

    orientation is a scipy.spatial.transformation.Rotation
    object that gives the relative rotation to the init_state. The
    init_state is defined by how the fields are implemented (e.g.
    cyl upright in xy-plane)

    Both attributes _position and _orientation.as_rotvec() are of shape (N,3),
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

        # set _position and _orientation attributes
        self._init_position_orientation(position, orientation)

        # style
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

    def _init_position_orientation(self, position, orientation):
        """
        tile up position and orientation input at Class init and set attributes
        _position and _orientation.
        pos: position input
        ori: orientation.as_quat() input
        """

        # format position and orientation inputs
        pos = position_input_check(position)
        ori = orientation_input_check(orientation)

        # padding logic: if one is longer than the other, edge-pad up the other
        len_pos = pos.shape[0]
        len_ori = ori.shape[0]

        if len_pos>len_ori:
            ori = np.pad(ori, ((0,len_pos-len_ori), (0,0)), 'edge')
        elif len_pos<len_ori:
            pos = np.pad(pos, ((0,len_ori-len_pos), (0,0)), 'edge')

        # set attributes
        self._position = pos
        self._orientation = R.from_quat(ori)

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
        """
        Set object position-path.

        Use edge-padding and end-slicing to adjust orientation path

        When a Collection position is set, then all children retain their
        relative position to the Collection BaseGeo.

        position: array_like, shape (3,) or (N,3)
            Position-path of object.
        """
        # check and set new position
        self._position = position_input_check(pos)

        # pad/slice orientation path to same length
        delta_path = len(self._position) - len(self._orientation)
        if delta_path>0:
            padding = ((0,delta_path), (0,0))
            ori_pad = np.pad(self._orientation.as_quat(), padding, 'edge')
            self._orientation = R.from_quat(ori_pad)
        elif delta_path<0:
            self._orientation = self._orientation[-delta_path:]

        # TODO for child in getattr(self, "children", []):
        #    relative_child_pos = child._position - self.position[-1]
        #    child._position = pos + relative_child_pos

        # set _position attribute with ndim=2 format


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

        # None input generates unit rotation, else tile to shape (N,)
        if rot is None:
            orient = R.from_quat([(0, 0, 0, 1)]*len(self._position))
        else:
            val = rot.as_quat()
            orient = R.from_quat([val]) if val.ndim ==1 else rot

        # set _orientation attribute with ndim=2 format
        self._orientation = orient

        # MISSING: apply position to match orientation format


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
    def __add__(self, obj):
        """
        Add up sources to a Collection object.

        Returns
        -------
        Collection: Collection
        """
        # pylint: disable=import-outside-toplevel
        from magpylib._src.obj_classes.class_Collection import Collection
        return Collection(self, obj)

    # methods -------------------------------------------------------
    def reset_path(self):
        """
        Reset object and children paths to position = (0,0,0) and
        orientation = unit rotation.

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
        self._position = np.array([[0., 0., 0.]])
        self._orientation = R.from_quat([[0., 0., 0., 1.]])

        # if Collection: apply to children
        #for child in getattr(self, "children", []):
        #    child._position = np.array([[0., 0., 0.]])
        #    child._orientation = R.from_quat([[0., 0., 0., 1.]])

        # targets = []
        # if getattr(self, "_object_type", None) == "Collection":
        #     # pylint: disable=no-member
        #     targets.extend(self.children)
        # # if BaseGeo apply to self
        # if getattr(self, "position", None) is not None:
        #     targets.append(self)
        # for obj in targets:
        #     # pylint: disable=protected-access
        #     obj._position = np.array([[0., 0., 0.]])
        #     obj._orientation = R.from_quat([[0., 0., 0., 1.]])

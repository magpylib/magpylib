"""BaseGeo class code"""
# pylint: disable=cyclic-import
# pylint: disable=too-many-instance-attributes
# pylint: disable=protected-access
import numpy as np
from scipy.spatial.transform import Rotation as R

from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib._src.input_checks import check_format_input_orientation
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.obj_classes.class_BaseTransform import BaseTransform
from magpylib._src.style import BaseStyle
from magpylib._src.utility import add_iteration_suffix


def pad_slice_path(path1, path2):
    """edge-pads or end-slices path 2 to fit path 1 format
    path1: shape (N,x)
    path2: shape (M,x)
    return: path2 with format (N,x)
    """
    delta_path = len(path1) - len(path2)
    if delta_path > 0:
        return np.pad(path2, ((0, delta_path), (0, 0)), "edge")
    if delta_path < 0:
        return path2[-delta_path:]
    return path2


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

    - show
    - move
    - rotate

    """

    _style_class = BaseStyle

    def __init__(
        self,
        position=(
            0.0,
            0.0,
            0.0,
        ),
        orientation=None,
        style=None,
        **kwargs,
    ):
        self._style_kwargs = {}
        self._parent = None
        # set _position and _orientation attributes
        self._init_position_orientation(position, orientation)

        if style is not None or kwargs:  # avoid style creation cost if not needed
            self._style_kwargs = self._process_style_kwargs(style=style, **kwargs)

    @staticmethod
    def _process_style_kwargs(style=None, **kwargs):
        if kwargs:
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
        return style

    def _init_position_orientation(self, position, orientation):
        """tile up position and orientation input and set _position and
        _orientation at class init. Because position and orientation inputs
        come at the same time, tiling is slightly different then with setters.
        pos: position input
        ori: orientation input
        """

        # format position and orientation inputs
        pos = check_format_input_vector(
            position,
            dims=(1, 2),
            shape_m1=3,
            sig_name="position",
            sig_type="array_like (list, tuple, ndarray) with shape (3,) or (n,3)",
            reshape=(-1, 3),
        )
        oriQ = check_format_input_orientation(orientation, init_format=True)

        # padding logic: if one is longer than the other, edge-pad up the other
        len_pos = pos.shape[0]
        len_ori = oriQ.shape[0]

        if len_pos > len_ori:
            oriQ = np.pad(oriQ, ((0, len_pos - len_ori), (0, 0)), "edge")
        elif len_pos < len_ori:
            pos = np.pad(pos, ((0, len_ori - len_pos), (0, 0)), "edge")

        # set attributes
        self._position = pos
        self._orientation = R.from_quat(oriQ)

    # properties ----------------------------------------------------
    @property
    def parent(self):
        """The object is a child of it's parent collection."""
        return self._parent

    @parent.setter
    def parent(self, inp):
        # pylint: disable=import-outside-toplevel
        from magpylib._src.obj_classes.class_Collection import Collection

        if isinstance(inp, Collection):
            inp.add(self, override_parent=True)
        elif inp is None:
            if self._parent is not None:
                self._parent.remove(self)
            self._parent = None
        else:
            raise MagpylibBadUserInput(
                "Input `parent` must be `None` or a `Collection` object."
                f"Instead received {type(inp)}."
            )

    @property
    def position(self):
        """
        Object position(s) in the global coordinates in units of mm. For m>1, the
        `position` and `orientation` attributes together represent an object path.
        """
        return np.squeeze(self._position)

    @position.setter
    def position(self, inp):
        """
        Set object position-path.

        Use edge-padding and end-slicing to adjust orientation path

        When a Collection position is set, then all children retain their
        relative position to the Collection BaseGeo.

        position: array_like, shape (3,) or (N,3)
            Position-path of object.
        """
        old_pos = self._position

        # check and set new position
        self._position = check_format_input_vector(
            inp,
            dims=(1, 2),
            shape_m1=3,
            sig_name="position",
            sig_type="array_like (list, tuple, ndarray) with shape (3,) or (n,3)",
            reshape=(-1, 3),
        )

        # pad/slice and set orientation path to same length
        oriQ = self._orientation.as_quat()
        self._orientation = R.from_quat(pad_slice_path(self._position, oriQ))

        # when there are children include their relative position
        for child in getattr(self, "children", []):
            old_pos = pad_slice_path(self._position, old_pos)
            child_pos = pad_slice_path(self._position, child._position)
            rel_child_pos = child_pos - old_pos
            # set child position (pad/slice orientation)
            child.position = self._position + rel_child_pos

    @property
    def orientation(self):
        """
        Object orientation(s) in the global coordinates. `None` corresponds to
        a unit-rotation. For m>1, the `position` and `orientation` attributes
        together represent an object path.
        """
        # cannot squeeze (its a Rotation object)
        if len(self._orientation) == 1:  # single path orientation - reduce dimension
            return self._orientation[0]
        return self._orientation  # return full path

    @orientation.setter
    def orientation(self, inp):
        """Set object orientation-path.

        inp: None or scipy Rotation, shape (1,) or (N,)
            Set orientation-path of object. None generates a unit orientation
            for every path step.
        """
        old_oriQ = self._orientation.as_quat()

        # set _orientation attribute with ndim=2 format
        oriQ = check_format_input_orientation(inp, init_format=True)
        self._orientation = R.from_quat(oriQ)

        # pad/slice position path to same length
        self._position = pad_slice_path(oriQ, self._position)

        # when there are children they rotate about self.position
        # after the old Collection orientation is rotated away.
        for child in getattr(self, "children", []):
            # pad/slice and set child path
            child.position = pad_slice_path(self._position, child._position)
            # compute rotation and apply
            old_ori_pad = R.from_quat(np.squeeze(pad_slice_path(oriQ, old_oriQ)))
            child.rotate(
                self.orientation * old_ori_pad.inv(), anchor=self._position, start=0
            )

    @property
    def style(self):
        """
        Object style in the form of a BaseStyle object. Input must be
        in the form of a style dictionary.
        """
        if getattr(self, "_style", None) is None:
            self._style = self._style_class()
        if self._style_kwargs:
            style_kwargs = self._style_kwargs.copy()
            self._style_kwargs = {}
            try:
                self._style.update(style_kwargs)
            except (AttributeError, ValueError) as e:
                e.args = (
                    f"{self!r} has been initialized with some invalid style arguments.\n"
                    + str(e),
                )
                raise
        return self._style

    @style.setter
    def style(self, val):
        self._style = self._validate_style(val)

    def _validate_style(self, val=None):
        val = {} if val is None else val
        style = self.style  # triggers style creation
        if isinstance(val, dict):
            style.update(val)
        elif not isinstance(val, self._style_class):
            raise ValueError(
                f"Input parameter `style` must be of type {self._style_class}.\n"
                f"Instead received type {type(val)}"
            )
        return style

    # dunders -------------------------------------------------------
    def __add__(self, obj):
        """Add up sources to a Collection object.

        Returns
        -------
        Collection: Collection
        """
        # pylint: disable=import-outside-toplevel
        from magpylib import Collection

        return Collection(self, obj)

    # methods -------------------------------------------------------
    def reset_path(self):
        """Set object position to (0,0,0) and orientation = unit rotation.

        Returns
        -------
        self: magpylib object

        Examples
        --------
        Demonstration of `reset_path` functionality:

        >>> import magpylib as magpy
        >>> obj = magpy.Sensor(position=(1,2,3))
        >>> obj.rotate_from_angax(45, 'z')
        Sensor...
        >>> print(obj.position)
        [1. 2. 3.]
        >>> print(obj.orientation.as_euler('xyz', degrees=True))
        [ 0.  0. 45.]

        >>> obj.reset_path()
        Sensor(id=...)
        >>> print(obj.position)
        [0. 0. 0.]
        >>> print(obj.orientation.as_euler('xyz', degrees=True))
        [0. 0. 0.]
        """
        self.position = (0, 0, 0)
        self.orientation = None
        return self

    def copy(self, **kwargs):
        """Returns a copy of the current object instance. The `copy` method returns a deep copy of
        the object, that is independent of the original object.

        Parameters
        ----------
        kwargs: dict
            Keyword arguments (for example `position=(1,2,3)`) are applied to the copy.

        Examples
        --------
        Create a `Sensor` object and copy to an another position:

        >>> import magpylib as magpy
        >>> sens1 = magpy.Sensor(style_label='sens1')
        >>> sens2 = sens1.copy(position=(2,6,10), style_label='sens2')
        >>> print(f"Instance {sens1.style.label} with position {sens1.position}.")
        Instance sens1 with position [0. 0. 0.].
        >>> print(f"Instance {sens2.style.label} with position {sens2.position}.")
        Instance sens2 with position [ 2.  6. 10.].
        """
        # pylint: disable=import-outside-toplevel
        from copy import deepcopy

        # avoid deepcopying the deep dependency upwards the tree structure
        if self.parent is not None:
            # using private attributes to avoid triggering `.add` method (see #530 bug)
            parent = self._parent
            self._parent = None
            obj_copy = deepcopy(self)
            self._parent = parent
        else:
            obj_copy = deepcopy(self)

        if getattr(self, "_style", None) is not None or bool(
            getattr(self, "_style_kwargs", False)
        ):
            # pylint: disable=no-member
            label = self.style.label
            if label is None:
                label = f"{type(self).__name__}_01"
            else:
                label = add_iteration_suffix(label)
            obj_copy.style.label = label
        style_kwargs = {}
        for k, v in kwargs.items():
            if k.startswith("style"):
                style_kwargs[k] = v
            else:
                setattr(obj_copy, k, v)
        if style_kwargs:
            style_kwargs = self._process_style_kwargs(**style_kwargs)
            obj_copy.style.update(style_kwargs)
        return obj_copy

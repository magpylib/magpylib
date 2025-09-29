"""Base class containing geometric properties and manipulation."""

# pylint: disable=cyclic-import
# pylint: disable=too-many-instance-attributes
# pylint: disable=protected-access
# pylint: disable=import-outside-toplevel

from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.transform import Rotation as R

from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib._src.input_checks import (
    check_format_input_orientation,
    check_format_input_vector,
)
from magpylib._src.obj_classes.class_BaseTransform import BaseTransform
from magpylib._src.style import BaseStyle
from magpylib._src.utility import add_iteration_suffix


def _pad_slice_path(path1, path2):
    """edge-pads or end-slices path 2 to fit path 1 format
    path1: shape (N, x)
    path2: shape (M, x)
    return: path2 with format (N,x)
    """
    delta_path = len(path1) - len(path2)
    if delta_path > 0:
        return np.pad(path2, ((0, delta_path), (0, 0)), "edge")
    if delta_path < 0:
        return path2[-delta_path:]
    return path2


class BaseGeo(BaseTransform, ABC):
    """Initializes basic properties inherited by ALL Magpylib objects

    Inherited from BaseTransform
    ----------------------------
    - move()
    - rotate()

    Properties of BaseGeo
    ---------------------
    - parent
    - position
    - orientation
    - volume
    - centroid
    - moment
    - style

    Methods of BaseGeo
    ------------------
    - __add__()
    - reset_path()
    - copy()

    Note
    ----
    position is a ndarray with shape (3,).

    orientation is a scipy.spatial.transformation.Rotation
    object that gives the relative rotation to the init_state. The
    init_state is defined by how the fields are implemented (e.g.
    cyl upright in xy-plane)

    Both attributes _position and _orientation.as_rotvec() are of shape (N, 3),
    and describe a path of length N. (N=1 if there is only one
    object position).
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

    # static methods ------------------------------------------------
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
                    msg = f"__init__() got an unexpected keyword argument {k!r}"
                    raise TypeError(msg)
            style.update(**style_kwargs)
        return style

    # private helper methods ----------------------------------------
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
            sig_type="array-like (list, tuple, ndarray) with shape (3,) or (n, 3)",
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

    def _validate_style(self, val=None):
        val = {} if val is None else val
        style = self.style  # triggers style creation
        if isinstance(val, dict):
            style.update(val)
        elif not isinstance(val, self._style_class):
            msg = (
                f"Input style must be an instance of {self._style_class.__name__}; "
                f"instead received type {type(val).__name__}."
            )
            raise ValueError(msg)
        return style

    # abstract methods that must be implemented by subclasses ------

    @abstractmethod
    def _get_centroid(self, squeeze=True):
        """Calculate and return the centroid of the object in units of (m).

        This method must be implemented by all subclasses.

        Returns
        -------
        numpy.ndarray, shape (n, 3) when there is a path, or squeeze(1, 3) when not
            Centroid coordinates [(x, y, z), ...] in (m).
        """

    # properties ----------------------------------------------------
    @property
    def parent(self):
        """Parent collection of the object."""
        return self._parent

    @parent.setter
    def parent(self, parent):
        """Set parent collection.

        Parameters
        ----------
        parent : Collection or None
            New parent collection. Use ``None`` to detach.
        """
        from magpylib._src.obj_classes.class_Collection import Collection  # noqa: I001, PLC0415

        if isinstance(parent, Collection):
            parent.add(self, override_parent=True)
        elif parent is None:
            if self._parent is not None:
                self._parent.remove(self)
            self._parent = None
        else:
            msg = (
                "Input parent must be None or a Collection instance; "
                f"instead received type {type(parent).__name__}."
            )
            raise MagpylibBadUserInput(msg)

    @property
    def position(self):
        """Return object position in global coordinates (m)."""
        return np.squeeze(self._position)

    @position.setter
    def position(self, position):
        """Set object position.

        Edge-padding or end-slicing is applied to keep orientation path length consistent.
        Child positions are updated to preserve relative offsets when part of a collection.

        Parameters
        ----------
        position : array-like, shape (3,) or (n, 3)
            New position(s) in units (m).
        """
        old_pos = self._position

        # check and set new position
        self._position = check_format_input_vector(
            position,
            dims=(1, 2),
            shape_m1=3,
            sig_name="position",
            sig_type="array-like (list, tuple, ndarray) with shape (3,) or (n, 3)",
            reshape=(-1, 3),
        )

        # pad/slice and set orientation path to same length
        oriQ = self._orientation.as_quat()
        self._orientation = R.from_quat(_pad_slice_path(self._position, oriQ))

        # when there are children include their relative position
        for child in getattr(self, "children", []):
            old_pos = _pad_slice_path(self._position, old_pos)
            child_pos = _pad_slice_path(self._position, child._position)
            rel_child_pos = child_pos - old_pos
            # set child position (pad/slice orientation)
            child.position = self._position + rel_child_pos

    @property
    def orientation(self):
        """Return object orientation.

        ``None`` corresponds to unit rotation.
        """
        # cannot squeeze (its a Rotation object)
        if len(self._orientation) == 1:  # single path orientation - reduce dimension
            return self._orientation[0]
        return self._orientation  # return full path

    @orientation.setter
    def orientation(self, orientation):
        """Set object orientation.

        Parameters
        ----------
        orientation : Rotation | None
            New orientation as ``scipy.spatial.transform.Rotation``. ``None`` generates a unit
            rotation for every path step.
        """
        old_oriQ = self._orientation.as_quat()

        # set _orientation attribute with ndim=2 format
        oriQ = check_format_input_orientation(orientation, init_format=True)
        self._orientation = R.from_quat(oriQ)

        # pad/slice position path to same length
        self._position = _pad_slice_path(oriQ, self._position)

        # when there are children they rotate about self.position
        # after the old Collection orientation is rotated away.
        for child in getattr(self, "children", []):
            # pad/slice and set child path
            child.position = _pad_slice_path(self._position, child._position)
            # compute rotation and apply
            old_ori_pad = R.from_quat(np.squeeze(_pad_slice_path(oriQ, old_oriQ)))
            child.rotate(
                self.orientation * old_ori_pad.inv(), anchor=self._position, start=0
            )

    @property
    def centroid(self):
        """Return centroid (m)."""
        return self._get_centroid()

    @property
    def _centroid(self):
        """Return centroid without squeezing (internal)."""
        return self._get_centroid(squeeze=False)

    @property
    def style(self):
        """Return object style as a ``BaseStyle`` instance."""
        if getattr(self, "_style", None) is None:
            self._style = self._style_class()
        if self._style_kwargs:
            style_kwargs = self._style_kwargs.copy()
            self._style_kwargs = {}
            try:
                self._style.update(style_kwargs)
            except (AttributeError, ValueError) as e:
                e.args = (
                    f"{self!r} has been initialized with some invalid style arguments."
                    + str(e),
                )
                raise
        return self._style

    @style.setter
    def style(self, style):
        """Set object style.

        Parameters
        ----------
        style : dict | BaseStyle
            Style specification. Dict keys are mapped onto style attributes.
        """
        self._style = self._validate_style(style)

    # public methods ------------------------------------------------
    def reset_path(self):
        """Reset path: set position to (0, 0, 0) and orientation to unit rotation.

        Returns
        -------
        Self
            Self (for chaining).

        Examples
        --------
        >>> import magpylib as magpy
        >>> obj = magpy.Sensor(position=(1, 2, 3))
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
        """Return deep copy with optional modifications.

        Parameters
        ----------
        **kwargs
            Attribute overrides applied to the copy (e.g. ``position=(1, 2, 3)``).

        Returns
        -------
        Self
            Deep-copied object (same concrete subclass as original).

        Examples
        --------
        Create a ``Sensor`` object and copy to an another position:

        >>> import magpylib as magpy
        >>> sens1 = magpy.Sensor(style_label='sens1')
        >>> sens2 = sens1.copy(position=(2, 6, 10), style_label='sens2')
        >>> print(f'Instance {sens1.style.label} with position {sens1.position}.')
        Instance sens1 with position [0. 0. 0.].
        >>> print(f'Instance {sens2.style.label} with position {sens2.position}.')
        Instance sens2 with position [ 2.  6. 10.].
        """
        # pylint: disable=import-outside-toplevel
        from copy import deepcopy  # noqa: PLC0415

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

    # dunders -------------------------------------------------------
    def __add__(self, obj):
        """Return ``Collection`` containing ``self`` and ``obj``.

        Parameters
        ----------
        obj : Sensor | Source
            Other operand.

        Returns
        -------
        Collection
            New collection with both operands.
        """
        # pylint: disable=import-outside-toplevel
        from magpylib import Collection  # noqa: PLC0415

        return Collection(self, obj)

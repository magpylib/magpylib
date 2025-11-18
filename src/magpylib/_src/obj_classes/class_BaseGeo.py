"""Base class containing geometric properties and manipulation."""

# pylint: disable=cyclic-import
# pylint: disable=too-many-instance-attributes
# pylint: disable=protected-access
# pylint: disable=import-outside-toplevel

import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager

import numpy as np
from scipy.spatial.transform import Rotation as R

from magpylib._src.display.display import show
from magpylib._src.display.traces_core import make_DefaultTrace
from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib._src.input_checks import (
    check_format_input_numeric,
    check_format_input_orientation,
)
from magpylib._src.obj_classes.class_BaseTransform import (
    BaseTransform,
    pad_path_properties,
)
from magpylib._src.style import BaseStyle
from magpylib._src.utility import add_iteration_suffix

UNITS = {
    "parent": None,
    "path_properties": None,
    "current": "A",
    "polarization": "T",
    "magnetization": "A/m",
    "position": "m",
    "orientation": "deg",
    "dimension": "m",
    "diameter": "m",
    "vertices": "m",
    "moment": "A·m²",
    "volume": "m³",
    "current_densities": "A/m",
}


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
    - show()
    - describe()

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
    _path_properties = ("position", "orientation")
    _path_sync_enabled = True
    show = show
    get_trace = make_DefaultTrace

    def __init__(
        self,
        position=(0.0, 0.0, 0.0),
        orientation=None,
        *,
        style=None,
        **kwargs,
    ):
        self._style_kwargs = {}
        self._parent = None

        # set path properties while holding sync of path lengths
        path_kwargs = {k: v for k, v in kwargs.items() if k in self._path_properties}
        kwargs = {k: v for k, v in kwargs.items() if k not in path_kwargs}

        # Initialize position and orientation if not provided in kwargs
        if "position" not in path_kwargs:
            path_kwargs["position"] = position
        if "orientation" not in path_kwargs:
            path_kwargs["orientation"] = orientation

        with self.hold_path_sync(sync_on_exit=True):
            for prop, val in path_kwargs.items():
                setattr(self, prop, val)

        if style is not None or kwargs:  # avoid style creation cost if not needed
            self._style_kwargs = self._process_style_kwargs(style=style, **kwargs)

    # path logic methods --------------------------------------------
    def __init_subclass__(cls):
        """Automatically aggregate '_path_properties' from parent classes when subclassing."""
        super().__init_subclass__()
        parent_attr = []
        for base in cls.__mro__[1:]:
            if hasattr(base, "_path_properties"):
                parent_attr.extend(base._path_properties)
                break  # only take first (nearest) base's attribute
        if "_path_properties" in cls.__dict__:
            cls._path_properties = tuple(
                dict.fromkeys([*parent_attr, *cls._path_properties])
            )
        else:
            cls._path_properties = tuple(parent_attr)

    def __setattr__(self, name, value):
        """Intercept public property assignments to trigger automatic path sync."""
        super().__setattr__(name, value)

        # Only sync for public path properties (not private _attributes)
        # This avoids recursion during internal operations
        if (
            not name.startswith("_")
            and name in self._path_properties
            and self._path_sync_enabled
        ):
            # Get the actual stored value (from private attribute) to determine target length
            private_name = f"_{name}"
            prop_value = getattr(self, private_name, None)
            if prop_value is not None:
                self._sync_all_paths(prop_value)

    def _sync_pos_orient_recursive(self, target_len):
        """
        Pad position, using public attribute which triggers orientation
        and recursive children padding.
        """
        if self._position is None:
            return
        n_path_new, n_path = target_len, len(self._position)
        if n_path_new < n_path:
            self.position = self._position[-n_path_new:]
        elif n_path_new > n_path:
            self.position = np.pad(
                self._position, ((0, n_path_new - n_path), (0, 0)), "edge"
            )

    def _sync_all_paths(self, prop=None, start=0, propagate=True, path_properties=None):
        """Synchronize all path properties to the same length."""
        if not self._path_sync_enabled:
            return

        path_properties = path_properties or self._path_properties
        lengths = [
            len(arr)
            for name in path_properties
            if (arr := getattr(self, f"_{name}", None)) is not None
        ]
        if not lengths:
            return

        target_len = max(lengths) if prop is None else len(prop)

        # Temporarily disable sync to avoid recursion during padding
        with self.hold_path_sync(sync_on_exit=False):
            # Pad private attributes of all path properties
            pad_path_properties(
                self, target_len, start=start, path_properties=path_properties
            )
            # Sync children positions for collections
            if propagate:
                self._sync_pos_orient_recursive(target_len)

    @contextmanager
    def hold_path_sync(self, *, sync_on_exit=True):
        """
        Temporarily disable auto-sync inside the block.
        If auto_sync=True (default), performs one global sync on exit.
        """
        old_state = self._path_sync_enabled
        self._path_sync_enabled = False
        try:
            yield
        finally:
            self._path_sync_enabled = old_state
            if sync_on_exit and old_state:
                self._sync_all_paths()

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
    def path_properties(self):
        """Tuple of property names that support paths.

        Path properties are object attributes that can store multiple values to describe
        a path through space. For example, ``position`` and ``orientation`` are path
        properties that can be arrays of shape (n, 3) and Rotation objects of length n,
        respectively, representing n positions along a path.

        Returns
        -------
        tuple of str
            Names of properties that can have path-like behavior. Common path properties
            include ``position``, ``orientation``, and for some objects ``dimension``,
            ``polarization``, ``magnetization``, etc.

        Examples
        --------
        >>> import magpylib as magpy
        >>> sensor = magpy.Sensor()
        >>> sensor.path_properties
        ('position', 'orientation')

        >>> cuboid = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
        >>> cuboid.path_properties
        ('position', 'orientation', 'dimension', 'polarization')
        """
        return self._path_properties

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
        return np.squeeze(self._position) if self._position is not None else None

    @position.setter
    def position(self, position):
        """Set object position.

        Path syncing is applied to keep all path properties consistent.
        Child positions are updated to preserve relative offsets when part of a collection.

        Parameters
        ----------
        position : array-like, shape (3,) or (n, 3)
            New position(s) in units (m).
        """
        old_pos = getattr(self, "_position", None)

        # check and set new position
        self._position = check_format_input_numeric(
            position,
            dtype=float,
            shapes=((3,), (None, 3)),
            name="position",
            reshape=(-1, 3),
        )

        # sync all paths (including orientation)
        self._sync_all_paths(self._position, propagate=False)

        # when there are children include their relative position
        if old_pos is not None:
            for child in getattr(self, "children", []):
                old_pos_padded = _pad_slice_path(self._position, old_pos)
                child_pos = _pad_slice_path(self._position, child._position)
                rel_child_pos = child_pos - old_pos_padded
                # set child position (syncs all child paths)
                child.position = self._position + rel_child_pos

    @property
    def orientation(self):
        """Return object orientation.

        ``None`` corresponds to unit rotation.
        """
        if self._orientation is None:
            return None
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
        old_ori = getattr(self, "_orientation", None)
        old_oriQ = old_ori.as_quat() if old_ori is not None else None

        # set _orientation attribute with ndim=2 format
        oriQ = check_format_input_orientation(orientation, init_format=True)
        self._orientation = R.from_quat(oriQ)

        # sync all paths (including position)
        self._sync_all_paths(oriQ, propagate=False)

        # when there are children they rotate about self.position
        # after the old Collection orientation is rotated away.
        if old_oriQ is not None:
            for child in getattr(self, "children", []):
                # pad/slice and set child path
                child.position = _pad_slice_path(oriQ, child._position)
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

    @centroid.setter
    def centroid(self, _input):
        """Throw error when trying to set centroid."""
        msg = "Cannot set property centroid. It is read-only."
        raise AttributeError(msg)

    @_centroid.setter
    def _centroid(self, _input):
        """Throw error when trying to set centroid."""
        msg = "Cannot set property _centroid. It is read-only."
        raise AttributeError(msg)

    @property
    def barycenter(self):
        """Return barycenter (m).

        .. deprecated:: 6.0.0
            Use :attr:`centroid` instead. The ``barycenter`` property will be removed
            in a future version.
        """
        warnings.warn(
            "The 'barycenter' property is deprecated and will be removed in a future "
            "version. Use 'centroid' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.centroid

    @property
    def _barycenter(self):
        """Return barycenter without squeezing (internal, deprecated)."""
        return self._centroid

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

    # display methods -----------------------------------------------
    def _property_names_generator(self):
        """returns a generator with class properties only"""
        return (
            attr
            for attr in dir(self)
            if isinstance(getattr(type(self), attr, None), property)
        )

    def _get_description(self, exclude=None, precision=3):
        """Return list of lines describing the object properties.

        Parameters
        ----------
        exclude : None | str | Sequence[str], default ('style',)
            Property names to omit from the description.
        precision : int, default 3
            Number of decimal places for floating point representation.

        Returns
        -------
        list of str
            One line per entry ready to be joined with newlines.
        """
        with np.printoptions(precision=precision, suppress=True, linewidth=200):
            if exclude is None:
                exclude = ()
            exclude = (
                ("barycenter", *exclude)
                if isinstance(exclude, (list, tuple))
                else ("barycenter", exclude)
            )
            params = list(self._property_names_generator())
            lines = [f"{self!r}"]
            lines.append(f"  • path length: {self._position.shape[0]}")
            for key in list(dict.fromkeys([*UNITS, *self.path_properties, *params])):
                k = key
                if not k.startswith("_") and k in params and k not in exclude:
                    unit = UNITS.get(k)
                    unit_str = f" {unit}" if unit else ""
                    val = ""
                    if k == "path_properties":
                        k, val = "path properties", ""
                    elif k in self.path_properties:
                        val = getattr(self, f"_{k}", None)
                        if isinstance(val, R):
                            val = val.as_rotvec(degrees=True)  # pylint: disable=no-member
                        if isinstance(val, np.ndarray):
                            axis = None if val.ndim <= 1 else 0
                            if len(val) == 1 or np.unique(val, axis=axis).shape[0] == 1:
                                val = val[0]
                            elif len(val) > 1:
                                k = f"{k} (last)"
                                val = val[-1]
                            if len(val.flatten()) > 20:
                                val = f"shape{val.shape}"
                    elif k == "pixel":
                        val = getattr(self, "pixel", None)
                        if isinstance(val, np.ndarray):
                            px_shape = val.shape[:-1]
                            val_str = f"{int(np.prod(px_shape))}"
                            if val.ndim > 2:
                                val_str += f" ({'x'.join(str(p) for p in px_shape)})"
                            val = val_str
                    elif k == "status_disconnected_data":
                        val = getattr(self, k)
                        if val is not None:
                            val = f"{len(val)} part{'s'[: len(val) ^ 1]}"
                    elif isinstance(getattr(self, k), list | tuple | np.ndarray):
                        val = np.array(getattr(self, k))
                        if len(val.flatten()) > 20:
                            val = f"shape{val.shape}"
                    else:
                        val = getattr(self, k)
                    val = str(val).replace("\n", " ")
                    indent = " " * 2 if key in self.path_properties else ""
                    lines.append(f"{indent}  • {k}: {val}{unit_str}")
            return lines

    def describe(self, *, exclude=("style", "field_func"), return_string=False):
        """Return or print a formatted description of object properties.

        Parameters
        ----------
        exclude : str | Sequence[str], default ('style', 'field_func')
            Property names to omit from the description.
        return_string : bool, default False
            If ``True`` return the description string; if ``False`` print it and
            return ``None``.

        Returns
        -------
        str | None
            Description string if ``return_string=True`` else ``None``.
        """
        lines = self._get_description(exclude=exclude)
        output = "\n".join(lines)

        if return_string:
            return output

        print(output)  # noqa: T201
        return None

    def _repr_html_(self):
        """Rich HTML representation for notebooks and other frontends."""
        lines = self._get_description(exclude=("style", "field_func"))
        return f"""<pre>{"<br>".join(lines)}</pre>"""

    def __repr__(self) -> str:
        """Return concise string representation for terminals and logs."""
        name = getattr(self, "name", None)
        if name is None:
            style = getattr(self, "style", None)
            name = getattr(style, "label", None)
        name_str = "" if name is None else f", label={name!r}"
        return f"{type(self).__name__}(id={id(self)!r}{name_str})"

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

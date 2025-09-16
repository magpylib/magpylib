# pylint: disable=too-many-positional-arguments

"""Polyline current class code"""

import warnings
from typing import ClassVar

import numpy as np

from magpylib._src.display.traces_core import make_Polyline
from magpylib._src.exceptions import MagpylibDeprecationWarning
from magpylib._src.fields.field_BH_polyline import current_vertices_field
from magpylib._src.input_checks import check_format_input_vertices
from magpylib._src.obj_classes.class_BaseExcitations import _BaseCurrent
from magpylib._src.obj_classes.class_BaseProperties import _BaseDipoleMoment
from magpylib._src.obj_classes.class_BaseTarget import _BaseTarget
from magpylib._src.obj_classes.target_meshing import _target_mesh_polyline
from magpylib._src.utility import unit_prefix


class Polyline(_BaseCurrent, _BaseTarget, _BaseDipoleMoment):
    """Line current flowing in straight paths from vertex to vertex.

    Can be used as ``sources`` input for magnetic field computation and ``target``
    input for force computation.

    The vertex positions are defined in the local object coordinates (rotate with
    object). When ``position=(0, 0, 0)`` and ``orientation=None`` global and local
    coordinates coincide.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    position : array-like, shape (3,) or (m, 3), default (0, 0, 0)
        Object position(s) in global coordinates in units (m). ``position`` and
        ``orientation`` attributes define the object path.
    orientation : None or Rotation, default None
        Object orientation(s) in global coordinates as a scipy Rotation. Rotation can
        have length 1 or m. ``None`` generates a unit-rotation.
    vertices : None or array-like, shape (n, 3), default None
        Current flows along the vertices in units (m) in the local object coordinates. At
        least two vertices must be given.
    current : None or float, default None
        Electrical current (A).
    meshing : None or int, default None
        Mesh fineness for force computation. Must be a positive integer at least the
        number of segments. Each segment gets one mesh point at its center. All
        remaining mesh points are distributed evenly along the polyline.
    style : None or dict, default None
        Style dictionary. Can also be provided via style underscore magic, e.g.
        ``style_color='red'``.

    Attributes
    ----------
    position : ndarray, shape (3,) or (m, 3)
        Same as constructor parameter ``position``.
    orientation : Rotation
        Same as constructor parameter ``orientation``.
    vertices : None or float
        Same as constructor parameter ``vertices``.
    current : None or float
        Same as constructor parameter ``current``.
    meshing : None or int
        Same as constructor parameter ``meshing``.
    centroid : ndarray, shape (3,) or (m, 3)
        Read-only. Object centroid computed via mean of vertices in units (m)
        in global coordinates. Can be a path.
    dipole_moment : ndarray, shape (3,)
        Read-only. Object dipole moment (A·m²) in local object coordinates. Can
        only be computed for a closed loop.
    parent : Collection or None
        Parent collection of the object.
    style : dict
        Style dictionary defining visual properties.

    Examples
    --------
    ``Polyline`` objects are magnetic field sources. In this example we compute the
    H-field (A/m) of a square-shaped line current with 1 A at the observer position
    (1, 1, 1) cm:

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> src = magpy.current.Polyline(
    ...     current=1,
    ...     vertices=((.01,0,0), (0,.01,0), (-.01,0,0), (0,-.01,0), (.01,0,0)),
    ... )
    >>> H = src.getH((.01,.01,.01))
    >>> with np.printoptions(precision=3):
    ...     print(H)
    [3.161 3.161 0.767]
    """

    # pylint: disable=dangerous-default-value
    _field_func = staticmethod(current_vertices_field)
    _force_type = "current"
    _field_func_kwargs_ndim: ClassVar[dict[str, int]] = {
        "current": 1,
        "vertices": 3,
        "segment_start": 2,
        "segment_end": 2,
    }
    get_trace = make_Polyline

    def __init__(
        self,
        current=None,
        vertices=None,
        position=(0, 0, 0),
        orientation=None,
        meshing=None,
        style=None,
        **kwargs,
    ):
        # instance attributes
        self.vertices = vertices

        # init inheritance
        super().__init__(position, orientation, current, style, **kwargs)

        # Initialize _BaseTarget
        _BaseTarget.__init__(self, meshing)

    # Properties
    @property
    def vertices(self):
        """
        The current flows along the vertices which are given in units of m in the
        local object coordinates (move/rotate with object). At least two vertices
        must be given.
        """
        return self._vertices

    @vertices.setter
    def vertices(self, vert):
        """Set polyline vertices.

        Parameters
        ----------
        vert : None or array-like, shape (n, 3)
            Vertex list (m) in local object coordinates. At least two vertices
            must be given.
        """
        self._vertices = check_format_input_vertices(vert)

    @property
    def _default_style_description(self):
        """Default style description text"""
        if self.vertices is None:
            return "no vertices"
        return f"{unit_prefix(self.current)}A" if self.current else "no current"

    # Methods
    def _get_centroid(self, squeeze=True):
        """Centroid of object in units of m."""
        if squeeze:
            if self.vertices is not None:
                return np.mean(self.vertices, axis=0) + self.position
            return self.position
        if self.vertices is not None:
            return np.mean(self.vertices, axis=0) + self._position
        return self._position

    def _get_dipole_moment(self):
        """Magnetic moment of object in units Am²."""
        # test init
        if self.vertices is None or self.current is None:
            return np.array((0.0, 0.0, 0.0))
        # test for closed polyline
        if (len(self.vertices) > 1) and (np.all(self.vertices[0] == self.vertices[-1])):
            return (
                self.current
                / 2
                * np.sum(np.cross(self.vertices[:-1], self.vertices[1:]), axis=0)
            )
        msg = (
            f"Cannot compute dipole moment of {self}. Dipole moment is only defined for closed "
            "Polylines (first and last vertex must be identical)."
        )
        raise ValueError(msg)

    def _generate_mesh(self):
        """Generate mesh for force computation."""
        # Tests in getFT ensure that meshing, dimension and excitation are set

        # Special special case: fewer points than segments, cannot be caught in
        #    meshing setter because vertices might not have been set yet
        n_segments = len(self.vertices) - 1
        if self.meshing < n_segments:
            msg = (
                "getFT Polyline bad meshing input. number of points is less than"
                " number of Polyline segments. Setting one point per segment in computation"
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            n_target = n_segments
        else:
            n_target = self.meshing

        return _target_mesh_polyline(self.vertices, self.current, n_target)


class Line(Polyline):
    """Line is deprecated, see Polyline"""

    # pylint: disable=method-hidden
    @staticmethod
    def _field_func(*args, **kwargs):
        """Catch Deprecation warning in getBH_dict"""
        _deprecation_warn()
        return current_vertices_field(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        _deprecation_warn()
        super().__init__(*args, **kwargs)


def _deprecation_warn():
    warnings.warn(
        (
            "Line is deprecated and will be removed in a future version, "
            "use Polyline instead."
        ),
        MagpylibDeprecationWarning,
        stacklevel=2,
    )

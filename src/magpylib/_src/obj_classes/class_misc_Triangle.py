"""Magnet Triangle class"""

# pylint: disable=too-many-positional-arguments

from typing import ClassVar

import numpy as np

from magpylib._src.display.traces_core import make_Triangle
from magpylib._src.fields.field_BH_triangle import _BHJM_triangle
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet
from magpylib._src.style import TriangleStyle


class Triangle(BaseMagnet):
    """Triangular surface with homogeneous magnetic surface charge.

    Can be used as ``sources`` input for magnetic field computation and ``target``
    input for force computation.

    When ``position=(0, 0, 0)`` and ``orientation=None`` the local object coordinates
    of the Triangle vertices coincide with the global coordinate system.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    position : array-like, shape (3,) or (p, 3), default (0, 0, 0)
        Object position(s) in global coordinates in units (m). ``position`` and
        ``orientation`` attributes define the object path.
    orientation : None | Rotation, default None
        Object orientation(s) in global coordinates as a scipy Rotation. Rotation can
        have length 1 or p. ``None`` generates a unit-rotation.
    vertices : None | array-like, shape (3, 3), default None
        Triangle vertices in the local object coordinates in units (m).
    polarization : None | array-like, shape (3,), default None
        Magnetic polarization vector J = mu0*M in units (T), given in the
        local object coordinates. Sets also ``magnetization``.
    magnetization : None | array-like, shape (3,), default None
        Magnetization vector M = J/mu0 in units (A/m), given in the local
        object coordinates. Sets also ``polarization``.
    style : None | dict, default None
        Style dictionary. Can also be provided via style underscore magic, e.g.
        ``style_color='red'``.

    Attributes
    ----------
    position : ndarray, shape (3,) or (p, 3)
        Same as constructor parameter ``position``.
    orientation : Rotation
        Same as constructor parameter ``orientation``.
    vertices : None | ndarray, shape (3, 3)
        Same as constructor parameter ``vertices``.
    polarization : None | ndarray, shape (3,)
        Same as constructor parameter ``polarization``.
    magnetization : None | ndarray, shape (3,)
        Same as constructor parameter ``magnetization``.
    centroid : ndarray, shape (3,) or (p, 3)
        Read-only. Object centroid in units (m) in global coordinates.
        Can be a path.
    parent : Collection | None
        Parent collection of the object.
    style : dict
        Style dictionary defining visual properties.

    Notes
    -----
    Returns (0, 0, 0) on corners.

    Examples
    --------
    ``Triangle`` objects are magnetic field sources. Below we compute the H-field in
    (A/m) of a Triangle object with magnetic polarization ``(0.1, 0.2, 0.3)`` in units
    (T), defined by the vertices ``(0, 0, 0)``, ``(0.01, 0, 0)`` and ``(0, 0.01, 0)``
    (m) at the observer position ``(0.1, 0.1, 0.1)`` (m):

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> verts = [(0, 0, 0), (0.01, 0, 0), (0, 0.01, 0)]
    >>> src = magpy.misc.Triangle(polarization=(0.1, 0.2, 0.3), vertices=verts)
    >>> H = src.getH((0.1, 0.1, 0.1))
    >>> with np.printoptions(precision=3):
    ...     print(H)
    [18.889 18.889 19.546]
    """

    _field_func = staticmethod(_BHJM_triangle)
    _field_func_kwargs_ndim: ClassVar[dict[str, int]] = {
        "polarization": 2,
        "vertices": 2,
    }
    get_trace = make_Triangle
    _style_class = TriangleStyle

    def __init__(
        self,
        position=(0, 0, 0),
        orientation=None,
        vertices=None,
        polarization=None,
        magnetization=None,
        style=None,
        **kwargs,
    ):
        self.vertices = vertices

        # init inheritance
        super().__init__(
            position, orientation, magnetization, polarization, style, **kwargs
        )

    # Properties
    @property
    def vertices(self):
        """Triangle vertices in local object coordinates in units (m)."""
        return self._vertices

    @vertices.setter
    def vertices(self, val):
        """Set triangle vertices.

        Parameters
        ----------
        val : None | array-like, shape (3, 3)
            Triangle vertices in local object coordinates in units (m).
        """
        self._vertices = check_format_input_vector(
            val,
            dims=(2,),
            shape_m1=3,
            sig_name="Triangle.vertices",
            sig_type="array-like (list, tuple, ndarray) of shape (3, 3)",
            allow_None=True,
        )

    @property
    def _barycenter(self):
        """Object barycenter."""
        return self._get_barycenter(self._position, self._orientation, self._vertices)

    @property
    def barycenter(self):
        """Object barycenter in units (m) in global coordinates."""
        return np.squeeze(self._barycenter)

    @property
    def _default_style_description(self):
        """Default style description text"""
        if self.vertices is None:
            return "no vertices"
        return ""

    # Methods
    def _get_centroid(self, squeeze=True):
        """Centroid of object in units (m)."""
        if squeeze:
            return self.barycenter
        return self._barycenter

    # Static methods
    @staticmethod
    def _get_barycenter(position, orientation, vertices):
        """Returns the barycenter of the Triangle object."""
        centroid = (
            np.array([0.0, 0.0, 0.0]) if vertices is None else np.mean(vertices, axis=0)
        )
        return orientation.apply(centroid) + position

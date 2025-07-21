# pylint: disable=too-many-positional-arguments

"""Magnet Tetrahedron class code"""

from typing import ClassVar

import numpy as np

from magpylib._src.display.traces_core import make_Tetrahedron
from magpylib._src.fields.field_BH_tetrahedron import BHJM_magnet_tetrahedron
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet


class Tetrahedron(BaseMagnet):
    """Tetrahedron magnet with homogeneous magnetization.

    Can be used as `sources` input for magnetic field computation.

    When `position=(0,0,0)` and `orientation=None` the Tetrahedron vertices coordinates
    are the same as in the global coordinate system. The geometric center of the Tetrahedron
    is determined by its vertices and. It is not necessarily located in the origin an can
    be computed with the barycenter property.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    position: array_like, shape (3,) or (m,3)
        Object position(s) in the global coordinates in units of m. For m>1, the
        `position` and `orientation` attributes together represent an object path.
        When setting vertices, the initial position is set to the barycenter.

    orientation: scipy `Rotation` object with length 1 or m, default=`None`
        Object orientation(s) in the global coordinates. `None` corresponds to
        a unit-rotation. For m>1, the `position` and `orientation` attributes
        together represent an object path.

    vertices: ndarray, shape (4,3)
        Vertices [(x1,y1,z1), (x2,y2,z2), (x3,y3,z3), (x4,y4,z4)], in the relative
        coordinate system of the tetrahedron.

    polarization: array_like, shape (3,), default=`None`
        Magnetic polarization vector J = mu0*M in units of T,
        given in the local object coordinates (rotates with object).

    magnetization: array_like, shape (3,), default=`None`
        Magnetization vector M = J/mu0 in units of A/m,
        given in the local object coordinates (rotates with object).

    volume: float
        Object physical volume in units of m^3.

    parent: `Collection` object or `None`
        The object is a child of it's parent collection.

    style: dict
        Object style inputs must be in dictionary form, e.g. `{'color':'red'}` or
        using style underscore magic, e.g. `style_color='red'`.

    Attributes
    ----------
    barycenter: array_like, shape (3,)
        Read only property that returns the geometric barycenter (=center of mass)
        of the object.

    Returns
    -------
    magnet source: `Tetrahedron` object

    Examples
    --------
    `Tetrahedron` magnets are magnetic field sources. Below we compute the H-field in A/m of a
    tetrahedron magnet with polarization (0.1,0.2,0.3) in units of T dimensions defined
    through the vertices (0,0,0), (.01,0,0), (0,.01,0) and (0,0,.01) in units of m at the
    observer position (0.01,0.01,0.01) given in units of m:

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> verts = [(0,0,0), (.01,0,0), (0,.01,0), (0,0,.01)]
    >>> src = magpy.magnet.Tetrahedron(polarization=(.1,.2,.3), vertices=verts)
    >>> H = src.getH((.01,.01,.01))
    >>> with np.printoptions(precision=3):
    ...     print(H)
    [2070.898 1656.718 1242.539]
    """

    _field_func = staticmethod(BHJM_magnet_tetrahedron)
    _field_func_kwargs_ndim: ClassVar[dict[str, int]] = {
        "polarization": 1,
        "vertices": 3,
    }
    get_trace = make_Tetrahedron

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
        # instance attributes
        self.vertices = vertices

        # init inheritance
        super().__init__(
            position, orientation, magnetization, polarization, style, **kwargs
        )

    # property getters and setters
    @property
    def vertices(self):
        """Length of the Tetrahedron sides [a,b,c] in units of m."""
        return self._vertices

    @vertices.setter
    def vertices(self, dim):
        """Set Tetrahedron vertices (a,b,c), shape (3,), (meter)."""
        self._vertices = check_format_input_vector(
            dim,
            dims=(2,),
            shape_m1=3,
            length=4,
            sig_name="Tetrahedron.vertices",
            sig_type="array_like (list, tuple, ndarray) of shape (4,3)",
            allow_None=True,
        )

    @property
    def _barycenter(self):
        """Object barycenter."""
        return self._get_barycenter(self._position, self._orientation, self.vertices)

    @property
    def barycenter(self):
        """Object barycenter."""
        return np.squeeze(self._barycenter)

    @property
    def volume(self):
        """Volume of object in units of m³."""
        if self.vertices is None:
            return 0.0

        # Tetrahedron volume formula: |det(B-A, C-A, D-A)| / 6
        vertices = self.vertices
        v1 = vertices[1] - vertices[0]  # B - A
        v2 = vertices[2] - vertices[0]  # C - A
        v3 = vertices[3] - vertices[0]  # D - A

        # Create 3x3 matrix and compute determinant
        matrix = np.column_stack([v1, v2, v3])
        return abs(np.linalg.det(matrix)) / 6.0

    @staticmethod
    def _get_barycenter(position, orientation, vertices):
        """Returns the barycenter of a tetrahedron."""
        centroid = (
            np.array([0.0, 0.0, 0.0]) if vertices is None else np.mean(vertices, axis=0)
        )
        return orientation.apply(centroid) + position

    @property
    def _default_style_description(self):
        """Default style description text"""
        if self.vertices is None:
            return "no vertices"
        return ""

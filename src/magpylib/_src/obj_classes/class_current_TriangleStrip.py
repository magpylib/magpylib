# pylint: disable=too-many-positional-arguments

"""TriangleStrip current class code"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from magpylib._src.display.traces_core import make_TriangleStrip
from magpylib._src.fields.field_BH_current_sheet import BHJM_current_tristrip
from magpylib._src.input_checks import check_format_input_vertices
from magpylib._src.obj_classes.class_BaseExcitations import BaseCurrent
from magpylib._src.obj_classes.class_BaseTarget import BaseTarget
from magpylib._src.obj_classes.target_meshing import target_mesh_triangle_current
from magpylib._src.utility import unit_prefix


class TriangleStrip(BaseCurrent, BaseTarget):
    """Current flowing in straight lines along a Ribbon made of adjacent Triangles.

    Can be used as `sources` input for magnetic field computation and `target`
    input for force computation.

    The vertex positions are defined in the local object coordinates (rotate with object).
    When `position=(0,0,0)` and `orientation=None` global and local coordinates coincide.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    position: array_like, shape (3,) or (m,3), default=`(0,0,0)`
        Object position(s) in the global coordinates in units of m. For m>1, the
        `position` and `orientation` attributes together represent an object path.

    orientation: scipy `Rotation` object with length 1 or m, default=`None`
        Object orientation(s) in the global coordinates. `None` corresponds to
        a unit-rotation. For m>1, the `position` and `orientation` attributes
        together represent an object path.

    vertices: array_like, shape (n,3), default=`None`
        The current flows along a band which consists of Triangles {T1, T2, ...}
        defined by the vertices {V1, V2, V3, V4, ...} as T1=(V1,V2,V3),
        T2=(V2,V3,V4), ... The vertices are given in units of m in the local
        object coordinates (move/rotate with object). At least three vertices
        must be given, which define the first Triangle.

    current: float, default=`None`
        The total current flowing through the strip in units of A. It flows in the
        direction V1->V3 in the first triangle, V2->V4 in the second, ...

    meshing: int, default=`None`
        Parameter that defines the mesh finesse for force computation.
        Must be an integer >= number of triangles specifying the target mesh size.
        The mesh is generated via bisection along longest edges until target
        number is reached.

    meshing: int, default=`None`
        Parameter that defines the mesh fineness for force computation.
        Must be a positive integer specifying the target mesh size.

    volume: float
        Read-only. Object physical volume in units of m^3 - set to 0 for this class.

    centroid: np.ndarray, shape (3,) or (m,3)
        Read-only. Object centroid in units of m given by mean of vertices.
        m is the path length.

    dipole_moment: np.ndarray, shape (3,)
        Read-only. Object dipole moment in units of A*m² in the local object coordinates.

    parent: `Collection` object or `None`
        The object is a child of it's parent collection.

    style: dict
        Object style inputs must be in dictionary form, e.g. `{'color':'red'}` or
        using style underscore magic, e.g. `style_color='red'`.

    Returns
    -------
    current source: `TriangleStrip` object

    Examples
    --------
    `TriangleStrip` objects are magnetic field sources. In this example we compute the H-field in A/m
    of a square current sheet (two triangles) with 1 A current at the observer position (1,1,1) cm:

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> src = magpy.current.TriangleStrip(
    ...    current=1,
    ...    vertices=((0,0,0), (0,1,0), (1,0,0), (1,1,1)),
    ... )
    >>> H = src.getH((.01,.01,.01))
    >>> with np.printoptions(precision=3):
    ...     print(H)
    [ 0.026 -0.307 -0.371]

    Notes
    -----
    On the vertices the returned field is zero.
    """

    # pylint: disable=dangerous-default-value
    _field_func = staticmethod(BHJM_current_tristrip)
    _force_type = "current"
    _field_func_kwargs_ndim: ClassVar[dict[str, int]] = {
        "current": 1,
        "vertices": 3,
    }
    get_trace = make_TriangleStrip

    def __init__(
        self,
        current=None,
        vertices=None,
        position=(0, 0, 0),
        orientation=None,
        meshing=None,
        *,
        style=None,
        **kwargs,
    ):
        # instance attributes
        self.vertices = vertices

        # Inherit
        super().__init__(position, orientation, current, style, **kwargs)
        BaseTarget.__init__(self, meshing)

    # property getters and setters
    @property
    def vertices(self):
        """
        The current flows along the triangles ...
        """
        return self._vertices

    @vertices.setter
    def vertices(self, vert):
        """Set TriangleStrip vertices, array_like, meter."""
        self._vertices = check_format_input_vertices(vert, minlength=3)

    @property
    def _default_style_description(self):
        """Default style description text"""
        if self.vertices is None:
            return "no vertices"
        return f"{unit_prefix(self.current)}A" if self.current else "no current"

    # Methods
    def _get_volume(self):
        """Volume of object in units of m³."""
        return 0.0

    def _get_centroid(self, squeeze=True):
        """Centroid of object in units of m."""
        centr = np.mean(self.vertices, axis=0) + self._position
        if squeeze:
            return np.squeeze(centr)
        return centr

    def _get_dipole_moment(self):
        """Magnetic moment of object in units Am²."""
        # test init
        if self.vertices is None or self.current is None:
            return np.array((0.0, 0.0, 0.0))
        # test closed
        if not np.allclose(self.vertices[:2], self.vertices[-2:]):
            msg = (
                f"Cannot compute dipole moment of {self}. Dipole moment is only defined for closed "
                "CurrentStrips (first two and last two vertices must be identical)."
            )
            raise ValueError(msg)

        # number of triangles
        no_tris = len(self.vertices) - 2

        # create triangles
        trias = np.array([self.vertices[:-2], self.vertices[1:-1], self.vertices[2:]])
        trias = np.swapaxes(trias, 0, 1)

        centroids = np.array([(t[0] + t[1] + t[2]) / 3 for t in trias])
        areas = 0.5 * np.linalg.norm(
            np.cross(trias[:, 1] - trias[:, 0], trias[:, 2] - trias[:, 0]), axis=1
        )

        # create current density input
        v1 = trias[:, 1] - trias[:, 0]
        v2 = trias[:, 2] - trias[:, 0]
        v1v1 = np.sum(v1 * v1, axis=1)
        v2v2 = np.sum(v2 * v2, axis=1)
        v1v2 = np.sum(v1 * v2, axis=1)

        curr_densities = np.zeros((no_tris, 3), dtype=float)
        # catch two times the same vertex in one triangle, and set CD to zero there
        mask = (v2v2 != 0) * (v1v1 != 0)
        h = np.sqrt(v1v1[mask] - (v1v2[mask] ** 2 / v2v2[mask]))
        curr_densities[mask] = (
            v2[mask]
            / (
                np.sqrt(v2v2[mask]) * h / np.repeat(self.current, no_tris, axis=0)[mask]
            )[:, np.newaxis]
        )
        # moment of one triangle: A / 2 * cent x curr_density
        return np.sum(
            areas[:, np.newaxis] / 2 * np.cross(centroids, curr_densities), axis=0
        )

    def _generate_mesh(self):
        """Generate mesh for force computation."""
        triangles = np.array(
            [self.vertices[i : i + 3] for i in range(len(self.vertices) - 2)]
        )

        # suppress all triangles in computation that are ~1e-9 times
        # smaller than the total mesh surface
        mask_good = np.ones(len(triangles), dtype=bool)

        sideA = triangles[:, 1] - triangles[:, 0]
        sideB = triangles[:, 2] - triangles[:, 0]

        sideAA = np.sum(sideA * sideA, axis=1)
        sideBB = np.sum(sideB * sideB, axis=1)
        sideAB = np.sum(sideA * sideB, axis=1)
        area2 = sideAA * sideBB - sideAB * sideAB
        area2_rel = area2 / np.sum(area2)

        mask_good[area2_rel < 1e-19] = False

        triangles = triangles[mask_good]

        # compute current densities
        base_length = np.linalg.norm(sideB[mask_good], axis=1)
        height = np.sqrt(area2[mask_good]) / base_length
        cds = (
            sideB[mask_good]
            / base_length[:, np.newaxis]
            * self.current
            / height[:, np.newaxis]
        )

        return target_mesh_triangle_current(
            triangles,
            self.meshing,
            cds,
        )

    def _validate_meshing(self, value):
        """Makes only sense with at least n_mesh = n_faces."""
        if isinstance(value, int) and value >= len(self.vertices) - 2:
            pass
        else:
            msg = (
                "TriangleStrip meshing parameter must be an integer >= number of faces"
                f" for {self}. Instead got {value}."
            )
            raise ValueError(msg)

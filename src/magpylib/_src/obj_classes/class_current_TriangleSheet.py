# pylint: disable=too-many-positional-arguments

"""TriangleSheet current class code"""

from __future__ import annotations

from typing import ClassVar

from magpylib._src.display.traces_core import make_TriangleSheet
from magpylib._src.fields.field_BH_current_sheet import BHJM_current_trisheet
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.obj_classes.class_BaseExcitations import BaseSource
from magpylib._src.utility import unit_prefix


class TriangleSheet(BaseSource):
    """Surface current density flowing along triangular sheets.

    Can be used as `sources` input for magnetic field computation.

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
        A set of points in units of m in the local object coordinates from which the
        triangular faces are constructed by the additional `faces` input.

    faces: array_like, shape (n,3), default=`None`
        Indices of vertices. Each triplet represents one triangle of the mesh.

    current_densities: array_like, shape (n,3), default=`None`
        Electrical current densities flowing in the triangles in units of A/m.
        The effective current density is a projection of the given current density
        vector into the triangle plane.

    parent: `Collection` object or `None`
        The object is a child of it's parent collection.

    style: dict
        Object style inputs must be in dictionary form, e.g. `{'color':'red'}` or
        using style underscore magic, e.g. `style_color='red'`.

    Returns
    -------
    current source: `TriangleSheet` object

    Examples
    --------

    """

    # pylint: disable=dangerous-default-value
    _field_func = staticmethod(BHJM_current_trisheet)
    _field_func_kwargs_ndim: ClassVar[dict[str, int]] = {
        "current_densities": 3,
        "vertices": 3,
        "faces": 3,
    }
    get_trace = make_TriangleSheet

    def __init__(
        self,
        current_densities=None,
        vertices=None,
        faces=None,
        position=(0, 0, 0),
        orientation=None,
        style=None,
        **kwargs,
    ):
        # instance attributes
        self._current_densities, self._vertices, self._faces = self._input_check(
            current_densities, vertices, faces
        )

        # init inheritance
        super().__init__(position, orientation, style, **kwargs)

    # property getters and setters
    @property
    def vertices(self):
        """TriangleSheet Vertices"""
        return self._vertices

    @property
    def faces(self):
        """TriangleSheet Faces"""
        return self._faces

    @property
    def current_densities(self):
        """TriangleSheet CurrentDensities"""
        return self._current_densities

    @property
    def _default_style_description(self):
        """Default style description text"""
        if self.vertices is None:
            return "no vertices"
        return f"{unit_prefix(self.current)}A" if self.current else "no current"

    def _input_check(self, current_densities, vertices, faces):
        """check and format user inputs"""
        cd = check_format_input_vector(
            current_densities,
            dims=(2,),
            shape_m1=3,
            sig_name="TriangleSheet.current_densities",
            sig_type="`None` or array_like (list, tuple, ndarray) with shape (n,3)",
            allow_None=False,
        ).astype(float)
        verts = check_format_input_vector(
            vertices,
            dims=(2,),
            shape_m1=3,
            sig_name="TriangleSheet.vertices",
            sig_type="`None` or array_like (list, tuple, ndarray) with shape (n,3)",
            allow_None=False,
        ).astype(float)
        fac = check_format_input_vector(
            faces,
            dims=(2,),
            shape_m1=3,
            sig_name="TriangleSheet.faces",
            sig_type="`None` or array_like (list, tuple, ndarray) with shape (n,3)",
            allow_None=False,
        ).astype(int)

        if len(verts) < 3:
            msg = f"Parameter `vertices` of {self} must have at least 3 vertices."
            raise ValueError(msg)

        if len(faces) != len(cd):
            msg = f"Parameters `current_densities` and `faces` of {self} must have same length."
            raise ValueError(msg)

        try:
            verts[fac]
        except IndexError as e:
            msg = f"Some `faces` indices of {self} do not match with `vertices` array."
            raise IndexError(msg) from e

        return cd, verts, fac

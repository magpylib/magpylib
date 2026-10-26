"""
The `magpylib.graphics.model3d` sub-package provides functions that build
3D model traces of common shapes, to attach to Magpylib objects with
`obj.style.model3d.add_trace()`.
"""

__all__ = [
    "make_Arrow",
    "make_Cuboid",
    "make_CylinderSegment",
    "make_Ellipsoid",
    "make_Prism",
    "make_Pyramid",
    "make_Tetrahedron",
    "make_TriangularMesh",
]

from magpylib._src.display.traces_base import (
    make_Arrow,
    make_Cuboid,
    make_CylinderSegment,
    make_Ellipsoid,
    make_Prism,
    make_Pyramid,
    make_Tetrahedron,
    make_TriangularMesh,
)

"""
The `magpylib.display.plotly` sub-package provides useful functions for
convenient creation of 3D traces for commonly used objects in the
library.
"""

from __future__ import annotations

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

"""
The `magpylib.magnet` subpackage contains all magnet classes.
"""

from __future__ import annotations

__all__ = [
    "Cuboid",
    "Cylinder",
    "CylinderSegment",
    "Sphere",
    "Tetrahedron",
    "TriangularMesh",
]

from magpylib._src.obj_classes.class_magnet_Cuboid import Cuboid
from magpylib._src.obj_classes.class_magnet_Cylinder import Cylinder
from magpylib._src.obj_classes.class_magnet_CylinderSegment import CylinderSegment
from magpylib._src.obj_classes.class_magnet_Sphere import Sphere
from magpylib._src.obj_classes.class_magnet_Tetrahedron import Tetrahedron
from magpylib._src.obj_classes.class_magnet_TriangularMesh import TriangularMesh

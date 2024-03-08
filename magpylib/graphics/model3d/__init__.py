"""
The `magpylib.display.plotly` sub-package provides useful functions for
convenient creation of 3D traces for commonly used objects in the
library.
"""

__all__ = [
    "make_Arrow",
    "make_Ellipsoid",
    "make_Pyramid",
    "make_Cuboid",
    "make_CylinderSegment",
    "make_Prism",
    "make_Tetrahedron",
    "make_TriangularMesh",
]

from magpylib._src.display.traces_base import make_Arrow
from magpylib._src.display.traces_base import make_Cuboid
from magpylib._src.display.traces_base import make_CylinderSegment
from magpylib._src.display.traces_base import make_Ellipsoid
from magpylib._src.display.traces_base import make_Prism
from magpylib._src.display.traces_base import make_Pyramid
from magpylib._src.display.traces_base import make_Tetrahedron
from magpylib._src.display.traces_base import make_TriangularMesh

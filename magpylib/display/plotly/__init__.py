"""
The `magpylib.display.plotly` sub-package provides useful functions for
convenient creation of 3D traces for commonly used objects in the
library.
"""

__all__ = [
    "make_BaseArrow",
    "make_BaseEllipsoid",
    "make_BaseCone",
    "make_BaseCuboid",
    "make_BaseCylinderSegment",
    "make_BasePrism",
]

from magpylib._src.display.plotly.plotly_base_traces import (
    make_BaseArrow,
    make_BaseEllipsoid,
    make_BaseCone,
    make_BaseCuboid,
    make_BaseCylinderSegment,
    make_BasePrism,
)

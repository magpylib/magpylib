# pylint: disable=line-too-long

"""
plotly display utilities

This package regroups useful functions to easily create the 3D traces for commonly used objects in the library.

These objects are just simple representations as dictionaries and do not contain any other information than the ones need to create a 3D-model
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

"""
The `magpylib.current` subpackage contains all electric current classes.
"""

from __future__ import annotations

__all__ = ["Circle", "Line", "Loop", "Polyline", "TriangleStrip"]

from magpylib._src.obj_classes.class_current_Circle import Circle, Loop
from magpylib._src.obj_classes.class_current_Polyline import Line, Polyline
from magpylib._src.obj_classes.class_current_TriangleStrip import TriangleStrip

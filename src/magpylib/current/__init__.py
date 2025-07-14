"""
The `magpylib.current` subpackage contains all electric current classes.
"""

__all__ = ["Circle", "Line", "Loop", "Polyline", "TriangleSheet", "TriangleStrip"]

from magpylib._src.obj_classes.class_current_Circle import Circle, Loop
from magpylib._src.obj_classes.class_current_Polyline import Line, Polyline
from magpylib._src.obj_classes.class_current_TriangleSheet import TriangleSheet
from magpylib._src.obj_classes.class_current_TriangleStrip import TriangleStrip

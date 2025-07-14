"""
The `magpylib.current` subpackage contains all electric current classes.
"""

__all__ = ["Circle", "Line", "Loop", "Polyline"]

from magpylib._src.obj_classes.class_current_Circle import Circle, Loop
from magpylib._src.obj_classes.class_current_Polyline import Line, Polyline

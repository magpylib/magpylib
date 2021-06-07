"""
The current sub-package contains all electric current classes.

Circular(current, dim, pos, rot)
    Circular line current loop.

Line(current, vertices, pos, rot)
    Line current flowing from vertex to vertex.
"""

__all__ = ['Circular', 'Line']

from magpylib._lib.obj_classes import Circular, Line

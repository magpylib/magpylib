"""
The magnet sub-package contains all permanent magnet classes.

Currently implemented magnet classes are:

Box(mag, dim, pos, rot)
    Homogenously magnetized permanent magnet with Cuboid shape.

Cylinder(mag, dim, pos, rot)
    Homogenously magnetized permanent magnet with Cylinder shape.
"""

__all__ = ['Box', 'Cylinder', 'Sphere']

from magpylib._lib.obj_classes import Box, Cylinder, Sphere

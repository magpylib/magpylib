"""
This subpackage contains all magnet classes. Magnets are modeled with homogeneous
magnetization given in units of millitesla [mT] through mu0*M. See documentation
for details on field computations.
"""

__all__ = ['Box', 'Cylinder', 'Sphere']

from magpylib._lib.obj_classes import Box, Cylinder, Sphere

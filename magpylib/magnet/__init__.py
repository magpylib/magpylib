"""
This subpackage contains all magnet classes. Magnets are modeled with homogeneous
magnetization given in units of millitesla [mT] through mu0*M. See documentation
for details on field computations.
"""

__all__ = ['Cuboid', 'Cylinder', 'Sphere', 'CylinderSegment']

from magpylib._src.obj_classes import Cuboid, Cylinder, Sphere, CylinderSegment

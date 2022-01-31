"""
This sub-package gives direct access to our field implementations.
"""

__all__ = [
    'dipole_Bfield',
    'current_loop_Bfield',
    'current_line_Bfield',
    'magnet_sphere_Bfield',
    'magnet_cuboid_Bfield',
]

from magpylib._src.fields.field_BH_dipole import dipole_Bfield
from magpylib._src.fields.field_BH_loop import current_loop_Bfield
from magpylib._src.fields.field_BH_line import current_line_Bfield
from magpylib._src.fields.field_BH_sphere import magnet_sphere_Bfield
from magpylib._src.fields.field_BH_cuboid import magnet_cuboid_Bfield

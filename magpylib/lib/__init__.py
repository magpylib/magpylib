"""
This sub-package gives direct access to our field implementations.
"""

__all__ = ['current_loop_Bfield', 'magnet_sphere_Bfield', 'dipole_Bfield']

from magpylib._src.fields.field_BH_loop import current_loop_Bfield
from magpylib._src.fields.field_BH_sphere import magnet_sphere_Bfield
from magpylib._src.fields.field_BH_dipole import dipole_Bfield
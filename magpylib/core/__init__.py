"""
The core sub-package gives direct access to our field implementations.
"""

__all__ = [
    "dipole_field",
    "current_loop_field",
    "current_line_field",
    "magnet_sphere_field",
    "magnet_cuboid_field",
    "magnet_cylinder_field",
    "magnet_cylinder_segment_field",
]

from magpylib._src.fields.field_BH_dipole import dipole_field
from magpylib._src.fields.field_BH_loop import current_loop_field
from magpylib._src.fields.field_BH_line import current_line_field
from magpylib._src.fields.field_BH_sphere import magnet_sphere_field
from magpylib._src.fields.field_BH_cuboid import magnet_cuboid_field
from magpylib._src.fields.field_BH_cylinder import magnet_cylinder_field
from magpylib._src.fields.field_BH_cylinder_segment import magnet_cylinder_segment_field

"""
The core sub-package gives direct access to our field implementations.
"""

__all__ = [
    "dipole_field",
    "current_circle_field",
    "current_loop_field",
    "current_line_field",
    "current_polyline_field",
    "magnet_sphere_field",
    "magnet_cuboid_field",
    "magnet_cylinder_field",
    "magnet_cylinder_segment_field",
    "triangle_field",
    "magnet_tetrahedron_field",
]

from magpylib._src.fields.field_BH_dipole import dipole_field
from magpylib._src.fields.field_BH_circle import current_circle_field
from magpylib._src.fields.field_BH_circle import current_loop_field
from magpylib._src.fields.field_BH_polyline import current_line_field
from magpylib._src.fields.field_BH_polyline import current_polyline_field
from magpylib._src.fields.field_BH_sphere import magnet_sphere_field
from magpylib._src.fields.field_BH_cuboid import magnet_cuboid_field
from magpylib._src.fields.field_BH_cylinder import magnet_cylinder_field
from magpylib._src.fields.field_BH_cylinder_segment import magnet_cylinder_segment_field
from magpylib._src.fields.field_BH_triangle import triangle_field
from magpylib._src.fields.field_BH_tetrahedron import magnet_tetrahedron_field

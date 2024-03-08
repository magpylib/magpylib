"""
The core sub-package gives direct access to our field implementations.
"""

__all__ = [
    "magnet_cuboid_Bfield",
    "magnet_cylinder_axial_Bfield",
    "magnet_cylinder_diametral_Hfield",
    "magnet_cylinder_segment_Hfield",
    "magnet_sphere_Bfield",
    "current_circle_Hfield",
    "current_polyline_Hfield",
    "dipole_Hfield",
    "triangle_Bfield",
]

from magpylib._src.fields.field_BH_circle import current_circle_Hfield
from magpylib._src.fields.field_BH_cuboid import magnet_cuboid_Bfield
from magpylib._src.fields.field_BH_cylinder import magnet_cylinder_axial_Bfield
from magpylib._src.fields.field_BH_cylinder import magnet_cylinder_diametral_Hfield
from magpylib._src.fields.field_BH_cylinder_segment import (
    magnet_cylinder_segment_Hfield,
)
from magpylib._src.fields.field_BH_dipole import dipole_Hfield
from magpylib._src.fields.field_BH_polyline import current_polyline_Hfield
from magpylib._src.fields.field_BH_sphere import magnet_sphere_Bfield
from magpylib._src.fields.field_BH_triangle import triangle_Bfield

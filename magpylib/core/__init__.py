"""
The core sub-package gives direct access to our field implementations.
"""

__all__ = [
    "magnet_cuboid_Bfield",
    "magnet_cylinder_Bfield_axialM",
    "magnet_cylinder_Hfield_diametralM",
    "magnet_cylinder_segment_Hfield",
    "magnet_sphere_Bfield",
    "dipole_Hfield",
    "current_circle_Bfield",
]

from magpylib._src.fields.field_BH_cuboid import magnet_cuboid_Bfield
from magpylib._src.fields.field_BH_cylinder import magnet_cylinder_Bfield_axialM
from magpylib._src.fields.field_BH_cylinder import magnet_cylinder_Hfield_diametralM
from magpylib._src.fields.field_BH_dipole import dipole_Hfield
from magpylib._src.fields.field_BH_circle import current_circle_Bfield
from magpylib._src.fields.field_BH_cylinder_segment import (
    magnet_cylinder_segment_Hfield,
)
from magpylib._src.fields.field_BH_sphere import magnet_sphere_Bfield

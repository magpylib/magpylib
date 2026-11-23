"""Functional interface"""

__all__ = [
    "circle_field",
    "circle_mesh",
    "cuboid_field",
    "cuboid_mesh",
    "cylinder_field",
    "cylinder_mesh",
    "cylinder_segment_field",
    "cylinder_segment_mesh",
    "dipole_field",
    "force_torque",
    "polyline_field",
    "polyline_mesh",
    "sphere_field",
    "tetrahedron_field",
    "tetrahedron_mesh",
    "triangle_charge_field",
    "triangle_current_field",
    "triangle_sheet_mesh",
    "triangle_strip_mesh",
]

from magpylib._src.fields.field_BHfunc import (
    circle_field,
    cuboid_field,
    cylinder_field,
    cylinder_segment_field,
    dipole_field,
    polyline_field,
    sphere_field,
    tetrahedron_field,
    triangle_charge_field,
    triangle_current_field,
)
from magpylib._src.fields.field_FT_func import (
    circle_mesh,
    cuboid_mesh,
    cylinder_mesh,
    cylinder_segment_mesh,
    force_torque,
    polyline_mesh,
    tetrahedron_mesh,
    triangle_sheet_mesh,
    triangle_strip_mesh,
)

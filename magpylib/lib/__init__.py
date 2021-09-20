"""
This subpackage gives direct access to the implementations of the analytical
expressions.
"""

__all__ = ['magnet_cyl_dia_H_Furlani1994', 'magnet_cyl_dia_H_Rauber2021',
    'magnet_cyl_axial_B_Derby2009', 'magnet_cyl_tile_H_Slanovc2021',
    'magnet_cuboid_B_Yang1999','current_loop_B_Smythe1950']

from magpylib._src.fields.field_BH_cylinder import magnet_cyl_dia_H_Furlani1994
from magpylib._src.fields.field_BH_cylinder import magnet_cyl_dia_H_Rauber2021
from magpylib._src.fields.field_BH_cylinder import magnet_cyl_axial_B_Derby2009
from magpylib._src.fields.field_BH_cylinder_tile import magnet_cyl_tile_H_Slanovc2021
from magpylib._src.fields.field_BH_cuboid import magnet_cuboid_B_Yang1999
from magpylib._src.fields.field_BH_circular import current_loop_B_Smythe1950

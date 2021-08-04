"""
This subpackage gives direct access to the implementations of the analytical
expressions.
"""

__all__ = ['cyl_dia_H_Furlani1994', 'cyl_dia_H_Rauber2021',
    'cyl_axial_B_Derby2009', 'cyl_tile_H_Slanovc2021',
    'cuboid_B_Yang1999']

from magpylib._lib.fields.field_BH_cylinder import cyl_dia_H_Furlani1994
from magpylib._lib.fields.field_BH_cylinder import cyl_dia_H_Rauber2021
from magpylib._lib.fields.field_BH_cylinder import cyl_axial_B_Derby2009
from magpylib._lib.fields.field_BH_cylinder_tile import cyl_tile_H_Slanovc2021
from magpylib._lib.fields.field_BH_cuboid import cuboid_B_Yang1999

"""
===================
Basic functionality
===================
"""

# %%
# Just compute the field
# ----------------------
# The most fundamental functionality of the library - compute the field (B in \[mT\], H in \[kA/m\]) of a
# source (here Cylinder magnet) at the observer position (1,2,3).

from magpylib.magnet import Cylinder

src = Cylinder(magnetization=(222, 333, 444), dimension=(2, 2))
B = src.getB((1, 2, 3))
print(B)

# %%
# Field values of a path
# ----------------------
# In this example the field B in \[mT\] of the Cylinder magnet is evaluated for a moving observer,
# rotating 360° with 45° steps around the source along the z-axis and a radius of 5\[mm\].

from magpylib.magnet import Cylinder
from magpylib import Sensor

src = Cylinder(magnetization=(222, 333, 444), dimension=(2, 2))
sens = Sensor(position=(5, 0, 0)).rotate_from_angax(
    [45] * 8, "z", anchor=(0, 0, 0), start=0, increment=True
)
B = sens.getB(src)
print(B)


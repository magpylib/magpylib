"""
This subpackage contains all electric current classes. Currents are modeled as line-currents,
input is the current in units of Ampere [A]. Field computation formulas are obtained via the law of
Biot-Savardt.
"""

__all__ = ['Circular', 'Line']

from magpylib._lib.obj_classes import Circular, Line

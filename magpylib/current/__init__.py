"""
This subpackage contains all electric current classes. Currents are modeled as line-currents,
input is the current in units of Ampere [A]. Field computation formulas are obtained via the law of
Biot-Savart.
"""

__all__ = ['Loop', 'Line']

from magpylib._src.obj_classes import Loop, Line

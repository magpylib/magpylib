"""
Current Sources
===============
   
Sources for generating magnetic fields originated from
:class:`~magpylib.source.current.Line` or :class:`~magpylib.source.current.Circular` currents. 
Compatible with :class:`~magpylib.Collection`
"""

__all__ = ["Circular","Line"] # This is for Sphinx

from magpylib._lib.classes.currents import Circular
from magpylib._lib.classes.currents import Line
"""
magpylib.Collection
===================

This package serves the :class:`~magpylib.Collection` class, which
is used to group up and manipulate sources created with :py:mod:`magpylib.source`

Angle and Axis information may be retrieved 
and generated with the methods in :py:mod:`magpylib.math` 

"""


__all__ = ["Collection", "source", "math"] # This is for Sphinx

from ._lib.classes.collection import Collection
from . import source, math
from . import _lib



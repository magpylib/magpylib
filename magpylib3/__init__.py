"""
Magpylib provides 3D magnetic field computation based on analytical formulas.

Basic Functionality:
--------------------

Create source objects that represent physical magnetic frield sources. Classes
    can be found in .magnet, .current and .moment top-level sub-packages.

Group sources using the top-level Collection class

Compute magnetic fields in 3 ways:
    1. source.getB(positions)
    2. getB(*sources, pos_obs = positions)
    3. getBv(**kwargs)

Graphially display sources using Matplotlib through the top-level display() 
    function.
"""

# module level dunders
__version__ = '3.0.0'
__author__ =  'Michael Ortner & friends'

# interface
__all__ = ['magnet', 'current', 'moment', 'getB', 'getH', 'getBv', 'getHv', 'Collection', 'display']

# create interface
from . import magnet
from . import current
from . import moment
from ._lib.fields.field_BHwrapper import getB, getH, getBv, getHv
from ._lib.obj_classes import Collection
from ._lib.graphics import display


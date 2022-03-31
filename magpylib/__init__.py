# pylint: disable=line-too-long

"""
Welcome to Magpylib !
---------------------

Magpylib combines static 3D magnetic field computation for permanent magnets,
currents and other sources. The computations are based on analytical expressions
and are combined with an object oriented (magnet, current, sensor) position/orientation
interface and graphical output to display the system geometry.

Resources
---------

Examples and documentation on Read-the-docs:

https://magpylib.readthedocs.io/en/latest/

Github repository:

https://github.com/magpylib/magpylib

Original software publication (version 2):

https://www.sciencedirect.com/science/article/pii/S2352711020300170

"""

# module level dunders
__version__ = '4.0.0rc1'
__author__ =  'Michael Ortner & Alexandre Boisselet'
__credits__ = 'The Magpylib community'
__all__ = ['magnet', 'current', 'misc', 'getB', 'getH',
    'Sensor', 'Collection', 'show', 'defaults', '__version__',
    '__author__', '__credits__', 'core', 'graphics']

# create interface to outside of package
from magpylib import magnet, current, misc, core, graphics
from magpylib._src.defaults.defaults_classes import default_settings as defaults
from magpylib._src.fields import getB, getH
from magpylib._src.obj_classes import Sensor
from magpylib._src.obj_classes.class_Collection import Collection
from magpylib._src.display.display import show

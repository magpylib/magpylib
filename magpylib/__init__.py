# pylint: disable=line-too-long
"""
Welcome to Magpylib !
---------------------

Magpylib is a Python package for calculating 3D static magnetic fields of
magnets, line currents and other sources. The computation is based on
analytical expressions and therefore extremely fast. A user friendly
geometry interface enables convenient relative positioning between sources
and observers.

Help us develop the package further - we appreciate any feedback !

Resources
---------

Documentation and examples on Read-the-docs:

https://magpylib.readthedocs.io/en/latest/

Our Github repository:

https://github.com/magpylib/magpylib

The original software publication (version 2):

https://www.sciencedirect.com/science/article/pii/S2352711020300170

"""
# module level dunders
__version__ = "5.2beta"
__author__ = "Michael Ortner & Alexandre Boisselet"
__credits__ = "The Magpylib community"
__all__ = [
    "magnet",
    "current",
    "misc",
    "getB",
    "getH",
    "getM",
    "getJ",
    "Sensor",
    "Collection",
    "show",
    "show_context",
    "defaults",
    "__version__",
    "__author__",
    "__credits__",
    "core",
    "graphics",
    "mu_0",
]

# create interface to outside of package
from scipy.constants import mu_0

from magpylib import core
from magpylib import current
from magpylib import graphics
from magpylib import magnet
from magpylib import misc
from magpylib._src.defaults.defaults_classes import default_settings as defaults
from magpylib._src.defaults.defaults_utility import SUPPORTED_PLOTTING_BACKENDS
from magpylib._src.display.display import show
from magpylib._src.display.display import show_context
from magpylib._src.fields import getB
from magpylib._src.fields import getH
from magpylib._src.fields import getJ
from magpylib._src.fields import getM
from magpylib._src.obj_classes.class_Collection import Collection
from magpylib._src.obj_classes.class_Sensor import Sensor

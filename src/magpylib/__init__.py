"""
Copyright (c) 2025 Michael Ortner. All rights reserved.

magpylib: Python package for computation of magnetic fields of magnets, currents and moments.
"""

from scipy.constants import mu_0

from magpylib import core, current, graphics, magnet, misc
from magpylib._src.defaults.defaults_classes import default_settings as defaults
from magpylib._src.defaults.defaults_utility import SUPPORTED_PLOTTING_BACKENDS
from magpylib._src.display.display import show, show_context
from magpylib._src.fields import getB, getFT, getH, getJ, getM
from magpylib._src.obj_classes.class_Collection import Collection
from magpylib._src.obj_classes.class_Sensor import Sensor

from ._version import version as __version__

__all__ = [
    "SUPPORTED_PLOTTING_BACKENDS",
    "Collection",
    "Sensor",
    "__version__",
    "core",
    "current",
    "defaults",
    "getB",
    "getFT",
    "getH",
    "getJ",
    "getM",
    "graphics",
    "magnet",
    "misc",
    "mu_0",
    "show",
    "show_context",
]

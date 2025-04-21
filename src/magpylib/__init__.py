"""
Copyright (c) 2025 Michael Ortner. All rights reserved.

magpylib: Python package for computation of magnetic fields of magnets, currents and moments.
"""

from __future__ import annotations

from ._version import version as __version__

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

__all__ = ["__version__",
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
    "core",
    "graphics",
    "mu_0",
]

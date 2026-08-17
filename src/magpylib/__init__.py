"""Copyright (c) 2026 Michael Ortner. All rights reserved.

magpylib: Python package for computation of magnetic fields of magnets, currents and moments.
"""

from scipy.constants import mu_0

from magpylib import core, current, func, graphics, magnet, misc
from magpylib._src.defaults.defaults_classes import default_settings as defaults
from magpylib._src.defaults.defaults_utility import SUPPORTED_PLOTTING_BACKENDS
from magpylib._src.display import api as _display_api
from magpylib._src.display.backend_registry import register_backend
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
    "func",
    "getB",
    "getFT",
    "getH",
    "getJ",
    "getM",
    "graphics",
    "magnet",
    "misc",
    "mu_0",
    "register_backend",
    "show",
    "show_context",
]

# Everything a third-party backend might import from magpylib is in place now,
# so entry points may be resolved. They still are not, until something asks
# about a backend name -- this only lifts the bar that kept the first such
# lookup, which happens during this import, from resolving them too early.
_display_api.DisplayBackend._importing = False

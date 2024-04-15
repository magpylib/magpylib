"""
The `magpylib.display.style` sub-package provides different object styles.
"""

__all__ = [
    "MagnetStyle",
    "CurrentStyle",
    "DipoleStyle",
    "SensorStyle",
]

from magpylib._src.defaults.defaults_classes import CurrentStyle
from magpylib._src.defaults.defaults_classes import DipoleStyle
from magpylib._src.defaults.defaults_classes import MagnetStyle
from magpylib._src.defaults.defaults_classes import SensorStyle

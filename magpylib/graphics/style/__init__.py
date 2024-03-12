"""
The `magpylib.display.style` sub-package provides different object styles.
"""

__all__ = [
    "MagnetStyle",
    "CurrentStyle",
    "DipoleStyle",
    "SensorStyle",
]

from magpylib._src.style import CurrentStyle
from magpylib._src.style import DipoleStyle
from magpylib._src.style import MagnetStyle
from magpylib._src.style import SensorStyle

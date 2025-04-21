"""
The `magpylib.display` sub-package provides additional plotting
features for independent use.
"""

from __future__ import annotations

__all__ = ["Trace3d", "model3d", "style"]

from magpylib._src.style import Trace3d
from magpylib.graphics import model3d, style

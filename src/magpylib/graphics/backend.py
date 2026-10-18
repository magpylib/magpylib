"""magpylib.graphics.backend module
==================================

Public API for writing a magpylib display backend.

A backend subclasses `DisplayBackend`, declares what it can do, and renders
the `Scene` it is handed::

    from magpylib.graphics.backend import DisplayBackend

    class ThreeBackend(DisplayBackend):
        name = "three"
        description = "Interactive three.js renderer"
        api_version = 1  # the payload contract it was written against
        supports_animation = True

        def show(self, scene):
            for frame in scene.frames:
                for trace in frame.traces:
                    ...

Naming the class in the ``magpylib.backends`` entry-point group makes
``pip install`` enough for a user to select it -- no registration call::

    [project.entry-points."magpylib.backends"]
    three = "magpylib_three:ThreeBackend"

`magpylib.register_backend` is the imperative equivalent, for a backend
defined in a script or notebook.

The user guide page "Writing a custom display backend" specifies the trace
dialect a backend receives. Its vocabularies are `ALLOWED_SYMBOLS` and
`ALLOWED_LINESTYLES`, and `magic_to_dict` nests its underscore keys.
`pixel_fields`, `pixel_sizes` and `pixel_colors` let a backend draw sensor
pixels its own way while sizing and coloring them as the built-ins do;
`scaled_norms` and `colors_from_colorscale` scale and color other field
magnitudes, such as field lines, on the same scale.
"""

__all__ = [
    "ALLOWED_LINESTYLES",
    "ALLOWED_SYMBOLS",
    "API_VERSION",
    "ENTRY_POINT_GROUP",
    "PLACEMENT_KEYS",
    "TRACE_META_KEYS",
    "AnimationSettings",
    "DisplayBackend",
    "Frame",
    "Panel",
    "Scene",
    "colors_from_colorscale",
    "drawing_properties",
    "magic_to_dict",
    "pixel_colors",
    "pixel_fields",
    "pixel_sizes",
    "scaled_norms",
]

from magpylib._src.defaults import defaults_utility as _defaults_utility
from magpylib._src.defaults.defaults_utility import magic_to_dict
from magpylib._src.display import api as _api
from magpylib._src.display.api import (
    AnimationSettings,
    DisplayBackend,
    Frame,
    Panel,
    Scene,
    drawing_properties,
)
from magpylib._src.display.backend_tools import (
    colors_from_colorscale,
    pixel_colors,
    pixel_fields,
    pixel_sizes,
    scaled_norms,
)

# The constants are bound again here, each under a ``#:`` comment, because
# Sphinx documents a module's data only from comments in that module's own
# source. They are the very same objects as in magpylib._src.

#: The values ``marker_symbol`` takes in the trace dialect, Matplotlib-derived.
#: A user's ``generic`` model3d trace passes its own values through unchecked.
ALLOWED_SYMBOLS = _defaults_utility.ALLOWED_SYMBOLS

#: The values ``line_dash`` takes in the trace dialect, Matplotlib-derived.
#: A user's ``generic`` model3d trace passes its own values through unchecked.
ALLOWED_LINESTYLES = _defaults_utility.ALLOWED_LINESTYLES

#: Version of the payload contract this magpylib emits. Declare the version
#: a backend was written against as `DisplayBackend.api_version`; magpylib
#: warns when the two differ. Bumped only when a backend written against the
#: previous version could misread what it is handed, never for an addition.
API_VERSION = _api.API_VERSION

#: Entry-point group a package advertises its backend under, so that
#: ``pip install`` is enough to make it selectable.
ENTRY_POINT_GROUP = _api.ENTRY_POINT_GROUP

#: Trace keys placing a trace in the subplot grid, ``row`` and ``col``.
#: `drawing_properties` strips them: read them first.
PLACEMENT_KEYS = _api.PLACEMENT_KEYS

#: Trace keys carrying magpylib metadata rather than drawing properties, such
#: as ``object_id``. The set may grow in a later release: strip them with
#: `drawing_properties`, never by a list of your own.
TRACE_META_KEYS = _api.TRACE_META_KEYS

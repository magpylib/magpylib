"""Public API for writing a magpylib display backend.

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

See the user guide for the trace dialect a backend receives. Its vocabularies
are `ALLOWED_SYMBOLS` and `ALLOWED_LINESTYLES`, and `magic_to_dict` nests its
underscore keys. `sensor_pixel_fields`, `scale_pixel_norms` and
`colors_from_colorscale` let a backend draw sensor pixels its own way while
sizing and colouring them as the built-ins do.
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
    "scale_pixel_norms",
    "sensor_pixel_fields",
]

from magpylib._src.defaults.defaults_utility import (
    ALLOWED_LINESTYLES,
    ALLOWED_SYMBOLS,
    magic_to_dict,
)
from magpylib._src.display.api import (
    API_VERSION,
    ENTRY_POINT_GROUP,
    PLACEMENT_KEYS,
    TRACE_META_KEYS,
    AnimationSettings,
    DisplayBackend,
    Frame,
    Panel,
    Scene,
    drawing_properties,
)
from magpylib._src.display.backend_tools import (
    colors_from_colorscale,
    scale_pixel_norms,
    sensor_pixel_fields,
)

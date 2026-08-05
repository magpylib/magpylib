"""Public API for writing a magpylib display backend.

A backend subclasses `DisplayBackend`, declares what it can do, and renders
the `Scene` it is handed::

    from magpylib.graphics.backend import DisplayBackend

    class ThreeBackend(DisplayBackend):
        name = "three"
        description = "Interactive three.js renderer"
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

See the user guide for the trace dialect a backend receives.
"""

__all__ = [
    "API_VERSION",
    "ENTRY_POINT_GROUP",
    "TRACE_META_KEYS",
    "AnimationSettings",
    "DisplayBackend",
    "Frame",
    "Panel",
    "Scene",
    "drawing_properties",
]

from magpylib._src.display.api import (
    API_VERSION,
    ENTRY_POINT_GROUP,
    TRACE_META_KEYS,
    AnimationSettings,
    DisplayBackend,
    Frame,
    Panel,
    Scene,
    drawing_properties,
)

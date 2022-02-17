"""BaseGeo class code
READY FOR V4
"""

from magpylib._src.display.display import show

class BaseDisplayRepr:
    """Provides the display(self) and self.repr methods for all objects"""

    show = show
    _object_type = None

    def __repr__(self) -> str:
        name = getattr(self, "name", None)
        if name is None and hasattr(self, "style"):
            name = getattr(getattr(self, "style"), "label", None)
        name_str = "" if name is None else f", label={name!r}"
        return f"{type(self).__name__}(id={id(self)!r}{name_str})"

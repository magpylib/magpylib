"""BaseGeo class code"""

from magpylib._src.display.display import show

# ALL METHODS ON INTERFACE
class BaseDisplayRepr:
    """Provides the display(self) and self.repr methods for all objects

    Properties
    ----------

    Methods
    -------
    - show(self, **kwargs)
    - repr
    """

    show = show
    _object_type = None

    # ------------------------------------------------------------------
    # INTERFACE
    def __repr__(self) -> str:
        name = getattr(self, "name", None)
        if name is None and hasattr(self, "style"):
            name = getattr(getattr(self, "style"), "name", None)
        name_str = "" if name is None else f", name={name!r}"
        return f"{type(self).__name__}(id={id(self)!r}{name_str})"

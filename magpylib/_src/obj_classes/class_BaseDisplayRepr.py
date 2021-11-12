"""BaseGeo class code"""

from magpylib._src.display import display

# ALL METHODS ON INTERFACE
class BaseDisplayRepr:
    """ Provides the display(self) and self.repr methods for all objects

    Properties
    ----------

    Methods
    -------
    - display(self, **kwargs)
    - repr
    """

    display = display

    def __init__(self):
        if not hasattr(self, '_object_type'):
            self._object_type = None

    # ------------------------------------------------------------------
    # INTERFACE
    def __repr__(self) -> str:
        return f'{self._object_type}(id={str(id(self))})'

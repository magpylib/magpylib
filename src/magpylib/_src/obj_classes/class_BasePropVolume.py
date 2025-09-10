"""
Base class adding the VOLUME property.
This is a separate class from BaseMagnet because Dipole has no volume property
"""

from abc import ABC, abstractmethod


class BaseVolume(ABC):
    """Base class for Magpylib objects for inheriting the volume property."""

    @property
    def volume(self):
        """Return object volume."""
        return self._get_volume()

    @volume.setter
    def volume(self, value):
        """Throw error when trying to set volume."""
        raise AttributeError("Cannot set property `volume`. It is read-only.")

    @abstractmethod
    def _get_volume(self):
        """Calculate and return the volume of the object in units of m³."""

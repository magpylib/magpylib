"""
Base class adding the DIPOLE_MOMENT property.
This is a separate class because this property is inherited by a mix of different classes.
"""

from abc import ABC, abstractmethod

class BaseDipoleMoment(ABC):
    """Base class for Magpylib objects for inheriting the dipole_moment property."""

    @property
    def dipole_moment(self):
        """Return object dipole moment."""
        return self._get_dipole_moment()

    @dipole_moment.setter
    def dipole_moment(self, value):
        """Throw error when trying to set dipole moment."""
        raise AttributeError("Cannot set property `dipole_moment`. It is read-only.")

    @abstractmethod
    def _get_dipole_moment(self):
        """Calculate and return the dipole moment of the object in units of A·m²."""

"""Base class properties

Dipole moment is a separate class from BaseMagnet because this property is inherited by a mix of different classes.

Volume is a separate class from BaseMagnet because Dipole has no volume property
"""

import warnings
from abc import ABC, abstractmethod


class BaseDipoleMoment(ABC):
    """Base class for Magpylib objects for inheriting the dipole_moment property."""

    @property
    def dipole_moment(self):
        """Return dipole moment vector (A·m²) in the object's own frame, like its
        polarization; a Collection's is in the Collection's own frame."""
        return self._get_dipole_moment(squeeze=True)

    @property
    def _dipole_moment(self):
        """Return dipole moment vector (A·m²) without squeezing (internal)."""
        return self._get_dipole_moment(squeeze=False)

    @dipole_moment.setter
    def dipole_moment(self, _input):
        """Raise error on attempt to set read-only dipole moment."""
        msg = "Cannot set property dipole_moment. It is read-only."
        raise AttributeError(msg)

    @_dipole_moment.setter
    def _dipole_moment(self, _input):
        """Raise error on attempt to set read-only dipole moment."""
        msg = "Cannot set property _dipole_moment. It is read-only."
        raise AttributeError(msg)

    @abstractmethod
    def _get_dipole_moment(self, squeeze=True):
        """Return computed dipole moment (A·m²).

        Returns
        -------
        ndarray, shape (3,)
            Dipole moment vector in units (A·m²).
        """


class BaseVolume(ABC):
    """Base class for Magpylib objects for inheriting the volume property."""

    @property
    def volume(self):
        """Return object volume (m³)."""
        return self._get_volume()

    @volume.setter
    def volume(self, _input):
        """Raise error on attempt to set read-only volume."""
        msg = "Cannot set property volume. It is read-only."
        raise AttributeError(msg)

    @abstractmethod
    def _get_volume(self):
        """Return computed volume (m³).

        Returns
        -------
        float
            Volume in units (m³).
        """


class BaseBarycenter:
    """Deprecated ``barycenter`` alias of ``centroid``, on the classes that had it
    before version 6."""

    @property
    def barycenter(self):
        """Return barycenter (m).

        .. deprecated:: 6.0.0
            Use :attr:`centroid` instead. The ``barycenter`` property will be removed
            in a future version.
        """
        warnings.warn(
            "The 'barycenter' property is deprecated and will be removed in a future "
            "version. Use 'centroid' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.centroid  # pylint: disable=no-member  # mixed into BaseGeo classes

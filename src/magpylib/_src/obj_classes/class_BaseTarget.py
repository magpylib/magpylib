"""Base class for objects that can be targets of force computation."""

from abc import ABC, abstractmethod


class BaseTarget(ABC):
    """Base class for Magpylib objects that can be targets of force computation.

    - adds parameter meshing, default=None
    - adds parameter meshing setter
    - adds default method _validate_meshing()
    - enforces _generate_mesh() as abstract method
    """

    _force_type: str = None

    def __init__(self, meshing=None):
        """Initialize BaseTarget with meshing parameters."""
        self._meshing = meshing

    @property
    def meshing(self):
        """Get mesh parameters for force computation."""
        return self._meshing

    @meshing.setter
    def meshing(self, value):
        """Set mesh parameters for force computation."""
        # Basic validation - subclasses override on demand for specific requirements
        if value is not None:
            self._validate_meshing(value)
        self._meshing = value

    def _validate_meshing(self, value):
        """
        Basic meshing validation: allow positive integers
        Subclasses should override for specific requirements.
        """
        if isinstance(value, int) and value > 0:
            pass
        else:
            msg = f"Meshing parameter must be positive integer for {self}. Instead got {value}."
            raise ValueError(msg)

    @abstractmethod
    def _generate_mesh(self):
        """
        Generate meshing dictionary
        """

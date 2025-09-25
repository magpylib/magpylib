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
        self.meshing = meshing

    @property
    def meshing(self):
        """Return meshing specification for force computation."""
        return self._meshing

    @meshing.setter
    def meshing(self, value):
        """Set meshing specification.

        Parameters
        ----------
        value : int | None
            Meshing finesse parameter.
        """
        # Basic validation - subclasses may override for specific requirements
        if value is not None:
            self._validate_meshing(value)
        self._meshing = value

    def _validate_meshing(self, value):
        """Basic meshing validation: allow positive integers

        Subclasses override this method with class-specific requirements.
        """
        if isinstance(value, int) and value > 0:
            pass
        else:
            msg = f"Input meshing must be positive integer for {self}; instead received {value}."
            raise ValueError(msg)

    @abstractmethod
    def _generate_mesh(self):
        """Return meshing data structure.

        Returns
        -------
        dict
            Meshing representation required for downstream force calculations. dict
            contains keys 'pts' and 'cvecs' for currents, and 'pts' and 'moments' for
            magnets.
        """

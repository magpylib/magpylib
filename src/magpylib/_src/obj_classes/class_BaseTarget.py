"""Base class for objects that can be targets of force computation."""

from abc import ABC, abstractmethod
from numbers import Integral


def check_meshing_count(value, obj, minimum=1, what=None):
    """Check that a meshing value is an integer >= minimum and return it as int.

    Booleans are not integers here, NumPy integers are.

    Parameters
    ----------
    value : object
        The meshing input.
    obj : object
        The target, named in the error message.
    minimum : int, default 1
        Smallest accepted value.
    what : str | None, default None
        What the value must be, for the error message. Defaults to 'a positive
        integer' for minimum 1 and 'an integer >= minimum' otherwise.
    """
    if isinstance(value, Integral) and not isinstance(value, bool) and value >= minimum:
        return int(value)
    if what is None:
        what = "a positive integer" if minimum == 1 else f"an integer >= {minimum}"
    msg = f"Input meshing of {obj} must be {what}; instead received {value!r}."
    raise ValueError(msg)


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
        """Discretization setting: the target number of cells or elements the object
        is cut into, or ``None``."""
        return self._meshing

    @meshing.setter
    def meshing(self, value):
        """Set meshing specification.

        Parameters
        ----------
        value : int | None
            Target number of cells or elements; a Cuboid also takes a triple.
        """
        # Subclasses override _validate_meshing for class-specific requirements
        if value is not None:
            value = self._validate_meshing(value)
        self._meshing = value

    def _validate_meshing(self, value):
        """Check a meshing input and return it normalised: a positive integer.

        Subclasses override this method with class-specific requirements.
        """
        return check_meshing_count(value, self)

    @abstractmethod
    def _generate_mesh(self):
        """Return meshing data structure.

        Returns
        -------
        dict
            Meshing representation required for downstream force calculations. dict
            contains keys 'pts' and 'cvecs' for currents, and 'pts', 'moments' and
            'cells' for magnets.
        """

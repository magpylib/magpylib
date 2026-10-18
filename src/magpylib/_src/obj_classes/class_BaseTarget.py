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


def check_meshing_count_or_triple(value, obj):
    """Check a meshing that is a positive integer or a triple of them, and return
    it as an int or a tuple of 3 ints."""
    what = "a positive integer or a triple of positive integers"
    if isinstance(value, list | tuple) or (
        hasattr(value, "shape") and getattr(value, "ndim", 0) > 0
    ):
        if len(value) != 3 or any(
            isinstance(v, list | tuple) or getattr(v, "ndim", 0) > 0 for v in value
        ):
            msg = f"Input meshing of {obj} must be {what}; instead received {value!r}."
            raise ValueError(msg)
        return tuple(check_meshing_count(v, obj, what=what) for v in value)
    return check_meshing_count(value, obj, what=what)


class BaseTarget(ABC):
    """Base class for Magpylib objects that can be targets of force computation.

    - validates a ``meshing`` value for the class in ``_validate_meshing()``
    - enforces ``_generate_mesh(meshing)`` as abstract method
    """

    _force_type: str = None

    def _validate_meshing(self, value):
        """Check a meshing input and return it normalized: a positive integer.

        Subclasses override this method with class-specific requirements.
        """
        return check_meshing_count(value, self)

    @abstractmethod
    def _generate_mesh(self, meshing):
        """Return meshing data structure for the given, validated meshing value.

        Returns
        -------
        dict
            Meshing representation required for downstream force calculations. dict
            contains keys 'pts' and 'cvecs' for currents, and 'pts', 'moments' and
            'cells' for magnets.
        """

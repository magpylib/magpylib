"""BaseHomMag class code
DOCSTRINGS V4 READY
"""
from magpylib._src.input_checks import check_format_input_scalar
from magpylib._src.input_checks import check_format_input_vector


class BaseHomMag:
    """provides the magnetization attribute  for homogeneously magnetized magnets"""

    def __init__(self, magnetization):
        self.magnetization = magnetization

    @property
    def magnetization(self):
        """Object magnetization attribute getter and setter."""
        return self._magnetization

    @magnetization.setter
    def magnetization(self, mag):
        """Set magnetization vector, array_like, shape (3,), unit [mT]."""
        self._magnetization = check_format_input_vector(
            mag,
            dims=(1,),
            shape_m1=3,
            sig_name="magnetization",
            sig_type="array_like (list, tuple, ndarray) with shape (3,)",
            allow_None=True,
        )


class BaseCurrent:
    """provides scalar current attribute"""

    def __init__(self, current):
        self.current = current

    @property
    def current(self):
        """Object current attribute getter and setter."""
        return self._current

    @current.setter
    def current(self, current):
        """Set current value, scalar, unit [A]."""
        # input type and init check
        self._current = check_format_input_scalar(
            current,
            sig_name="current",
            sig_type="`None` or a number (int, float)",
            allow_None=True,
        )

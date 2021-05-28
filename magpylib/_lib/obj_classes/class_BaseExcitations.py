"""BaseHomMag class code"""

import numpy as np
from magpylib._lib.config import Config
from magpylib._lib.input_checks import (check_vector_init, check_vector_type, check_vector_format,
    check_scalar_init, check_scalar_type)


# MAG PROPERTY ON INTERFACE
class BaseHomMag:
    """
    provide magnetization attribute (homogeneous magnetization)

    Properties
    ----------
    mag

    Methods
    -------
    """
    def __init__(self, mag):
        self.mag = mag

    @property
    def mag(self):
        """ Magnet magnetization in units of [mT].
        """
        return self._mag

    @mag.setter
    def mag(self, mag):
        """ Set magnetization vector, shape (3,), unit [mT].
        """
        # input type and init check
        if Config.CHECK_INPUTS:
            check_vector_type(mag, 'magnetization')
            check_vector_init(mag, 'magnetization')

        # input type -> ndarray
        mag = np.array(mag, dtype=float)

        # input format check
        if Config.CHECK_INPUTS:
            check_vector_format(mag, (3,),'magnetization')

        self._mag = mag


# CURRENT PROPERTY ON INTERFACE
class BaseCurrent:
    """
    provide scalar current attribute

    Properties
    ----------
    current

    Methods
    -------
    """
    def __init__(self, current):
        self.current = current

    @property
    def current(self):
        """ Current in units of [A].
        """
        return self._current

    @current.setter
    def current(self, current):
        """ Set Current value, unit [A].
        """
        # input type and init check
        if Config.CHECK_INPUTS:
            check_scalar_init(current, 'current')
            check_scalar_type(current, 'current')

        self._current = float(current)

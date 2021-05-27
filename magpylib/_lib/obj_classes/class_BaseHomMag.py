"""BaseHomMag class code"""

import numpy as np
from magpylib._lib.config import Config
from magpylib._lib.input_checks import check_mag_type, check_mag_format, check_mag_init

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
        # input type check
        if Config.CHECK_INPUTS:
            check_mag_type(mag, 'magnetization')
            check_mag_init(mag, 'magnetization')

        # input type -> ndarray
        mag = np.array(mag, dtype=float)

        # input format check
        if Config.CHECK_INPUTS:
            check_mag_format(mag, 'magnetization')

        self._mag = mag

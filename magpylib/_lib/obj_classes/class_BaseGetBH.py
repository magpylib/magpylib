"""BaseGetBHsimple class code"""

from magpylib._lib.fields.field_wrap_BH_level2 import getBH_level2
from magpylib._lib.utility import format_star_input


# ALL METHODS ON INTERFACE
class BaseGetBH:
    """ provides simple getB and getH methods

    Properties
    ----------

    Methods
    -------
    - getB(self, *observers)
    - getH(self, *observers)
    """

    # ------------------------------------------------------------------
    # INTERFACE
    def getB(self, *observers, squeeze=True):
        """
        Compute B-field of source/Collection at observers.

        Parameters
        ----------
        observers: array_like or Sensor or list of Sensors
            Observers can be array_like positions of shape (N1, N2, ..., 3) or a Sensor object or
            a 1D list of K Sensor objects with pixel position shape of (N1, N2, ..., 3) in units
            of [mm].

        squeeze: bool, default=True
            If True, the output is squeezed, i.e. all axes of length 1 in the output (e.g. only
            a single sensor or only a single source) are eliminated.

        Returns
        -------
        B-field: ndarray, shape squeeze(M, K, N1, N2, ..., 3), unit [mT]
            B-field at each path position (M) for each sensor (K) and each sensor pixel position
            (N) in units of [mT].
            Output is squeezed, i.e. every dimension of length 1 (single sensor or no sensor or
            single pixel) is removed.
        """
        observers = format_star_input(observers)
        return getBH_level2(True, self, observers, False, squeeze)

    # ------------------------------------------------------------------
    # INTERFACE
    def getH(self, *observers, squeeze=True):
        """
        Compute H-field of source at observers.

        Parameters
        ----------
        observers: array_like or Sensor or list of Sensors
            Observers can be array_like positions of shape (N1, N2, ..., 3) or a Sensor object or
            a 1D list of K Sensor objects with pixel position shape of (N1, N2, ..., 3) in units
            of [mm].

        squeeze: bool, default=True
            If True, the output is squeezed, i.e. all axes of length 1 in the output (e.g. only
            a single sensor or only a single source) are eliminated.

        Returns
        -------
        H-field: ndarray, shape squeeze(M, K, N1, N2, ..., 3), unit [kA/m]
            B-field at each path position (M) for each sensor (K) and each sensor pixel position
            (N) in units of [kA/m].
            Output is squeezed, i.e. every dimension of length 1 (single sensor or no sensor or
            single pixel) is removed.
        """
        observers = format_star_input(observers)
        return getBH_level2(False, self, observers, False, squeeze)

"""BaseGetBHsimple class code"""

from magpylib._src.fields.field_wrap_BH_level2 import getBH_level2
from magpylib._src.utility import format_star_input


# ALL METHODS ON INTERFACE
class BaseGetBH:
    """provides simple getB and getH methods

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
        Compute B-field in units of [mT] for given observers.

        Parameters
        ----------

        observers: array_like or Sensors
            Observers can be array_like positions of shape (N1, N2, ..., 3) where the field
            should be evaluated or Sensor objects with pixel shape (N1, N2, ..., 3). Pixel
            shapes (or observer positions) of all inputs must be the same. All positions
            are given in units of [mm].

        squeeze: bool, default=True
            If True, the output is squeezed, i.e. all axes of length 1 in the output (e.g.
            only a single source) are eliminated.

        Returns
        -------
        B-field: ndarray, shape squeeze(M, K, N1, N2, ..., 3)
            B-field at each path position (M) for each sensor (K) and each sensor pixel
            position (N1,N2,...) in units of [mT]. Sensor pixel positions are equivalent
            to simple observer positions. Paths of objects that are shorter than M will be
            considered as static beyond their end.

        Examples
        --------
        """
        observers = format_star_input(observers)
        return getBH_level2(self, observers, sumup=False, squeeze=squeeze, field='B')

    # ------------------------------------------------------------------
    # INTERFACE
    def getH(self, *observers, squeeze=True):
        """
        Compute H-field in units of [kA/m] for given observers.

        Parameters
        ----------

        observers: array_like or Sensors
            Observers can be array_like positions of shape (N1, N2, ..., 3) where the field
            should be evaluated or Sensor objects with pixel shape (N1, N2, ..., 3). Pixel
            shapes (or observer positions) of all inputs must be the same. All positions
            are given in units of [mm].

        squeeze: bool, default=True
            If True, the output is squeezed, i.e. all axes of length 1 in the output (e.g.
            only a single source) are eliminated.

        Returns
        -------
        H-field: ndarray, shape squeeze(M, K, N1, N2, ..., 3)
            H-field at each path position (M) for each sensor (K) and each sensor pixel
            position (N1,N2,...) in units of [kA/m]. Sensor pixel positions are equivalent
            to simple observer positions. Paths of objects that are shorter than M will be
            considered as static beyond their end.

        Examples
        --------
        """
        observers = format_star_input(observers)
        return getBH_level2(self, observers, sumup=False, squeeze=squeeze, field='H')

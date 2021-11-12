"""BaseGetBHsimple class code"""

from magpylib._src.fields.field_wrap_BH_level2 import getBH_level2
from magpylib._src.utility import format_star_input


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
        Compute B-field in units of [mT] for given observers.

        Parameters
        ----------

        observers: array_like or Sensors
            Observers can be array_like positions of shape (N1, N2, ..., 3) where the field
            should be evaluated or Sensor objects with pixel shape (N1, N2, ..., 3). Pixel
            shapes (or observer positions) of all inputs must be the same. All positions
            are given in units of [mm].

        sumup: bool, default=False
            If True, the fields of all sources are summed up.

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
        Compute the B-field [mT] at a sensor directly through the source method:

        >>> import magpylib as magpy
        >>> source = magpy.magnet.Sphere(magnetization=(1000,0,0), diameter=1)
        >>> sensor = magpy.Sensor(position=(1,2,3))
        >>> B = source.getB(sensor)
        >>> print(B)
        [-0.62497314  0.34089444  0.51134166]

        Compute the B-field [mT] of a source at five path positions as seen
        by an observer at position (1,2,3):

        >>> import magpylib as magpy
        >>> source = magpy.magnet.Sphere(magnetization=(1000,0,0), diameter=1)
        >>> source.move([(x,0,0) for x in [1,2,3,4,5]])
        >>> B = source.getB((1,2,3))
        >>> print(B)
        [[-0.88894262  0.          0.        ]
        [-0.62497314 -0.34089444 -0.51134166]
        [-0.17483825 -0.41961181 -0.62941771]
        [ 0.09177028 -0.33037301 -0.49555952]
        [ 0.17480239 -0.22080302 -0.33120453]]

        Compute the B-field [mT] of a source at two sensors:

        >>> import magpylib as magpy
        >>> source = magpy.current.Loop(current=15, diameter=1)
        >>> sens1 = magpy.Sensor(position=(1,2,3))
        >>> sens2 = magpy.Sensor(position=(2,3,4))
        >>> B = source.getB(sens1, sens2)
        >>> print(B)
        [[0.01421427 0.02842853 0.02114728]
        [0.00621368 0.00932052 0.00501254]]

        """
        observers = format_star_input(observers)
        return getBH_level2(True, self, observers, False, squeeze)

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

        sumup: bool, default=False
            If True, the fields of all sources are summed up.

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
        Compute the H-field [kA/m] at a sensor directly through the source method:

        >>> import magpylib as magpy
        >>> source = magpy.magnet.Sphere(magnetization=(1000,0,0), diameter=1)
        >>> sensor = magpy.Sensor(position=(1,2,3))
        >>> H = source.getH(sensor)
        >>> print(H)
        [-0.49733782  0.27127518  0.40691277]

        Compute the H-field [kA/m] of a source at five path positions as seen
        by an observer at position (1,2,3):

        >>> import magpylib as magpy
        >>> source = magpy.magnet.Sphere(magnetization=(1000,0,0), diameter=1)
        >>> source.move([(x,0,0) for x in [1,2,3,4,5]])
        >>> H = source.getH((1,2,3))
        >>> print(H)
        [[-0.70739806  0.          0.        ]
        [-0.49733782 -0.27127518 -0.40691277]
        [-0.13913186 -0.33391647 -0.5008747 ]
        [ 0.07302847 -0.26290249 -0.39435373]
        [ 0.13910332 -0.17570946 -0.26356419]]

        Compute the H-field [kA/m] of a source at two sensors:

        >>> import magpylib as magpy
        >>> source = magpy.current.Loop(current=15, diameter=1)
        >>> sens1 = magpy.Sensor(position=(1,2,3))
        >>> sens2 = magpy.Sensor(position=(2,3,4))
        >>> H = source.getH(sens1, sens2)
        >>> print(H)
        [[0.01131135 0.02262271 0.01682847]
        [0.00494469 0.00741704 0.00398885]]

        """
        observers = format_star_input(observers)
        return getBH_level2(False, self, observers, False, squeeze)

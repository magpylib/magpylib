"""Sensor class code"""
import numpy as np
from magpylib._lib.obj_classes.class_BaseGeo import BaseGeo
from magpylib._lib.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._lib.utility import format_star_input
from magpylib._lib.config import Config
from magpylib._lib.input_checks import check_vector_type, check_position_format
from magpylib._lib.fields.field_wrap_BH_level2 import getBH_level2


# ON INTERFACE
class Sensor(BaseGeo, BaseDisplayRepr):
    """
    Magnetic field sensor. Can be used as observer input for magnetic field
    computation.

    Reference position and orientation: Local (Sensor) and global coordinate systems
    coincide when pos = (0,0,0) and orientation = unit rotation.

    Parameters
    ----------
    position: array_like, shape (3,) or (M,3), default=(0,0,0)
        Object position in the global CS relative to the reference position in
        units of [mm]. For M>1, the position attribute represents a path in the
        global CS. The attributes orientation and position must always be of
        the same length.

    pixel: array_like, shape (3,) or (N1,N2,...,3), default=(0,0,0)
        Sensor pixel positions inside of a sensor package. Positions are given in
        the local Sensor CS.

    orientation: scipy Rotation object with length 1 or M, default=unit rotation
        Object orientation in the global CS relative to the reference orientation.
        For M>1 orientation represents different values along a path. The attributes
        orientation and position must always be of the same length.

    Returns
    -------
    Sensor object: Sensor

    Examples
    --------

    By default a Sensor is initialized at position (0,0,0), with unit rotation
    and pixel (0,0,0):

    >>> import magpylib as mag3
    >>> sensor = mag3.Sensor()
    >>> print(sensor.position)
    [0. 0. 0.]
    >>> print(sensor.pixel)
    [0. 0. 0.]
    >>> print(sensor.orientation.as_quat())
    [0. 0. 0. 1.]

    Sensors are observers for magnetic field computation. In this example we compute the
    H-field as seen by the sensor in the center of a circular current loop:

    >>> import magpylib as mag3
    >>> sensor = mag3.Sensor()
    >>> loop = mag3.current.Circular(current=1, diameter=1)
    >>> H = sensor.getH(loop)
    >>> print(H)
    [0. 0. 1.]

    Field computation is performed at every pixel of a sensor. The above example is reproduced
    for a 2x2-pixel sensor:

    >>> import magpylib as mag3
    >>> sensor = mag3.Sensor(pixel=[[(0,0,0), (0,0,1)],[(0,0,2), (0,0,3)]])
    >>> loop = mag3.current.Circular(current=1, diameter=1)
    >>> H = sensor.getH(loop)
    >>> print(H.shape)
    (2, 2, 3)
    >>> print(H)
    [[[0.         0.         1.        ]
      [0.         0.         0.08944272]]
     [[0.         0.         0.0142668 ]
      [0.         0.         0.00444322]]]

    Compute the field of a sensor along a path. The path positions are chosen so that
    they coincide with the pixel positions of the previous example:

    >>> import magpylib as mag3
    >>> loop = mag3.current.Circular(current=1, diameter=1)
    >>> sensor = mag3.Sensor()
    >>> sensor.move([(0,0,1)]*3, start=1, increment=True)
    >>> print(sensor.position)
    [[0. 0. 0.]
     [0. 0. 1.]
     [0. 0. 2.]
     [0. 0. 3.]]
    >>> H = sensor.getH(loop)
    >>> print(H)
    [[0.         0.         1.        ]
     [0.         0.         0.08944272]
     [0.         0.         0.0142668 ]
     [0.         0.         0.00444322]]

    """

    def __init__(
            self,
            position = (0,0,0),
            pixel=(0,0,0),
            orientation = None):

        # inherit base_geo class
        BaseGeo.__init__(self, position, orientation)
        BaseDisplayRepr.__init__(self)

        # set mag and dim attributes
        self.pixel = pixel
        self.object_type = 'Sensor'


    # properties ----------------------------------------------------
    @property
    def pixel(self):
        """ Sensor pixel attribute getter and setter.
        """
        return self._pixel


    @pixel.setter
    def pixel(self, pix):
        """
        Set Sensor pixel positions in Sensor CS, array_like, shape (...,3,)
        """
        # check input type
        if Config.CHECK_INPUTS:
            check_vector_type(pix, 'pixel_position')

        # input type -> ndarray
        pix = np.array(pix, dtype=float)

        # check input format
        if Config.CHECK_INPUTS:
            check_position_format(pix, 'pixel_position')

        self._pixel = pix


    # methods -------------------------------------------------------
    def getB(self, *sources, sumup=False, squeeze=True):
        """
        Compute B-field in [mT] for given sources.

        Parameters
        ----------
        sources: source objects or Collections
            Sources can be a mixture of L source objects or Collections.

        sumup: bool, default=False
            If True, the fields of all sources are summed up.

        squeeze: bool, default=True
            If True, the output is squeezed, i.e. all axes of length 1 in the output (e.g. only
            a single sensor or only a single source) are eliminated.

        Returns
        -------
        B-field: ndarray, shape squeeze(L, M, N1, N2, ..., 3)
            B-field of each source (L) at each path position (M) and each sensor pixel
            position (N1,N2,...) in units of [mT]. Paths of objects that are shorter than
            M will be considered as static beyond their end.

        Examples
        --------

        Sensors are observers for magnetic field computation. In this example we compute the
        B-field [mT] as seen by the sensor in the center of a circular current loop:

        >>> import magpylib as mag3
        >>> sensor = mag3.Sensor()
        >>> loop = mag3.current.Circular(current=1, diameter=1)
        >>> B = sensor.getB(loop)
        >>> print(B)
        [0.         0.         1.25663706]

        Field computation is performed at every pixel of a sensor. The above example is reproduced
        for a 2x2-pixel sensor:

        >>> import magpylib as mag3
        >>> sensor = mag3.Sensor(pixel=[[(0,0,0), (0,0,1)],[(0,0,2), (0,0,3)]])
        >>> loop = mag3.current.Circular(current=1, diameter=1)
        >>> B = sensor.getB(loop)
        >>> print(B.shape)
        (2, 2, 3)
        >>> print(B)
        [[[0.         0.         1.25663706]
          [0.         0.         0.11239704]]
         [[0.         0.         0.01792819]
          [0.         0.         0.00558351]]]

        Compute the field of a sensor along a path. The path positions are chosen so that
        they coincide with the pixel positions in the previous example.

        >>> import magpylib as mag3
        >>> loop = mag3.current.Circular(current=1, diameter=1)
        >>> sensor = mag3.Sensor()
        >>> sensor.move([(0,0,1)]*3, start=1, increment=True)
        >>> print(sensor.position)
        [[0. 0. 0.]
         [0. 0. 1.]
         [0. 0. 2.]
         [0. 0. 3.]]
        >>> B = sensor.getB(loop)
        >>> print(B)
        [[0.         0.         1.25663706]
         [0.         0.         0.11239704]
         [0.         0.         0.01792819]
         [0.         0.         0.00558351]]

        """
        sources = format_star_input(sources)
        return getBH_level2(True, sources, self, sumup, squeeze)


    def getH(self, *sources, sumup=False, squeeze=True):
        """
        Compute H-field in [kA/m] for given sources.

        Parameters
        ----------
        sources: source objects or Collections
            Sources can be a mixture of L source objects or Collections.

        sumup: bool, default=False
            If True, the fields of all sources are summed up.

        squeeze: bool, default=True
            If True, the output is squeezed, i.e. all axes of length 1 in the output (e.g. only
            a single sensor or only a single source) are eliminated.

        Returns
        -------
        H-field: ndarray, shape squeeze(L, M, N1, N2, ..., 3)
            H-field of each source (L) at each path position (M) and each sensor pixel
            position (N1,N2,...) in units of [kA/m]. Paths of objects that are shorter than
            M will be considered as static beyond their end.

        Examples
        --------

        Sensors are observers for magnetic field computation. In this example we compute the
        H-field as seen by the sensor in the center of a circular current loop:

        >>> import magpylib as mag3
        >>> sensor = mag3.Sensor()
        >>> loop = mag3.current.Circular(current=1, diameter=1)
        >>> H = sensor.getH(loop)
        >>> print(H)
        [0. 0. 1.]

        Field computation is performed at every pixel of a sensor. The above example is reproduced
        for a 2x2-pixel sensor:

        >>> import magpylib as mag3
        >>> sensor = mag3.Sensor(pixel=[[(0,0,0), (0,0,1)],[(0,0,2), (0,0,3)]])
        >>> loop = mag3.current.Circular(current=1, diameter=1)
        >>> H = sensor.getH(loop)
        >>> print(H.shape)
        (2, 2, 3)
        >>> print(H)
        [[[0.         0.         1.        ]
          [0.         0.         0.08944272]]
         [[0.         0.         0.0142668 ]
          [0.         0.         0.00444322]]]

        Compute the field of a sensor along a path. The path positions are chosen so that
        they coincide with the pixel positions in the previous example.

        >>> import magpylib as mag3
        >>> loop = mag3.current.Circular(current=1, diameter=1)
        >>> sensor = mag3.Sensor()
        >>> sensor.move([(0,0,1)]*3, start=1, increment=True)
        >>> print(sensor.position)
        [[0. 0. 0.]
         [0. 0. 1.]
         [0. 0. 2.]
         [0. 0. 3.]]
        >>> H = sensor.getH(loop)
        >>> print(H)
        [[0.         0.         1.        ]
         [0.         0.         0.08944272]
         [0.         0.         0.0142668 ]
         [0.         0.         0.00444322]]

        """
        sources = format_star_input(sources)
        return getBH_level2(False, sources, self, sumup, squeeze)

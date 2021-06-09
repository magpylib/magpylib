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
    Magnetic field sensor. Can be used as observer input.

    Reference position: Origin of the local CS of the sensor.

    Reference orientation: Local and global CS coincide at initialization.

    Parameters
    ----------
    position: array_like, shape (3,) or (M,3), default=(0,0,0)
        Reference position in the global CS in units of [mm]. For M>1, the
        position attribute represents a path in the global CS. The attributes
        orientation and position must always be of the same length.

    pixel: array_like, shape (3,) or (N1,N2,...,3), default=(0,0,0)
        Sensor pixel positions inside of a sensor package. Positions are given in
        the local Sensor CS.

    orientation: scipy Rotation object with length 1 or M, default=unit rotation
        Orientation relative to the reference orientation. For M>1 orientation
        represents different values along a path. The attributes orientation and
        position must always be of the same length.

    Returns
    -------
    Sensor object: Sensor
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
        """
        sources = format_star_input(sources)
        return getBH_level2(False, sources, self, sumup, squeeze)

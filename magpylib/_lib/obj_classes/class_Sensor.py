"""Sensor class code"""

import numpy as np
from magpylib._lib.obj_classes.class_BaseGeo import BaseGeo
from magpylib._lib.fields import getB, getH

class Sensor(BaseGeo):
    """ sensor class........
    """

    def __init__(
            self,
            pos_pix=(0,0,0),
            pos = (0,0,0),
            rot = None):

        # inherit base_geo class
        BaseGeo.__init__(self, pos, rot)

        # set mag and dim attributes
        self.pos_pix = pos_pix

    # properties ----------------------------------------------------

    @property
    def pos_pix(self):
        """ Pixel pos in sensor CS

        Returns
        -------
        sensor pixel positions: np.array, shape (3,) or (N1,N2,...,3)
        """
        return self._pos_pix


    @pos_pix.setter
    def pos_pix(self, inp):
        """ set sensor pixel positions

        inp: array_like, shape (3,) or (N1,N2,...,3)
            set pixel positions in sensor CS
        """
        inp = np.array(inp, dtype=float)       # secure input
        self._pos_pix = inp


    # dunders -------------------------------------------------------
    def __repr__(self) -> str:
        return 'Sensor (' + str(id(self)) + ')'


    # methods -------------------------------------------------------
    def getB(self, sources):
        """ Compute B-field of sources at sensor.

        Parameters
        ----------
        sources: src_obj or list of src_obj
            Source object or a 1D list of L source objects and collections. Pathlength of
            all sources must be the same (or 1). Pathlength=1 sources will be considered
            as static.

        niter (int): Number of iterations in the computation of the
            diametral component of the field

        Returns
        -------
        B-field: ndarray, shape (L, M, K, N1, N2, ..., 3), unit [mT]
            B-field of each source at each path position for each sensor and each sensor pixel
            position in units of mT.
            Output is squeezed, i.e. every dimension of length 1 (single source or sumup=True or
            single sensor or no sensor) is removed.
        """
        B = getB(sources, self)
        return B


    def getH(self, sources):
        """ Compute H-field of sources at sensor.

        Parameters
        ----------
        sources: src_obj or list of src_obj
            Source object or a 1D list of L source objects and collections. Pathlength of
            all sources must be the same (or 1). Pathlength=1 sources will be considered
            as static.

        niter (int): Number of iterations in the computation of the
            diametral component of the field

        Returns
        -------
        H-field: ndarray, shape (L, M, K, N1, N2, ..., 3), unit [kA/m]
            H-field of each source at each path position for each sensor and each sensor pixel
            position in units of kA/m.
            Output is squeezed, i.e. every dimension of length 1 (single source or sumup=True or
            single sensor or no sensor) is removed.
        """
        H = getH(sources, self)
        return H

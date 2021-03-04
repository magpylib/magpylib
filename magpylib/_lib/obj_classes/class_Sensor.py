"""Sensor class code"""

import numpy as np
from magpylib._lib.obj_classes.class_BaseGeo import BaseGeo
from magpylib._lib.fields import getB_from_sensor, getH_from_sensor

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

    # methods -------------------------------------------------------
    def getB(self, sources):
        return getB_from_sensor(sources, self)

    def getH(self, sources):
        return getH_from_sensor(sources, self)
"""Sensor class code"""

import numpy as np
from magpylib3._lib.obj_classes.class_BaseGeo import BaseGeo


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

        inp = np.array(inp, dtype=np.float)       # secure input
        self._pos_pix = inp


    # dunders -------------------------------------------------------

    # methods -------------------------------------------------------

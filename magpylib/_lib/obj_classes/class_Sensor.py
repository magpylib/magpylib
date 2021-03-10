"""Sensor class code"""

import numpy as np
from magpylib._lib.obj_classes.class_BaseGeo import BaseGeo
from magpylib._lib.fields import getB, getH


# ON INTERFACE
class Sensor(BaseGeo):
    """
    3D Magnetic field sensor.

    init_state: The axes of the sensor are parallel to the global CS axes.

    Properties
    ----------
    pos_pix: array_like, shape (3,) or (N1,N2,...,3), default=(0,0,0)
        Sensor pixel inside of 'package'. Positions are given in local sensor CS.
        getBH computations return the field at the sensor pixels.

    pos: array_like, shape (3,) or (N,3), default=(0,0,0), unit [mm]
        Position of geometric center of Sensor in units of [mm]. For N>1 pos respresents a path in
        in the global CS.

    rot: scipy Rotation object with length 1 or N, default=unit rotation
        Sensor rotation relative to the init_state. For N>1 rot represents different rotations
        along a position-path.

    Dunders
    -------
    __repr__:
        returns string "Sensor(id(self))"

    Methods
    -------
    getB(sources):
        Compute B-field of sources at Sensor.

    getH(sources):
        Compute H-field of sources at Sensor.

    display(markers=[(0,0,0)], axis=None, direc=False, show_path=True):
        Display Sensor graphically using Matplotlib.

    move_by(displacement, steps=None):
        Linear displacement of Sensor by argument vector.

    move_to(target_pos, steps=None):
        Linear motion of Sensor to target_pos.

    rotate(rot, anchor=None, steps=None):
        Rotate Sensor about anchor.

    rotate_from_angax(angle, axis, anchor=None, steps=None, degree=True):
        Sensor rotation from angle-axis-anchor input.

    reset_path():
        Set Sensor.pos to (0,0,0) and Sensor.rot to unit rotation.

    Returns
    -------
    Sensor object
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
        """
        Pixel pos in Sensor CS.

        Returns
        -------
        Sensor pixel positions: np.array, shape (3,) or (N1,N2,...,3)
        """
        return self._pos_pix


    @pos_pix.setter
    def pos_pix(self, inp):
        """
        Set Sensor pixel positions.

        inp: array_like, shape (3,) or (N1,N2,...,3)
            Set pixel positions in Sensor CS.
        """
        inp = np.array(inp, dtype=float)       # secure input
        self._pos_pix = inp


    # dunders -------------------------------------------------------
    def __repr__(self) -> str:
        return f'Sensor({str(id(self))})'


    # methods -------------------------------------------------------
    def getB(self, sources, sumup=False, **specs):
        """
        Compute B-field of sources at Sensor.

        Parameters
        ----------
        sources: src objects, Collections or arbitrary lists thereof
            Source object or a 1D list of L source objects and/or collections. Pathlength of all
            sources must be M or 1. Sources with Pathlength=1 will be considered as static.

        sumup: bool, default=False
            If True, the field of all sources is summed up.

        Specific kwargs
        ---------------
        niter: int, default=50
            Diametral iterations (Simpsons formula) for Cylinder Sources integral computation.

        Returns
        -------
        B-field: ndarray, shape squeeze(L, M, N1, N2, ..., 3), unit [mT]
            B-field of each source (L) at each path position (M) and each sensor pixel position (N)
            in units of [mT].
            Output is squeezed, i.e. every dimension of length 1 (single source or sumup=True or
            or single pixel) is removed.
        """
        B = getB(sources, self, sumup=sumup, **specs)
        return B


    def getH(self, sources, sumup=False, **specs):
        """
        Compute H-field of sources at Sensor.

        Parameters
        ----------
        sources: src objects, Collections or arbitrary lists thereof
            Source object or a 1D list of L source objects and/or collections. Pathlength of all
            sources must be M or 1. Sources with Pathlength=1 will be considered as static.

        sumup: bool, default=False
            If True, the field of all sources is summed up.

        Specific kwargs
        ---------------
        niter: int, default=50
            Diametral iterations (Simpsons formula) for Cylinder Sources integral computation.

        Returns
        -------
        H-field: ndarray, shape squeeze(L, M, N1, N2, ..., 3), unit [kA/m]
            H-field of each source (L) at each path position (M) and each sensor pixel position (N)
            in units of [kA/m].
            Output is squeezed, i.e. every dimension of length 1 (single source or sumup=True or
            or single pixel) is removed.
        """
        H = getH(sources, self, sumup=sumup, **specs)
        return H

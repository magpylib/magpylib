"""Magnet Cylinder class code"""

import numpy as np
from magpylib._lib.fields import getB, getH
from magpylib._lib.obj_classes.class_BaseGeo import BaseGeo
from magpylib._lib.obj_classes.class_Collection import Collection
from magpylib._lib.exceptions import MagpylibBadUserInput

# init for tool tips
d=h=None
mx=my=mz=None

class Cylinder(BaseGeo):
    """ Homogeneous cylinder magnet.

    init_state: the geometric center is in the CS origin, the axis of
        the cylinder coincides with the z-axis.

    Properties
    ----------
    mag: array_like, shape (3,)
        Homogeneous magnet magnetization vector (remanence field) in units of [mT].

    dim: array_like, shape (2,)
        Dimension/Size of the cylinder with diameter/height [d,h] in units of mm.

    pos: array_like, shape (3,) or (N,3), default=(0,0,0)
        Position of geometric center of magnet in units of [mm].

    rot: scipy Rotation object, default=unit rotation
        Source rotations relative to the init_state.

    Dunders
    -------
    __add__:
        Adding sources creates a collection "col = src1 + src2"

    __repr__:
        returns string "Cylinder (id(self))"

    Methods
    -------
    getB: observers, niter=50
        Compute B-field of source at observer positions.

    getH: observers, niter=50
        Compute H-field of source at observer positions.

    display:
        Display source graphically.

    move_by:
        Linear displacement of source by input vector.

    move_to: array_like, shape (3,)
        Linear displacement of source to target position.

    rotate: scipy Rotation object
        Rotate source.

    rotate_from_angax: angle(float), axis(array_like, shape(3,))
        Rotate source with angle-axis input.

    Returns
    -------
    Cylinder object
    """

    def __init__(
            self,
            mag = (mx,my,mz),
            dim = (d,h),
            pos = (0,0,0),
            rot = None):

        # inherit base_geo class
        BaseGeo.__init__(self, pos, rot)

        # set mag and dim attributes
        self.mag = mag
        self.dim = dim

    # properties ----------------------------------------------------
    @property
    def mag(self):
        """ magnet magnetization in mT
        """
        return self._mag


    @mag.setter
    def mag(self, value):
        """ set magnetization vector, vec3, mT
        """
        if None in value:
            raise MagpylibBadUserInput('Magnetization input required')
        self._mag = np.array(value,dtype=float)


    @property
    def dim(self):
        """ cylinder dimension (d,h) in mm
        """
        return self._dim


    @dim.setter
    def dim(self, value):
        """ set cylinder dimension (d,h), vec2, mm
        """
        if None in value:
            raise MagpylibBadUserInput('Dimension input required')
        self._dim = np.array(value,dtype=float)


    # dunders -------------------------------------------------------
    def __add__(self, sources):
        """ sources add up to a Collection object
        """
        return Collection(self,sources)

    def __repr__(self) -> str:
        return f'Cylinder ({str(id(self))})'


    # methods -------------------------------------------------------
    def getB(self, observers, niter=50):
        """ Compute B-field of magnet at observer positions.

        Parameters
        ----------
        observers: array_like or sens_obj or list of sens_obj
            Observers can be array_like positions of shape (N1, N2, ..., 3) or a sensor or
            a 1D list of K sensors with pos_pix shape of (N1, N2, ..., 3)
            in units of millimeters.

        niter (int): Number of iterations in the computation of the
            diametral component of the field

        Returns
        -------
        B-field: ndarray, shape (M, K, N1, N2, ..., 3), unit [mT]
            B-field of magnet at each path position M for each sensor K and each sensor pixel
            position N in units of mT.
            Output is squeezed, i.e. every dimension of length 1 (single sensor or no sensor)
            is removed.
        """
        B = getB(self, observers, niter=niter)
        return B


    def getH(self, observers, niter=50):
        """ Compute H-field of magnet at observer positions.

        Parameters
        ----------
        observers: array_like or sens_obj or list of sens_obj
            Observers can be array_like positions of shape (N1, N2, ..., 3) or a sensor or
            a 1D list of K sensors with pos_pix shape of (N1, N2, ..., 3)
            in units of millimeters.

        niter (int): Number of iterations in the computation of the
            diametral component of the field

        Returns
        -------
        H-field: ndarray, shape (M, K, N1, N2, ..., 3), unit [kA/m]
            B-field of magnet at each path position M for each sensor K and each sensor pixel
            position N in units of kA/m.
            Output is squeezed, i.e. every dimension of length 1 (single sensor or no sensor)
            is removed.
        """
        H = getH(self, observers, niter=niter)
        return H

"""Magnet Cylinder class code"""

import numpy as np
from magpylib._lib.fields import getB, getH
from magpylib._lib.obj_classes.class_BaseGeo import BaseGeo
from magpylib._lib.obj_classes.class_BaseDisplay import BaseDisplay
from magpylib._lib.exceptions import MagpylibBadUserInput, MagpylibBadInputShape
from magpylib._lib.utility import format_getBH_class_inputs
from magpylib._lib.config import Config

# init for tool tips
d=h=None
mx=my=mz=None


# ON INTERFACE
class Cylinder(BaseGeo, BaseDisplay):
    """
    Cylinder magnet with homogeneous magnetization.

    init_state: the geometric center is in the global CS origin, the axis of the
        cylinder coincides  with the z-axis.

    Properties
    ----------
    mag: array_like, shape (3,), unit [mT]
        Magnetization vector (remanence field) in units of [mT].

    dim: array_like, shape (2,), unit [mm]
        Dimension/Size of the Cylinder with diameter/height (d,h) in units of [mm].

    pos: array_like, shape (3,) or (N,3), default=(0,0,0), unit [mm]
        Position of Cylinder center in units of [mm]. For N>1 pos respresents a path in
        in the global CS.

    rot: scipy Rotation object with length 1 or N, default=unit rotation
        Source rotation relative to the init_state. For N>1 rot represents different rotations
        along a position-path.

    Dunders
    -------

    __add__:
        Adding sources creates a Collection "col = src1 + src2"

    __repr__:
        returns string "Cylinder(id)"

    Methods
    -------
    getB(observers):
        Compute B-field of Cylinder at observers.

    getH(observers):
        Compute H-field of Cylinder at observers.

    display(markers=[(0,0,0)], axis=None, direc=False, show_path=True):
        Display Cylinder graphically using Matplotlib.

    move_by(displacement, steps=None):
        Linear displacement of Cylinder by argument vector.

    move_to(target_pos, steps=None):
        Linear motion of Cylinder to target_pos.

    rotate(rot, anchor=None, steps=None):
        Rotate Cylinder about anchor.

    rotate_from_angax(angle, axis, anchor=None, steps=None, degree=True):
        Cylinder rotation from angle-axis-anchor input.

    reset_path():
        Set Cylinder.pos to (0,0,0) and Cylinder.rot to unit rotation.

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
        """ Magnet magnetization in units of [mT].
        """
        return self._mag


    @mag.setter
    def mag(self, magnetization):
        """ Set magnetization vector, shape (3,), unit [mT].
        """
        # input check
        if Config.CHECK_INPUTS:
            if None in magnetization:
                raise MagpylibBadUserInput('Magnetization input required')

        # secure type
        magnetization = np.array(magnetization,dtype=float)

        # input check
        if Config.CHECK_INPUTS:
            if magnetization.shape != (3,):
                raise MagpylibBadInputShape('Bad magnetization input shape.')

        self._mag = magnetization


    @property
    def dim(self):
        """ Cylinder dimension (d,h), unit [mm]
        """
        return self._dim

    @dim.setter
    def dim(self, dimension):
        """ Set cylinder dimension (d,h), shape (2,), unit [mm]
        """
        # input check
        if Config.CHECK_INPUTS:
            if None in dimension:
                raise MagpylibBadUserInput('Dimension input required')

        dimension = np.array(dimension,dtype=float)

        # input check
        if Config.CHECK_INPUTS:
            if dimension.shape != (2,):
                raise MagpylibBadInputShape('Bad dimension input shape.')

        self._dim = dimension


    # dunders -------------------------------------------------------
    def __repr__(self) -> str:
        return f'Cylinder({str(id(self))})'


    # methods -------------------------------------------------------
    def getB(self, *observers, squeeze=True, niter=50):
        """
        Compute B-field of source at observers.

        Parameters
        ----------
        observers: array_like or Sensor or list of Sensors
            Observers can be array_like positions of shape (N1, N2, ..., 3) or a Sensor object or
            a 1D list of K Sensor objects with pixel position shape of (N1, N2, ..., 3) in units
            of [mm].

        squeeze: bool, default=True
            If True, the output is squeezed, i.e. all axes of length 1 in the output (e.g. only
            a single sensor or only a single source) are eliminated.

        niter: int, default=50
            Diametral iterations (Simpsons formula) for Cylinder Sources integral computation.

        Returns
        -------
        B-field: ndarray, shape squeeze(M, K, N1, N2, ..., 3), unit [mT]
            B-field at each path position (M) for each sensor (K) and each sensor pixel position
            (N) in units of [mT].
            Output is squeezed, i.e. every dimension of length 1 (single sensor or no sensor or
            single pixel) is removed.
        """
        observers = format_getBH_class_inputs(observers)
        B = getB(self, observers, squeeze=squeeze, niter=niter)
        return B


    def getH(self, *observers, squeeze=True, niter=50):
        """ Compute H-field of magnet at observer positions.

        Parameters
        ----------
        observers: array_like or sens_obj or list of sens_obj
            Observers can be array_like positions of shape (N1, N2, ..., 3) or a sensor or
            a 1D list of K sensors with pos_pix shape of (N1, N2, ..., 3)
            in units of millimeters.

        squeeze: bool, default=True
            If True, the output is squeezed, i.e. all axes of length 1 in the output (e.g. only
            a single sensor or only a single source) are eliminated.

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
        observers = format_getBH_class_inputs(observers)
        H = getH(self, observers, squeeze=squeeze, niter=niter)
        return H

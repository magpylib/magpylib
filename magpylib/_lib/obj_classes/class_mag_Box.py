"""Magnet Box class code"""

import numpy as np
from magpylib._lib.fields import getB, getH
from magpylib._lib.obj_classes.class_BaseGeo import BaseGeo
from magpylib._lib.exceptions import MagpylibBadUserInput, MagpylibBadInputShape
from magpylib._lib.utility import format_getBH_class_inputs
from magpylib._lib.config import Config

# init for tool tips
a=b=c=None
mx=my=mz=None


# ON INTERFACE
class Box(BaseGeo):
    """
    Cuboid magnet with homogeneous magnetization.

    init_state: The geometric center is in the global CS origin and the sides of the Box are
    parallel to the x/y/z basis vectors.

    Properties
    ----------
    mag: array_like, shape (3,), unit [mT]
        Magnetization vector (remanence field) in units of [mT].

    dim: array_like, shape (3,), unit [mm]
        Dimension/Size of the Box with sides [a,b,c] in units of [mm].

    pos: array_like, shape (3,) or (N,3), default=(0,0,0), unit [mm]
        Position of Box center in units of [mm]. For N>1 pos respresents a path in
        in the global CS.

    rot: scipy Rotation object with length 1 or N, default=unit rotation
        Source rotation relative to the init_state. For N>1 rot represents different rotations
        along a position-path.

    Dunders
    -------

    __add__:
        Adding sources creates a Collection "col = src1 + src2"

    __repr__:
        returns string "Box(id)"

    Methods
    -------
    getB(observers):
        Compute B-field of Box at observers.

    getH(observers):
        Compute H-field of Box at observers.

    display(markers=[(0,0,0)], axis=None, direc=False, show_path=True):
        Display Box graphically using Matplotlib.

    move_by(displacement, steps=None):
        Linear displacement of Box by argument vector.

    move_to(target_pos, steps=None):
        Linear motion of Box to target_pos.

    rotate(rot, anchor=None, steps=None):
        Rotate Box about anchor.

    rotate_from_angax(angle, axis, anchor=None, steps=None, degree=True):
        Box rotation from angle-axis-anchor input.

    reset_path():
        Set Box.pos to (0,0,0) and Box.rot to unit rotation.

    Returns
    -------
    Box object
    """

    def __init__(
            self,
            mag = (mx,my,mz),
            dim = (a,b,c),
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
        """ Box dimension (a,b,c) in [mm].
        """
        return self._dim

    @dim.setter
    def dim(self, dimension):
        """ Set Box dimension (a,b,c), shape (3,), [mm].
        """
        # input check
        if Config.CHECK_INPUTS:
            if None in dimension:
                raise MagpylibBadUserInput('Dimension input required')

        dimension = np.array(dimension,dtype=float)

        # input check
        if Config.CHECK_INPUTS:
            if dimension.shape != (3,):
                raise MagpylibBadInputShape('Bad dimension input shape.')

        self._dim = dimension


    # dunders -------------------------------------------------------
    def __repr__(self) -> str:
        return f'Box({str(id(self))})'


    # methods -------------------------------------------------------
    def getB(self, *observers, squeeze=True):
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

        Returns
        -------
        B-field: ndarray, shape squeeze(M, K, N1, N2, ..., 3), unit [mT]
            B-field at each path position (M) for each sensor (K) and each sensor pixel position
            (N) in units of [mT].
            Output is squeezed, i.e. every dimension of length 1 (single sensor or no sensor or
            single pixel) is removed.
        """
        observers = format_getBH_class_inputs(observers)
        B = getB(self, observers, False, squeeze)
        return B


    def getH(self, *observers, squeeze=True):
        """
        Compute H-field of source at observers.

        Parameters
        ----------
        observers: array_like or Sensor or list of Sensors
            Observers can be array_like positions of shape (N1, N2, ..., 3) or a Sensor object or
            a 1D list of K Sensor objects with pixel position shape of (N1, N2, ..., 3) in units
            of [mm].

        squeeze: bool, default=True
            If True, the output is squeezed, i.e. all axes of length 1 in the output (e.g. only
            a single sensor or only a single source) are eliminated.

        Returns
        -------
        H-field: ndarray, shape squeeze(M, K, N1, N2, ..., 3), unit [kA/m]
            B-field at each path position (M) for each sensor (K) and each sensor pixel position
            (N) in units of [kA/m].
            Output is squeezed, i.e. every dimension of length 1 (single sensor or no sensor or
            single pixel) is removed.
        """
        observers = format_getBH_class_inputs(observers)
        H = getH(self, observers, False, squeeze)
        return H

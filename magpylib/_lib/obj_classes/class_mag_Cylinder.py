"""Magnet Cylinder class code"""

import numpy as np
from magpylib._lib.fields import getB, getH
from magpylib._lib.obj_classes.class_BaseGeo import BaseGeo
from magpylib._lib.exceptions import MagpylibBadUserInput
from magpylib._lib.utility import format_getBH_class_inputs

# init for tool tips
d=h=None
mx=my=mz=None


# ON INTERFACE
class Cylinder(BaseGeo):
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
        Position of geometric center of magnet in units of [mm]. For N>1 pos respresents a path in
        in the global CS.

    rot: scipy Rotation object with length 1 or N, default=unit rotation
        Source rotation relative to the init_state. For N>1 rot represents different rotations
        along a position-path.

    Dunders
    -------

    __add__:
        Adding sources creates a collection "col = src1 + src2"

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
            self,mag = (mx,my,mz),dim=(d,h),
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
    def mag(self, value):
        """ Set magnetization vector, shape (3,), unit [mT].
        """
        if None in value:
            raise MagpylibBadUserInput('Magnetization input required')
        self._mag = np.array(value,dtype=float)


    @property
    def dim(self):
        """ Cylinder dimension (d,h), unit [mm]
        """
        return self._dim


    @dim.setter
    def dim(self, value):
        """ Set cylinder dimension (d,h), shape (2,), unit [mm]
        """
        if None in value:
            raise MagpylibBadUserInput('Dimension input required')
        self._dim = np.array(value,dtype=float)


    # dunders -------------------------------------------------------

    def __repr__(self) -> str:
        return f'Cylinder({str(id(self))})'


    # methods -------------------------------------------------------
    def getB(self, *observers, niter=50):
        """
        Compute B-field of source at observers.

        Parameters
        ----------
        observers: array_like or Sensor or list of Sensors
            Observers can be array_like positions of shape (N1, N2, ..., 3) or a Sensor object or
            a 1D list of K Sensor objects with pixel position shape of (N1, N2, ..., 3) in units
            of [mm].

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
        B = getB(self, observers, niter=niter)
        return B


    def getH(self, *observers, niter=50):
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
        observers = format_getBH_class_inputs(observers)
        H = getH(self, observers, niter=niter)
        return H

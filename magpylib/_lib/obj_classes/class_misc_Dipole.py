"""Dipole class code"""

import numpy as np
from magpylib._lib.fields import getB, getH
from magpylib._lib.obj_classes.class_BaseGeo import BaseGeo
from magpylib._lib.obj_classes.class_BaseDisplay import BaseDisplay
from magpylib._lib.exceptions import MagpylibBadUserInput, MagpylibBadInputShape
from magpylib._lib.utility import format_getBH_class_inputs
from magpylib._lib.config import Config

# init for tool tips
mx=my=mz=None


# ON INTERFACE
class Dipole(BaseGeo, BaseDisplay):
    """
    Magnetic dipole moment.

    Properties
    ----------
    moment: array_like, shape (3,), unit [mT*mm^3]
        Magnetic dipole moment in units of [mT*mm^3]. For homogeneous magnets the
        relation is moment=magnetization*volume.

    pos: array_like, shape (3,) or (N,3), default=(0,0,0), unit [mm]
        Position of Sphere center in units of [mm]. For N>1 pos respresents a path in
        in the global CS.

    rot: scipy Rotation object with length 1 or N, default=unit rotation
        Source rotation relative to the init_state. For N>1 rot represents different rotations
        along a position-path.

    Dunders
    -------

    __add__:
        Adding sources creates a Collection "col = src1 + src2"

    __repr__:
        returns string "Dipole(id)"

    Methods
    -------
    getB(observers):
        Compute B-field of Dipole at observers.

    getH(observers):
        Compute H-field of Dipole at observers.

    display(markers=[(0,0,0)], axis=None, direc=False, show_path=True):
        Display Dipole graphically using Matplotlib.

    move_by(displacement, steps=None):
        Linear displacement of Dipole by argument vector.

    move_to(target_pos, steps=None):
        Linear motion of Dipole to target_pos.

    rotate(rot, anchor=None, steps=None):
        Rotate Dipole about anchor.

    rotate_from_angax(angle, axis, anchor=None, steps=None, degree=True):
        Dipole rotation from angle-axis-anchor input.

    reset_path():
        Set Dipole.pos to (0,0,0) and Dipole.rot to unit rotation.

    Returns
    -------
    Dipole object
    """

    def __init__(
            self,
            moment = (mx,my,mz),
            pos = (0,0,0),
            rot = None):

        # inherit base_geo class
        BaseGeo.__init__(self, pos, rot)

        # set moment attribute using setter
        self.moment = moment

    # properties ----------------------------------------------------
    @property
    def moment(self):
        """ Dipole moment in units of [mT*mm^3].
        """
        return self._moment

    @moment.setter
    def moment(self, mom):
        """ Set dipole moment vector, shape (3,), unit [mT*mm^3].
        """
        # input check
        if Config.CHECK_INPUTS:
            if None in mom:
                raise MagpylibBadUserInput('Moment input required')

        # secure type
        mom = np.array(mom, dtype=float)

        # input check
        if Config.CHECK_INPUTS:
            if mom.shape != (3,):
                raise MagpylibBadInputShape('Bad dipole moment input shape.')

        self._moment = mom


    # dunders -------------------------------------------------------

    def __repr__(self) -> str:
        return f'Dipole({str(id(self))})'


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
        print('woot')
        observers = format_getBH_class_inputs(observers)
        B = getB(self, observers, squeeze=squeeze)
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
        H = getH(self, observers, squeeze=squeeze)
        return H

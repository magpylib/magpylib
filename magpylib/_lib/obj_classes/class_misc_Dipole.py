"""Dipole class code"""

import numpy as np
from magpylib._lib.obj_classes.class_BaseGeo import BaseGeo
from magpylib._lib.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._lib.obj_classes.class_BaseGetBH import BaseGetBH
from magpylib._lib.config import Config
from magpylib._lib.input_checks import (check_vector_format, check_vector_type,
    check_vector_init)

# init for tool tips
mx=my=mz=None

# ON INTERFACE
class Dipole(BaseGeo, BaseDisplayRepr, BaseGetBH):
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
            position = (0,0,0),
            orientation = None):

        # inherit base_geo class
        BaseGeo.__init__(self, position, orientation)
        BaseDisplayRepr.__init__(self)

        # set moment attribute using setter
        self.moment = moment
        self.obj_type = 'Dipole'

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
        # input type check
        if Config.CHECK_INPUTS:
            check_vector_type(mom, 'moment')
            check_vector_init(mom, 'moment')

        # secure type
        mom = np.array(mom, dtype=float)

        # input format check
        if Config.CHECK_INPUTS:
            check_vector_format(mom, (3,), 'moment')

        self._moment = mom

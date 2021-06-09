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

    Reference position: Dipole position.

    Reference orientation: Local and global CS coincide at initialization.

    Parameters
    ----------
    moment: array_like, shape (3,), unit [mT*mm^3]
        Magnetic dipole moment in units of [mT*mm^3] given in the local CS of the
        Dipole object. For homogeneous magnets there is a relation
        moment=magnetization*volume.

    position: array_like, shape (3,) or (M,3), default=(0,0,0)
        Reference position in the global CS in units of [mm]. For M>1, the
        position attribute represents a path in the global CS. The attributes
        orientation and position must always be of the same length.

    orientation: scipy Rotation object with length 1 or M, default=unit rotation
        Orientation relative to the reference orientation. For M>1 orientation
        represents different values along a path. The attributes orientation and
        position must always be of the same length.

    Returns
    -------
    Dipole object: Dipole
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
        self.object_type = 'Dipole'

    # properties ----------------------------------------------------
    @property
    def moment(self):
        """ Object moment attributes getter and setter.
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

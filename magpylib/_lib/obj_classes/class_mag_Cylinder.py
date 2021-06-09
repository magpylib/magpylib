"""Magnet Cylinder class code"""

import numpy as np
from magpylib._lib.obj_classes.class_BaseGeo import BaseGeo
from magpylib._lib.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._lib.obj_classes.class_BaseGetBH import BaseGetBH
from magpylib._lib.obj_classes.class_BaseExcitations import BaseHomMag
from magpylib._lib.config import Config
from magpylib._lib.input_checks import check_vector_format, check_vector_init, check_vector_type

# init for tool tips
d=h=None
mx=my=mz=None

# ON INTERFACE
class Cylinder(BaseGeo, BaseDisplayRepr, BaseGetBH, BaseHomMag):
    """
    Cylinder magnet with homogeneous magnetization.

    Reference position: Geometric center of the Cylinder.

    Reference orientation: Cylinder axis coincides with the z-axis of the global CS.

    By default ITER_CYLINDER=50 for iteration of diametral magnetization computation.
    Use ``magpylib.Config.ITER_CYLINDER=X`` to change this setting.

    Parameters
    ----------
    magnetization: array_like, shape (3,)
        Magnetization vector (mu0*M, remanence field) in units of [mT] given in
        the local CS of the Cylinder object.

    dimension: array_like, shape (2,)
        Dimension/Size of the Cylinder with diameter/height (d,h) in units of [mm].

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
    Cylinder object: Cylinder
    """

    def __init__(
            self,
            magnetization = (mx,my,mz),
            dimension = (d,h),
            position = (0,0,0),
            orientation = None):

        # inherit base_geo class
        BaseGeo.__init__(self, position, orientation)
        BaseDisplayRepr.__init__(self)
        BaseHomMag.__init__(self, magnetization)

        # set attributes
        self.dimension = dimension
        self.object_type = 'Cylinder'

    @property
    def dimension(self):
        """ Object dimension attribute getter and setter.
        """
        return self._dimension

    @dimension.setter
    def dimension(self, dim):
        """ Set Cylinder dimension (d,h), shape (2,), [mm].
        """
        # input type and init check
        if Config.CHECK_INPUTS:
            check_vector_type(dim, 'dimension')
            check_vector_init(dim, 'dimension')

        # input type -> ndarray
        dim = np.array(dim,dtype=float)

        # input format check
        if Config.CHECK_INPUTS:
            check_vector_format(dim, (2,), 'dimension')

        self._dimension = dim

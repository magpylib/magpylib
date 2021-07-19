"""Magnet Cylinder class code"""

import numpy as np
from magpylib._lib.obj_classes.class_BaseGeo import BaseGeo
from magpylib._lib.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._lib.obj_classes.class_BaseGetBH import BaseGetBH
from magpylib._lib.obj_classes.class_BaseExcitations import BaseHomMag
from magpylib._lib.config import Config
from magpylib._lib.input_checks import check_input_cyl_sect, check_vector_type

# init for tool tips
d1=d2=h=phi1=phi2=None
mx=my=mz=None

# ON INTERFACE
class CylinderSection(BaseGeo, BaseDisplayRepr, BaseGetBH, BaseHomMag):
    """
    Cylinder-Section/Tile (Ring-Section) magnet with homogeneous magnetization. 

    Local object coordinates: The geometric center of the full Cylinder is located
    in the origin of the local object coordinate system. The Cylinder axis conincides
    with the local CS z-axis. Local (Cylinder) and global CS coincide when
    position=(0,0,0) and orientation=unit_rotation.

    Parameters
    ----------
    magnetization: array_like, shape (3,)
        Magnetization vector (mu0*M, remanence field) in units of [mT] given in
        the local CS of the Cylinder object.

    dimension: array_like, shape (5,)
        Dimension/Size of a Cylinder-Section (d1,d2,h,phi1,phi2) where d1 < d2 denote inner
        and outer diameter in units of [mm], phi1 < phi2 denote the Cylinder section angles
        in units of [deg] and h the Cylinder height in units of [mm].

    position: array_like, shape (3,) or (M,3), default=(0,0,0)
        Object position (local CS origin) in the global CS in units of [mm].
        For M>1, the position represents a path. The position and orientation
        parameters must always be of the same length.

    orientation: scipy Rotation object with length 1 or M, default=unit rotation
        Object orientation (local CS orientation) in the global CS. For M>1
        orientation represents different values along a path. The position and
        orientation parameters must always be of the same length.

    Returns
    -------
    CylinderSection object: CylinderSection

    Examples
    --------

    A Cylinder-Tile with inner diameter d1=2, outer diameter d2=4, spanning over 90 deg in the
    positive quadrant of the local CS and height h=5.

    >>> import magpylib as magpy
    >>> magnet = magpy.magnet.Cylinder(magnetization=(100,100,100), dimension=(2,4,5,0,90))
    >>> B = magnet.getB((1,2,3))
    >>> print(B)
    [-2.74825633  9.77282601 21.43280135]

    """

    def __init__(
            self,
            magnetization = (mx,my,mz),
            dimension = (d1,d2,h,phi1,phi2),
            position = (0,0,0),
            orientation = None):

        # init inheritance
        BaseGeo.__init__(self, position, orientation)
        BaseDisplayRepr.__init__(self)
        BaseHomMag.__init__(self, magnetization)

        # instance attributes
        self.dimension = dimension
        self._object_type = 'CylinderSection'

    # property getters and setters
    @property
    def dimension(self):
        """ Object dimension attribute getter and setter.
        """
        return self._dimension

    @dimension.setter
    def dimension(self, dim):
        """ Set Cylinder dimension (d1,d2,h,phi1,phi2), shape (5,), [mm, deg].
        """
        # input type check
        if Config.CHECK_INPUTS:
            check_vector_type(dim, 'dimension')

        # input type -> ndarray
        dim = np.array(dim,dtype=float)

        # input format check
        if Config.CHECK_INPUTS:
            check_input_cyl_sect(dim)

        self._dimension = dim

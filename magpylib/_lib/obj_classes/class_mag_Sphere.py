"""Magnet Sphere class code"""

from magpylib._lib.obj_classes.class_BaseGeo import BaseGeo
from magpylib._lib.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._lib.obj_classes.class_BaseGetBH import BaseGetBH
from magpylib._lib.obj_classes.class_BaseExcitations import BaseHomMag
from magpylib._lib.config import Config
from magpylib._lib.input_checks import check_scalar_init, check_scalar_type

# init for tool tips
mx=my=mz=None

# ON INTERFACE
class Sphere(BaseGeo, BaseDisplayRepr, BaseGetBH, BaseHomMag):
    """
    Spherical magnet with homogeneous magnetization.

    Reference position: Center of Sphere.

    Reference orientation: Local and global CS coincide at initialization.

    Parameters
    ----------
    magnetization: array_like, shape (3,)
        Magnetization vector (mu0*M, remanence field) in units of [mT] given in
        the local CS of the Sphere object.

    diameter: float
        Diameter of the Sphere in units of [mm].

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
    Sphere object: Sphere
    """

    def __init__(
            self,
            magnetization = (mx,my,mz),
            diameter = None,
            position = (0,0,0),
            orientation = None):

        # inherit base_geo class
        BaseGeo.__init__(self, position, orientation)
        BaseDisplayRepr.__init__(self)
        BaseHomMag.__init__(self, magnetization)

        # set attributes
        self.diameter = diameter
        self.object_type = 'Sphere'

    @property
    def diameter(self):
        """ Object diameter attribute getter and setter.
        """
        return self._diameter

    @diameter.setter
    def diameter(self, dia):
        """ Set Sphere diameter, float, [mm].
        """
        # input type and init check
        if Config.CHECK_INPUTS:
            check_scalar_init(dia, 'diameter')
            check_scalar_type(dia, 'diameter')

        self._diameter = float(dia)

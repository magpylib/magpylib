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

    Local object coordinates: The geometric center of the Cylinder is located
    in the origin of the local object coordinate system. The Cylinder axis conincides
    with the local CS z-axis. Local (Cylinder) and global CS coincide when
    position=(0,0,0) and orientation=unit_rotation.

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
        Object position (local CS origin) in the global CS in units of [mm].
        For M>1, the position represents a path. The position and orientation
        parameters must always be of the same length.

    orientation: scipy Rotation object with length 1 or M, default=unit rotation
        Object orientation (local CS orientation) in the global CS. For M>1
        orientation represents different values along a path. The position and
        orientation parameters must always be of the same length.

    Returns
    -------
    Cylinder object: Cylinder

    Examples
    --------
    By default a Cylinder is initialized at position (0,0,0), with unit rotation:

    >>> import magpylib as mag3
    >>> magnet = mag3.magnet.Cylinder(magnetization=(100,100,100), dimension=(1,1))
    >>> print(magnet.position)
    [0. 0. 0.]
    >>> print(magnet.orientation.as_quat())
    [0. 0. 0. 1.]

    Cylinders are magnetic field sources. Below we compute the H-field [kA/m] of the
    above Cylinder at the observer position (1,1,1),

    >>> H = magnet.getH((1,1,1))
    >>> print(H)
    [1.95851744 1.95851744 1.8657571 ]

    or at a set of observer positions:

    >>> H = magnet.getH([(1,1,1), (2,2,2), (3,3,3)])
    >>> print(H)
    [[1.95851744 1.95851744 1.8657571 ]
     [0.24025917 0.24025917 0.23767364]
     [0.07101874 0.07101874 0.07068512]]

    The same result is obtained when the Cylinder moves along a path,
    away from the observer:

    >>> magnet.move([(-1,-1,-1), (-2,-2,-2)], start=1)
    >>> H = magnet.getH((1,1,1))
    >>> print(H)
    [[1.95851744 1.95851744 1.8657571 ]
     [0.24025917 0.24025917 0.23767364]
     [0.07101874 0.07101874 0.07068512]]
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

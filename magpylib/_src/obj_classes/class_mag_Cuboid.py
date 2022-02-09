"""Magnet Cuboid class code"""

import numpy as np
from magpylib._src.obj_classes.class_BaseGeo import BaseGeo
from magpylib._src.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._src.obj_classes.class_BaseGetBH import BaseGetBH
from magpylib._src.obj_classes.class_BaseExcitations import BaseHomMag
from magpylib._src.defaults.defaults_classes import default_settings as Config
from magpylib._src.input_checks import check_vector_format, check_vector_type

# init for tool tips
a = b = c = None
mx = my = mz = None

# ON INTERFACE
class Cuboid(BaseGeo, BaseDisplayRepr, BaseGetBH, BaseHomMag):
    """
    Cuboid magnet with homogeneous magnetization.

    Local object coordinates: The geometric center of the Cuboid is located
    in the origin of the local object coordinate system. Cuboid sides are
    parallel to the local basis vectors. Local (Cuboid) and global CS coincide when
    position=(0,0,0) and orientation=unit_rotation.

    Parameters
    ----------
    magnetization: array_like, shape (3,)
        Magnetization vector (mu0*M, remanence field) in units of [mT] given in
        the local CS of the Cuboid object.

    dimension: array_like, shape (3,)
        Dimension/Size of the Cuboid with sides [a,b,c] in units of [mm].

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
    Cuboid object: Cuboid

    Examples
    --------
    By default a Cuboid is initialized at position (0,0,0), with unit rotation:

    >>> import magpylib as magpy
    >>> magnet = magpy.magnet.Cuboid(magnetization=(100,100,100), dimension=(1,1,1))
    >>> print(magnet.position)
    [0. 0. 0.]
    >>> print(magnet.orientation.as_quat())
    [0. 0. 0. 1.]

    Cuboids are magnetic field sources. Below we compute the H-field [kA/m] of the above Cuboid
    at the observer position (1,1,1),

    >>> H = magnet.getH((1,1,1))
    >>> print(H)
    [2.4844679 2.4844679 2.4844679]

    or at a set of observer positions:

    >>> H = magnet.getH([(1,1,1), (2,2,2), (3,3,3)])
    >>> print(H)
    [[2.4844679  2.4844679  2.4844679 ]
     [0.30499798 0.30499798 0.30499798]
     [0.0902928  0.0902928  0.0902928 ]]

    The same result is obtained when the Cuboid moves along a path,
    away from the observer:

    >>> magnet.move([(-1,-1,-1), (-2,-2,-2)], start=1)
    >>> H = magnet.getH((1,1,1))
    >>> print(H)
    [[2.4844679  2.4844679  2.4844679 ]
     [0.30499798 0.30499798 0.30499798]
     [0.0902928  0.0902928  0.0902928 ]]
    """

    def __init__(
        self,
        magnetization=(mx, my, mz),
        dimension=(a, b, c),
        position=(0, 0, 0),
        orientation=None,
        style=None,
        **kwargs,
    ):

        # instance attributes
        self.dimension = dimension
        self._object_type = "Cuboid"

        # init inheritance
        BaseGeo.__init__(self, position, orientation, style=style, **kwargs)
        BaseDisplayRepr.__init__(self)
        BaseHomMag.__init__(self, magnetization)

    # property getters and setters
    @property
    def dimension(self):
        """Object dimension attribute getter and setter."""
        return self._dimension

    @dimension.setter
    def dimension(self, dim):
        """Set Cuboid dimension (a,b,c), shape (3,), [mm]."""
        # input type check
        if Config.checkinputs:
            check_vector_type(dim, "dimension")

        # input type -> ndarray
        dim = np.array(dim, dtype=float)

        # input format check
        if Config.checkinputs:
            check_vector_format(dim, (3,), "dimension")

        self._dimension = dim

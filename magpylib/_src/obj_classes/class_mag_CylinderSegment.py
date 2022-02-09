"""Magnet Cylinder class code"""

import numpy as np
from magpylib._src.obj_classes.class_BaseGeo import BaseGeo
from magpylib._src.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._src.obj_classes.class_BaseGetBH import BaseGetBH
from magpylib._src.obj_classes.class_BaseExcitations import BaseHomMag
from magpylib._src.defaults.defaults_classes import default_settings as Config
from magpylib._src.input_checks import check_input_cyl_sect, check_vector_type

# init for tool tips
d1 = d2 = h = phi1 = phi2 = None
mx = my = mz = None

# ON INTERFACE
class CylinderSegment(BaseGeo, BaseDisplayRepr, BaseGetBH, BaseHomMag):
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
        Dimension/Size of a Cylinder-Section (r1,r2,h,phi1,phi2) where r1 < r2 denote inner
        and outer radius in units of [mm], phi1 < phi2 denote the Cylinder section angles
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
    CylinderSegment object: CylinderSegment

    Examples
    --------

    Initialize a CylinderSegment with inner diameter d1=1, outer diameter d2=2,
    height h=2 spanning over 90 deg in the positive quadrant. By default a CylinderSegment
    is initialized at position (0,0,0), which corresponds to the center of the full
    Cylinder, with unit rotation:

    >>> import magpylib as magpy
    >>> magnet = magpy.magnet.CylinderSegment(magnetization=(100,100,100), dimension=(0.5,1,2,0,90))
    >>> print(magnet.position)
    [0. 0. 0.]
    >>> print(magnet.orientation.as_quat())
    [0. 0. 0. 1.]

    CylinderSegments are magnetic field sources. Below we compute the H-field [kA/m] of the
    above CylinderSegment at the observer position (1,1,1),

    >>> H = magnet.getH((1,1,1))
    >>> print(H)
    [9.68649054 9.68649054 7.98245579]

    or at a set of observer positions:

    >>> H = magnet.getH([(1,1,1), (2,2,2), (3,3,3)])
    >>> print(H)
    [[9.68649054 9.68649054 7.98245579]
     [0.55368221 0.55368221 0.68274112]
     [0.13864524 0.13864524 0.16646942]]

    The same result is obtained when the Cylinder moves along a path,
    away from the observer:

    >>> magnet.move([(-1,-1,-1), (-2,-2,-2)], start=1)
    >>> H = magnet.getH((1,1,1))
    >>> print(H)
    [[9.68649054 9.68649054 7.98245579]
     [0.55368221 0.55368221 0.68274112]
     [0.13864524 0.13864524 0.16646942]]
    """

    def __init__(
        self,
        magnetization=(mx, my, mz),
        dimension=(d1, d2, h, phi1, phi2),
        position=(0, 0, 0),
        orientation=None,
        style=None,
        **kwargs,
    ):

        # instance attributes
        self.dimension = dimension
        self._object_type = "CylinderSegment"

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
        """Set Cylinder dimension (r1,r2,h,phi1,phi2), shape (5,), [mm, deg]."""
        # input type check
        if Config.checkinputs:
            check_vector_type(dim, "dimension")

        # input type -> ndarray
        dim = np.array(dim, dtype=float)

        # input format check
        if Config.checkinputs:
            check_input_cyl_sect(dim)

        self._dimension = dim

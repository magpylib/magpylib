"""Dipole class code"""

import numpy as np
from magpylib._src.obj_classes.class_BaseGeo import BaseGeo
from magpylib._src.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._src.obj_classes.class_BaseGetBH import BaseGetBH
from magpylib._src.defaults.defaults_classes import default_settings as Config
from magpylib._src.input_checks import check_vector_format, check_vector_type

# init for tool tips
mx = my = mz = None

# ON INTERFACE
class Dipole(BaseGeo, BaseDisplayRepr, BaseGetBH):
    """
    Magnetic dipole moment.

    Local object coordinates: The Dipole is located in the origin of
    the local object coordinate system. Local (Dipole) and global CS coincide when
    position=(0,0,0) and orientation=unit_rotation.

    Parameters
    ----------
    moment: array_like, shape (3,), unit [mT*mm^3]
        Magnetic dipole moment in units of [mT*mm^3] given in the local CS.
        For homogeneous magnets there is a relation moment=magnetization*volume.

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
    Dipole object: Dipole

    Examples
    --------
    By default a Dipole is initialized at position (0,0,0), with unit rotation:

    >>> import magpylib as magpy
    >>> dipole = magpy.misc.Dipole(moment=(100,100,100))
    >>> print(dipole.position)
    [0. 0. 0.]
    >>> print(dipole.orientation.as_quat())
    [0. 0. 0. 1.]

    Dipoles are magnetic field sources. Below we compute the H-field [kA/m] of the above Dipole at
    an observer position (1,1,1),

    >>> H = dipole.getH((1,1,1))
    >>> print(H)
    [2.43740886 2.43740886 2.43740886]

    or at a set of observer positions:

    >>> H = dipole.getH([(1,1,1), (2,2,2), (3,3,3)])
    >>> print(H)
    [[2.43740886 2.43740886 2.43740886]
     [0.30467611 0.30467611 0.30467611]
     [0.0902744  0.0902744  0.0902744 ]]

    The same result is obtained when the Dipole object moves along a path,
    away from the observer:

    >>> dipole.move([(-1,-1,-1), (-2,-2,-2)], start=1)
    >>> H = dipole.getH((1,1,1))
    >>> print(H)
    [[2.43740886 2.43740886 2.43740886]
     [0.30467611 0.30467611 0.30467611]
     [0.0902744  0.0902744  0.0902744 ]]
    """

    def __init__(
        self,
        moment=(mx, my, mz),
        position=(0, 0, 0),
        orientation=None,
        style=None,
        **kwargs,
    ):
        # instance attributes
        self.moment = moment
        self._object_type = "Dipole"

        # init inheritance
        BaseGeo.__init__(self, position, orientation, style=style, **kwargs)
        BaseDisplayRepr.__init__(self)

    # property getters and setters
    @property
    def moment(self):
        """Object moment attributes getter and setter."""
        return self._moment

    @moment.setter
    def moment(self, mom):
        """Set dipole moment vector, shape (3,), unit [mT*mm^3]."""
        # input type check
        if Config.checkinputs:
            check_vector_type(mom, "moment")

        # secure type
        mom = np.array(mom, dtype=float)

        # input format check
        if Config.checkinputs:
            check_vector_format(mom, (3,), "moment")

        self._moment = mom

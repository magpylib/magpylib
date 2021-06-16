"""Circular current class code"""

from magpylib._lib.obj_classes.class_BaseGeo import BaseGeo
from magpylib._lib.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._lib.obj_classes.class_BaseGetBH import BaseGetBH
from magpylib._lib.obj_classes.class_BaseExcitations import BaseCurrent
from magpylib._lib.config import Config
from magpylib._lib.input_checks import check_scalar_type, check_scalar_init

# init for tool tips
i0=None

# ON INTERFACE
class Circular(BaseGeo, BaseDisplayRepr, BaseGetBH, BaseCurrent):
    """
    Circular current loop.

    Reference position: Center of the loop.

    Reference orientation: The loop lies in the xy-plane of the global CS.

    Parameters
    ----------
    current: float
        Electrical current in units of [A].

    diameter: float
        Diameter of the loop in units of [mm].

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
    Circular object: Circular

    Examples
    --------
    # By default a Circular is initialized at position (0,0,0), with unit rotation:

    >>> import magpylib as mag3
    >>> magnet = mag3.current.Circular(current=100, diameter=2)
    >>> print(magnet.position)
    [0. 0. 0.]
    >>> print(magnet.orientation.as_quat())
    [0. 0. 0. 1.]

    Circulars are magnetic field sources. Below we compute the H-field [kA/m] of the
    above Circular at the observer position (1,1,1),

    >>> H = magnet.getH((1,1,1))
    >>> print(H)
    [4.96243034 4.96243034 2.12454191]

    or at a set of observer positions:

    >>> H = magnet.getH([(1,1,1), (2,2,2), (3,3,3)])
    >>> print(H)
    [[4.96243034 4.96243034 2.12454191]
     [0.61894364 0.61894364 0.06167939]
     [0.18075829 0.18075829 0.00789697]]

    The same result is obtained when the Circular moves along a path,
    away from the observer:

    >>> magnet.move([(-1,-1,-1), (-2,-2,-2)], start=1)
    >>> H = magnet.getH((1,1,1))
    >>> print(H)
    [[4.96243034 4.96243034 2.12454191]
     [0.61894364 0.61894364 0.06167939]
     [0.18075829 0.18075829 0.00789697]]
    """

    def __init__(
            self,
            current = i0,
            diameter = None,
            position = (0,0,0),
            orientation = None):

        # inherit base_geo class
        BaseGeo.__init__(self, position, orientation)
        BaseDisplayRepr.__init__(self)
        BaseCurrent.__init__(self, current)

        # set mag and dim attributes
        self.diameter = diameter
        self.object_type = 'Circular'

    @property
    def diameter(self):
        """ Object diameter attribute getter and setter.
        """
        return self._diameter

    @diameter.setter
    def diameter(self, dia):
        """ Set Circular loop diameter, float, [mm].
        """
        # input type and init check
        if Config.CHECK_INPUTS:
            check_scalar_init(dia, 'diameter')
            check_scalar_type(dia, 'diameter')

        self._diameter = float(dia)

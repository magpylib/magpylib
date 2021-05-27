"""Magnet Sphere class code"""

from magpylib._lib.obj_classes.class_BaseGeo import BaseGeo
from magpylib._lib.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._lib.obj_classes.class_BaseGetBH import BaseGetBH
from magpylib._lib.obj_classes.class_BaseHomMag import BaseHomMag
from magpylib._lib.exceptions import MagpylibBadUserInput
from magpylib._lib.config import Config

# init for tool tips
dia=None
mx=my=mz=None

# ON INTERFACE
class Sphere(BaseGeo, BaseDisplayRepr, BaseGetBH, BaseHomMag):
    """
    Spherical magnet with homogeneous magnetization.

    init_state: The center of the sphere lies in the CS origin.

    Properties
    ----------
    mag: array_like, shape (3,), unit [mT]
        Magnetization vector (remanence field) in units of [mT].

    dim: float, unit [mm]
        Diameter of the Sphere in units of [mm].

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
        returns string "Sphere(id)"

    Methods
    -------
    getB(observers):
        Compute B-field of Sphere at observers.

    getH(observers):
        Compute H-field of Sphere at observers.

    display(markers=[(0,0,0)], axis=None, direc=False, show_path=True):
        Display Sphere graphically using Matplotlib.

    move_by(displacement, steps=None):
        Linear displacement of Sphere by argument vector.

    move_to(target_pos, steps=None):
        Linear motion of Sphere to target_pos.

    rotate(rot, anchor=None, steps=None):
        Rotate Sphere about anchor.

    rotate_from_angax(angle, axis, anchor=None, steps=None, degree=True):
        Sphere rotation from angle-axis-anchor input.

    reset_path():
        Set Sphere.pos to (0,0,0) and Sphere.rot to unit rotation.

    Returns
    -------
    Sphere object
    """

    def __init__(
            self,
            mag = (mx,my,mz),
            dim = dia,
            pos = (0,0,0),
            rot = None):

        # inherit base_geo class
        BaseGeo.__init__(self, pos, rot)
        BaseDisplayRepr.__init__(self)
        BaseHomMag.__init__(self, mag)

        # set attributes
        self.dim = dim
        self.obj_type = 'Sphere'

    @property
    def dim(self):
        """ Sphere dimension dia in [mm].
        """
        return self._dim

    @dim.setter
    def dim(self, dimension):
        """ Set Sphere dimension dia, float, [mm].
        """
        # input check
        if Config.CHECK_INPUTS:
            if dimension is None:
                raise MagpylibBadUserInput('Dimension input required')

        self._dim = float(dimension)

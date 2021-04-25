"""Magnet Sphere class code"""

from magpylib._lib.obj_classes.class_BaseGeo import BaseGeo
from magpylib._lib.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._lib.obj_classes.class_BaseGetBH import BaseGetBH
from magpylib._lib.exceptions import MagpylibBadUserInput
from magpylib._lib.config import Config

# init for tool tips
dia=None
c0=None

# ON INTERFACE
class Circular(BaseGeo, BaseDisplayRepr, BaseGetBH):
    """
    Circular current loop with.

    init_state: The current loop lies in the xy-plane of the global CS and the
        geometric center coincides with the origin.

    Properties
    ----------
    current: float, unit [A]
        Current that flows in the loop.

    dim: float, unit [mm]
        Diameter of the loop in units of [mm].

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
        returns string "Circular(id)"

    Methods
    -------
    getB(observers):
        Compute B-field of loop at observers.

    getH(observers):
        Compute H-field of loop at observers.

    display(markers=[(0,0,0)], axis=None, direc=False, show_path=True):
        Display loop graphically using Matplotlib.

    move_by(displacement, steps=None):
        Linear displacement of loop by argument vector.

    move_to(target_pos, steps=None):
        Linear motion of loop to target_pos.

    rotate(rot, anchor=None, steps=None):
        Rotate loop about anchor.

    rotate_from_angax(angle, axis, anchor=None, steps=None, degree=True):
        loop rotation from angle-axis-anchor input.

    reset_path():
        Set Circular.pos to (0,0,0) and Circular.rot to unit rotation.

    Returns
    -------
    Circular object
    """

    def __init__(
            self,
            current = c0,
            dim = dia,
            pos = (0,0,0),
            rot = None):

        # inherit base_geo class
        BaseGeo.__init__(self, pos, rot)
        BaseDisplayRepr.__init__(self)

        # set mag and dim attributes
        self.current = current
        self.dim = dim
        self.obj_type = 'Circular'

    # properties ----------------------------------------------------
    @property
    def current(self):
        """ Current in units of [A].
        """
        return self._current

    @current.setter
    def current(self, current_inp):
        """ Set current, float, unit [A].
        """
        # input check
        if Config.CHECK_INPUTS:
            if current_inp is None:
                raise MagpylibBadUserInput('Current input required')

        # secure type
        current = float(current_inp)

        self._current = current


    @property
    def dim(self):
        """ Circular loop diameter in [mm].
        """
        return self._dim

    @dim.setter
    def dim(self, dimension):
        """ Set Circular loop diameter, float, [mm].
        """
        # input check
        if Config.CHECK_INPUTS:
            if dimension is None:
                raise MagpylibBadUserInput('Dimension input required')

        self._dim = float(dimension)

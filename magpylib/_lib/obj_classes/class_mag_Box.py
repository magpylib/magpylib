"""Magnet Box class code"""

import numpy as np
from magpylib._lib.obj_classes.class_BaseGeo import BaseGeo
from magpylib._lib.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._lib.obj_classes.class_BaseGetBH import BaseGetBH
from magpylib._lib.obj_classes.class_BaseExcitations import BaseHomMag
from magpylib._lib.config import Config
from magpylib._lib.input_checks import check_vector_format, check_vector_init, check_vector_type

# init for tool tips
a=b=c=None
mx=my=mz=None

# ON INTERFACE
class Box(BaseGeo, BaseDisplayRepr, BaseGetBH, BaseHomMag):
    """
    Cuboid magnet with homogeneous magnetization.

    init_state: The geometric center is in the global CS origin and the sides of the Box are
    parallel to the x/y/z basis vectors.

    Properties
    ----------
    mag: array_like, shape (3,), unit [mT]
        Magnetization vector (remanence field) in units of [mT].

    dim: array_like, shape (3,), unit [mm]
        Dimension/Size of the Box with sides [a,b,c] in units of [mm].

    pos: array_like, shape (3,) or (N,3), default=(0,0,0), unit [mm]
        Position of Box center in units of [mm]. For N>1 pos respresents a path in
        in the global CS.

    rot: scipy Rotation object with length 1 or N, default=unit rotation
        Source rotation relative to the init_state. For N>1 rot represents different rotations
        along a position-path.

    Dunders
    -------

    __add__:
        Adding sources creates a Collection "col = src1 + src2"

    __repr__:
        returns string "Box(id)"

    Methods
    -------
    getB(observers):
        Compute B-field of Box at observers.

    getH(observers):
        Compute H-field of Box at observers.

    display(markers=[(0,0,0)], axis=None, direc=False, show_path=True):
        Display Box graphically using Matplotlib.

    move_by(displacement, steps=None):
        Linear displacement of Box by argument vector.

    move_to(target_pos, steps=None):
        Linear motion of Box to target_pos.

    rotate(rot, anchor=None, steps=None):
        Rotate Box about anchor.

    rotate_from_angax(angle, axis, anchor=None, steps=None, degree=True):
        Box rotation from angle-axis-anchor input.

    reset_path():
        Set Box.pos to (0,0,0) and Box.rot to unit rotation.

    Returns
    -------
    Box object
    """

    def __init__(
            self,
            magnetization = (mx,my,mz),
            dimension = (a,b,c),
            position = (0,0,0),
            orientation = None):

        # inherit
        BaseGeo.__init__(self, position, orientation)
        BaseDisplayRepr.__init__(self)
        BaseHomMag.__init__(self, magnetization)

        # set attributes
        self.dimension = dimension
        self.object_type = 'Box'


    # properties ----------------------------------------------------

    @property
    def dimension(self):
        """ Box dimension (a,b,c) in [mm].
        """
        return self._dimension

    @dimension.setter
    def dimension(self, dim):
        """ Set Box dimension (a,b,c), shape (3,), [mm].
        """
        # input type and init check
        if Config.CHECK_INPUTS:
            check_vector_type(dim, 'dimension')
            check_vector_init(dim, 'dimension')

        # input type -> ndarray
        dim = np.array(dim,dtype=float)

        # input format check
        if Config.CHECK_INPUTS:
            check_vector_format(dim, (3,), 'dimension')

        self._dimension = dim

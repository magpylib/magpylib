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

    init_state: the geometric center is in the global CS origin, the axis of the
        cylinder coincides  with the z-axis.

    By default ITER_CYLINDER=50 for iteration of diametral magnetization computation.
        Use "magpylib.Config.ITER_CYLINDER=X" to change this setting.

    Properties
    ----------
    mag: array_like, shape (3,), unit [mT]
        Magnetization vector (remanence field) in units of [mT].

    dim: array_like, shape (2,), unit [mm]
        Dimension/Size of the Cylinder with diameter/height (d,h) in units of [mm].

    pos: array_like, shape (3,) or (N,3), default=(0,0,0), unit [mm]
        Position of Cylinder center in units of [mm]. For N>1 pos respresents a path in
        in the global CS.

    rot: scipy Rotation object with length 1 or N, default=unit rotation
        Source rotation relative to the init_state. For N>1 rot represents different rotations
        along a position-path.

    Dunders
    -------

    __add__:
        Adding sources creates a Collection "col = src1 + src2"

    __repr__:
        returns string "Cylinder(id)"

    Methods
    -------
    getB(observers):
        Compute B-field of Cylinder at observers.

    getH(observers):
        Compute H-field of Cylinder at observers.

    display(markers=[(0,0,0)], axis=None, direc=False, show_path=True):
        Display Cylinder graphically using Matplotlib.

    move_by(displacement, steps=None):
        Linear displacement of Cylinder by argument vector.

    move_to(target_pos, steps=None):
        Linear motion of Cylinder to target_pos.

    rotate(rot, anchor=None, steps=None):
        Rotate Cylinder about anchor.

    rotate_from_angax(angle, axis, anchor=None, steps=None, degree=True):
        Cylinder rotation from angle-axis-anchor input.

    reset_path():
        Set Cylinder.pos to (0,0,0) and Cylinder.rot to unit rotation.

    Returns
    -------
    Cylinder object
    """

    def __init__(
            self,
            mag = (mx,my,mz),
            dim = (d,h),
            pos = (0,0,0),
            rot = None):

        # inherit base_geo class
        BaseGeo.__init__(self, pos, rot)
        BaseDisplayRepr.__init__(self)
        BaseHomMag.__init__(self, mag)

        # set attributes
        self.dim = dim
        self.obj_type = 'Cylinder'

    @property
    def dim(self):
        """ Cylinder dimension (d,h), unit [mm]
        """
        return self._dim

    @dim.setter
    def dim(self, dim):
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

        self._dim = dim

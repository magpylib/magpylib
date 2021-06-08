"""Line current class code"""

import numpy as np
from magpylib._lib.obj_classes.class_BaseGeo import BaseGeo
from magpylib._lib.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._lib.obj_classes.class_BaseGetBH import BaseGetBH
from magpylib._lib.obj_classes.class_BaseExcitations import BaseCurrent
from magpylib._lib.config import Config
from magpylib._lib.input_checks import check_vector_init, check_vertex_format, check_vector_type

# init for tool tips
i0=None
pos1=pos2=None

# ON INTERFACE
class Line(BaseGeo, BaseDisplayRepr, BaseGetBH, BaseCurrent):
    """
    Line current.

    init_state: The line current flows in straight lines from vertex to vertex. The total
        assembly of line currents given by the vertices represents the Line object.

    Properties
    ----------
    current: float, unit [A]
        Current that flows along the lines.

    vertices: array_like, shape (N,3), unit [mm]
        The line current flows in straight lines from vertex to vertex.

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
        returns string "Line(id)"

    Methods
    -------
    getB(observers):
        Compute B-field of Line at observers.

    getH(observers):
        Compute H-field of Line at observers.

    display(markers=[(0,0,0)], axis=None, direc=False, show_path=True):
        Display Line graphically using Matplotlib.

    move_by(displacement, steps=None):
        Linear displacement of Line by argument vector.

    move_to(target_pos, steps=None):
        Linear motion of Line to target_pos.

    rotate(rot, anchor=None, steps=None):
        Rotate Line about anchor.

    rotate_from_angax(angle, axis, anchor=None, steps=None, degree=True):
        Line rotation from angle-axis-anchor input.

    reset_path():
        Set Line.pos to (0,0,0) and Line.rot to unit rotation.

    Returns
    -------
    Line object
    """
    # pylint: disable=dangerous-default-value

    def __init__(
            self,
            current = i0,
            vertices = [pos1, pos2],
            position = (0,0,0),
            orientation = None):

        # inherit base_geo class
        BaseGeo.__init__(self, position, orientation)
        BaseDisplayRepr.__init__(self)
        BaseCurrent.__init__(self, current)

        # set mag and dim attributes
        self.vertices = vertices
        self.obj_type = 'Line'

    @property
    def vertices(self):
        """ Line vertices in units of [mm].
        """
        return self._vertices

    @vertices.setter
    def vertices(self, vert):
        """ Set Line vertices, array_like, [mm].
        """
        # input type and init check
        if Config.CHECK_INPUTS:
            check_vector_type(vert, 'vertices')
            check_vector_init(vert, 'vertices')

        vert = np.array(vert)

        # input format check
        if Config.CHECK_INPUTS:
            check_vertex_format(vert)

        self._vertices = vert

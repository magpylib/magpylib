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
    Current flowing in straight lines from vertex to vertex.

    init_state: Vertices lie as placed in the local coordinate system of the Line object.

    Properties
    ----------
    current: float
        Electrical current in units of [A].

    vertices: array_like, shape (N,3)
        Current flows along vertices, given in units of [mm].

    position: array_like, shape (3,) or (M,3), default=(0,0,0)
        Position of the Line current local CS origin in units of [mm]. For M>1, the
        position attribute represents a path in the global CS. The attributes
        orientation and position must always be of the same length.

    orientation: scipy Rotation object with length 1 or M, default=unit rotation
        Line orientation relative to the initial state. For M>1 orientation
        represents different values along a path. The attributes orientation and
        position must always be of the same length.

    Returns
    -------
    Line object: Line
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
        self.object_type = 'Line'

    @property
    def vertices(self):
        """ Object vertices attribute getter and setter.
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

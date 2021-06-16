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

    Reference position: Origin of the local CS of the Line.

    Reference orientation: Local and global CS coincide at initialization.

    Parameters
    ----------
    current: float
        Electrical current in units of [A].

    vertices: array_like, shape (N,3)
        Current flows along vertices, given in units of [mm]. Vertices are defined
        in the local CS of the Line object.

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
    Line object: Line

    Examples
    --------
    # By default a Line is initialized at position (0,0,0), with unit rotation:

    >>> import magpylib as mag3
    >>> magnet = mag3.current.Line(current=100, vertices=[(-1,0,0),(1,0,0)])
    >>> print(magnet.position)
    [0. 0. 0.]
    >>> print(magnet.orientation.as_quat())
    [0. 0. 0. 1.]

    Lines are magnetic field sources. Below we compute the H-field [kA/m] of the above Line at the
    observer position (1,1,1),

    >>> H = magnet.getH((1,1,1))
    >>> print(H)
    [ 0.         -3.24873667  3.24873667]

    or at a set of observer positions:

    >>> H = magnet.getH([(1,1,1), (2,2,2), (3,3,3)])
    >>> print(H)
    [[ 0.         -3.24873667  3.24873667]
     [ 0.         -0.78438229  0.78438229]
     [ 0.         -0.34429579  0.34429579]]

    The same result is obtained when the Line moves along a path,
    away from the observer:

    >>> magnet.move([(-1,-1,-1), (-2,-2,-2)], start=1)
    >>> H = magnet.getH((1,1,1))
    >>> print(H)
    [[ 0.         -3.24873667  3.24873667]
     [ 0.         -0.78438229  0.78438229]
     [ 0.         -0.34429579  0.34429579]]
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

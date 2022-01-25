"""Line current class code"""

import numpy as np
from magpylib._src.obj_classes.class_BaseGeo import BaseGeo
from magpylib._src.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._src.obj_classes.class_BaseGetBH import BaseGetBH
from magpylib._src.obj_classes.class_BaseExcitations import BaseCurrent
from magpylib._src.defaults.defaults_classes import default_settings as Config
from magpylib._src.input_checks import check_vertex_format, check_vector_type

# init for tool tips
i0 = None
pos1 = pos2 = (None, None, None)

# ON INTERFACE
class Line(BaseGeo, BaseDisplayRepr, BaseGetBH, BaseCurrent):
    """
    Current flowing in straight lines from vertex to vertex.

    Local object coordinates: The Line current vertices are defined in the local object
    coordinate system. Local (Line) and global CS coincide when position=(0,0,0)
    and orientation=unit_rotation.

    Parameters
    ----------
    current: float
        Electrical current in units of [A].

    vertices: array_like, shape (N,3)
        The current flows along the vertices which are given in units of [mm] in the
        local CS of the Line object.

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
    Line object: Line

    Examples
    --------
    # By default a Line is initialized at position (0,0,0), with unit rotation:

    >>> import magpylib as magpy
    >>> magnet = magpy.current.Line(current=100, vertices=[(-1,0,0),(1,0,0)])
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
        current=i0,
        vertices=[pos1, pos2],
        position=(0, 0, 0),
        orientation=None,
        style=None,
        **kwargs,
    ):

        # instance attributes
        self.vertices = vertices
        self._object_type = "Line"

        # init inheritance
        BaseGeo.__init__(self, position, orientation, style=style, **kwargs)
        BaseDisplayRepr.__init__(self)
        BaseCurrent.__init__(self, current)

    # property getters and setters
    @property
    def vertices(self):
        """Object vertices attribute getter and setter."""
        return self._vertices

    @vertices.setter
    def vertices(self, vert):
        """Set Line vertices, array_like, [mm]."""
        # input type check
        if Config.checkinputs:
            check_vector_type(vert, "vertices")

        # input type -> ndarray
        vert = np.array(vert)

        # input format check
        if Config.checkinputs:
            check_vertex_format(vert)

        self._vertices = vert

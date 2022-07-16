"""Magnet Tetrahedron class code
DOCSTRINGS V4 READY
"""
import numpy as np

from magpylib._src.fields.field_BH_tetrahedron import magnet_tetrahedron_field
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._src.obj_classes.class_BaseExcitations import BaseHomMag
from magpylib._src.obj_classes.class_BaseGeo import BaseGeo
from magpylib._src.obj_classes.class_BaseGetBH import BaseGetBH
from magpylib._src.utility import Registered


@Registered(
    kind="source",
    family="magnet",
    field_func=magnet_tetrahedron_field,
    source_kwargs_ndim={"magnetization": 1, "vertices": 3},
)
class Tetrahedron(BaseGeo, BaseDisplayRepr, BaseGetBH, BaseHomMag):
    """Tetrahedron magnet with homogeneous magnetization.

    Can be used as `sources` input for magnetic field computation.

    When `position=(0,0,0)` and `orientation=None` the Tetrahedron vertices coordinates
    are the same as in the global coordinate system. The geometric center of the Tetrahedron
    is determined by its vertices and is not necessarily located in the origin.

    Parameters
    ----------
    magnetization: array_like, shape (3,), default=`None`
        Magnetization vector (mu0*M, remanence field) in units of [mT] given in
        the local object coordinates (rotates with object).

    vertices: ndarray, shape (4,3)
        Vertices (x1,y1,z1), (x2,y2,z2), (x3,y3,z3), (x4,y4,z4), in the relative
        coordinate system of the tetrahedron.

    position: array_like, shape (3,) or (m,3)
        Object position(s) in the global coordinates in units of [mm]. For m>1, the
        `position` and `orientation` attributes together represent an object path.
        When setting vertices, the initial position is set to the barycenter.

    orientation: scipy `Rotation` object with length 1 or m, default=`None`
        Object orientation(s) in the global coordinates. `None` corresponds to
        a unit-rotation. For m>1, the `position` and `orientation` attributes
        together represent an object path.

    parent: `Collection` object or `None`
        The object is a child of it's parent collection.

    style: dict
        Object style inputs must be in dictionary form, e.g. `{'color':'red'}` or
        using style underscore magic, e.g. `style_color='red'`.

    Returns
    -------
    magnet source: `Tetrahedron` object

    Examples
    --------
    `Tetrahedron` magnets are magnetic field sources. Below we compute the H-field [kA/m] of a
    tetrahedral magnet with magnetization (100,200,300) in units of [mT] and 1 [mm] sides
    at the observer position (0,0,0) given in units of [mm]:

    >>> import magpylib as magpy
    >>> vertices = [(1,0,-1/2**0.5),(0,1,1/2**0.5),(-1,0,-1/2**0.5),(1,-1,1/2**0.5)]
    >>> src = magpy.magnet.Tetrahedron((100,200,300), vertices=vertices)
    >>> H = src.getH((0,0,0))
    >>> print(H)
    [  3.42521345 -40.76504699 -70.06509857]

    We rotate the source object, and compute the B-field, this time at a set of observer positions:

    >>> src.rotate_from_angax(45, 'x')
    Tetrahedron(id=...)
    >>> B = src.getB([(1,1,1), (2,2,2), (3,3,3)])
    >>> print(B)
    [[3.2653876  7.77807843 0.41141725]
     [0.49253111 0.930953   0.0763492 ]
     [0.1497206  0.26663798 0.02164654]]
    """

    def __init__(
        self,
        magnetization=None,
        vertices=None,
        position=(0, 0, 0),
        orientation=None,
        style=None,
        **kwargs,
    ):

        # instance attributes
        self.vertices = vertices
        self._object_type = "Tetrahedron"

        # init inheritance
        BaseGeo.__init__(self, position, orientation, style=style, **kwargs)
        BaseDisplayRepr.__init__(self)
        BaseHomMag.__init__(self, magnetization)

    # property getters and setters
    @property
    def vertices(self):
        """Length of the Tetrahedron sides [a,b,c] in units of [mm]."""
        return self._vertices

    @vertices.setter
    def vertices(self, dim):
        """Set Tetrahedron vertices (a,b,c), shape (3,), [mm]."""
        self._vertices = check_format_input_vector(
            dim,
            dims=(2,),
            shape_m1=3,
            length=4,
            sig_name="Tetrahedron.vertices",
            sig_type="array_like (list, tuple, ndarray) of shape (4,3)",
            allow_None=True,
        )

    @property
    def _barycenter(self):
        """Object barycenter."""
        return self._get_barycenter(self._position, self._orientation, self.vertices)

    @property
    def barycenter(self):
        """Object barycenter."""
        return np.squeeze(self._barycenter)

    @staticmethod
    def _get_barycenter(position, orientation, vertices):
        """Returns the barycenter of a tetrahedron."""
        centroid = np.mean(vertices, axis=0)
        barycenter = orientation.apply(centroid) + position
        return barycenter

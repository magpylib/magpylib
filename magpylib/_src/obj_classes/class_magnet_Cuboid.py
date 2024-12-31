# pylint: disable=too-many-positional-arguments

"""Magnet Cuboid class code"""

from magpylib._src.display.traces_core import make_Cuboid
from magpylib._src.fields.field_BH_cuboid import BHJM_magnet_cuboid
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet
from magpylib._src.utility import unit_prefix


class Cuboid(BaseMagnet):
    """Cuboid magnet with homogeneous magnetization.

    Can be used as `sources` input for magnetic field computation.

    When `position=(0,0,0)` and `orientation=None` the Cuboid sides are parallel
    to the global coordinate basis vectors and the geometric center of the Cuboid
    is located in the origin.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    position: array_like, shape (3,) or (m,3), default=`(0,0,0)`
        Object position(s) in the global coordinates in units of m.
        For m>1, the `position` and `orientation` attributes together
        represent an object path.

    orientation: scipy `Rotation` object with length 1 or m, default=`None`
        Object orientation(s) in the global coordinates. `None` corresponds to
        a unit-rotation. For m>1, the `position` and `orientation` attributes
        together represent an object path.

    dimension: array_like, shape (3,), default=`None`
        Length of the cuboid sides [a,b,c] in meters.

    polarization: array_like, shape (3,), default=`None`
        Magnetic polarization vector J = mu0*M in units of T,
        given in the local object coordinates (rotates with object).

    magnetization: array_like, shape (3,), default=`None`
        Magnetization vector M = J/mu0 in units of A/m,
        given in the local object coordinates (rotates with object).

    parent: `Collection` object or `None`
        The object is a child of it's parent collection.

    style: dict
        Object style inputs must be in dictionary form, e.g. `{'color':'red'}` or
        using style underscore magic, e.g. `style_color='red'`.

    Returns
    -------
    magnet source: `Cuboid` object

    Examples
    --------
    `Cuboid` magnets are magnetic field sources. Below we compute the H-field in A/m of a
    cubical magnet with magnetic polarization of (0.5,0.6,0.7) in units of T and
    0.01 meter sides at the observer position (0.01,0.01,0.01) given in units of m:

    >>> import magpylib as magpy
    >>> src = magpy.magnet.Cuboid(polarization=(.5,.6,.7), dimension=(.01,.01,.01))
    >>> H = src.getH((.01,.01,.01))
    >>> print(H)
    [16149.04135639 14906.8074059  13664.57345541]

    We rotate the source object, and compute the B-field, this time at a set of observer positions:

    >>> src.rotate_from_angax(45, 'x')
    Cuboid(id=...)
    >>> B = src.getB([(.01,0,0), (0,.01,0), (0,0,.01)])
    >>> print(B)
    [[ 0.06739119  0.00476528 -0.0619486 ]
     [-0.03557183 -0.01149497 -0.08403664]
     [-0.03557183  0.00646436  0.14943466]]
    """

    _field_func = staticmethod(BHJM_magnet_cuboid)
    _field_func_kwargs_ndim = {"polarization": 2, "dimension": 2}
    get_trace = make_Cuboid

    def __init__(
        self,
        position=(0, 0, 0),
        orientation=None,
        dimension=None,
        polarization=None,
        magnetization=None,
        style=None,
        **kwargs,
    ):
        # instance attributes
        self.dimension = dimension

        # init inheritance
        super().__init__(
            position, orientation, magnetization, polarization, style, **kwargs
        )

    # property getters and setters
    @property
    def dimension(self):
        """Length of the cuboid sides [a,b,c] in arbitrary length units, e.g. in meter."""
        return self._dimension

    @dimension.setter
    def dimension(self, dim):
        """Set Cuboid dimension (a,b,c), shape (3,)"""
        self._dimension = check_format_input_vector(
            dim,
            dims=(1,),
            shape_m1=3,
            sig_name="Cuboid.dimension",
            sig_type="array_like (list, tuple, ndarray) of shape (3,) with positive values",
            allow_None=True,
            forbid_negative0=True,
        )

    @property
    def _default_style_description(self):
        """Default style description text"""
        if self.dimension is None:
            return "no dimension"
        d = [unit_prefix(d) for d in self.dimension]
        return f"{d[0]}m|{d[1]}m|{d[2]}m"

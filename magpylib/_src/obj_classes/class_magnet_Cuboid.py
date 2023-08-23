"""Magnet Cuboid class code"""
from magpylib._src.display.traces_core import make_Cuboid
from magpylib._src.fields.field_BH_cuboid import magnet_cuboid_field
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet
from magpylib._src.utility import unit_prefix


class Cuboid(BaseMagnet):
    """Cuboid magnet with homogeneous magnetization.

    Can be used as `sources` input for magnetic field computation.

    When `position=(0,0,0)` and `orientation=None` the Cuboid sides are parallel
    to the global coordinate basis vectors and the geometric center of the Cuboid
    is located in the origin.

    Units can be chosen freely. B-field output unit is the same as magnetization
    input unit. Computation is independend of Length-unit. See online documentation
    for fore information

    Parameters
    ----------
    magnetization: array_like, shape (3,), default=`None`
        Magnetization (polarization) vector (mu0*M, remanence field) in arbitrary
        units given in the local object coordinates (rotates with object).

    dimension: array_like, shape (3,), default=`None`
        Length of the cuboid sides [a,b,c] in arbitrary units.

    position: array_like, shape (3,) or (m,3), default=`(0,0,0)`
        Object position(s) in the global coordinates in arbitrary units. For m>1, the
        `position` and `orientation` attributes together represent an object path.

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
    magnet source: `Cuboid` object

    Examples
    --------
    `Cuboid` magnets are magnetic field sources. Below we compute the H-field in kA/m of a
    cubical magnet with magnetization (100,200,300) in units of mT and 1 mm sides
    at the observer position (1,1,1) given in units of mm:

    >>> import magpylib as magpy
    >>> src = magpy.magnet.Cuboid(magnetization=(100,200,300), dimension=(1,1,1))
    >>> H = src.getH((1,1,1))
    >>> print(H)
    [6.21116976 4.9689358  3.72670185]

    We rotate the source object, and compute the B-field, this time at a set of observer positions:

    >>> src.rotate_from_angax(45, 'x')
    Cuboid(id=...)
    >>> B = src.getB([(1,1,1), (2,2,2), (3,3,3)])
    >>> print(B)
    [[4.30496934 6.9363475  0.50728577]
     [0.54127889 0.86827283 0.05653357]
     [0.1604214  0.25726266 0.01664045]]

    The same result is obtained when the rotated source moves along a path away from an
    observer at position (1,1,1). Here we use a `Sensor` object as observer.

    >>> sens = magpy.Sensor(position=(1,1,1))
    >>> src.move([(-1,-1,-1), (-2,-2,-2)])
    Cuboid(id=...)
    >>> B = src.getB(sens)
    >>> print(B)
    [[4.30496934 6.9363475  0.50728577]
     [0.54127889 0.86827283 0.05653357]
     [0.1604214  0.25726266 0.01664045]]
    """

    _field_func = staticmethod(magnet_cuboid_field)
    _field_func_kwargs_ndim = {"magnetization": 2, "dimension": 2}
    get_trace = make_Cuboid

    def __init__(
        self,
        magnetization=None,
        dimension=None,
        position=(0, 0, 0),
        orientation=None,
        style=None,
        **kwargs,
    ):
        # instance attributes
        self.dimension = dimension

        # init inheritance
        super().__init__(position, orientation, magnetization, style, **kwargs)

    # property getters and setters
    @property
    def dimension(self):
        """Length of the cuboid sides [a,b,c] in units of mm."""
        return self._dimension

    @dimension.setter
    def dimension(self, dim):
        """Set Cuboid dimension (a,b,c), shape (3,), mm."""
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
        d = [unit_prefix(d / 1000) for d in self.dimension]
        return f"{d[0]}m|{d[1]}m|{d[2]}m"

"""Magnet Cylinder class code"""
from magpylib._src.display.traces_core import make_Cylinder
from magpylib._src.fields.field_BH_cylinder_segment import magnet_cylinder_field
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet
from magpylib._src.utility import unit_prefix


class Cylinder(BaseMagnet):
    """Cylinder magnet with homogeneous magnetization.

    Can be used as `sources` input for magnetic field computation.

    When `position=(0,0,0)` and `orientation=None` the geometric center of the
    cylinder lies in the origin of the global coordinate system and
    the cylinder axis coincides with the global z-axis.

    Parameters
    ----------
    magnetization: array_like, shape (3,), default=`None`
        Magnetization vector (mu0*M, remanence field) in units of mT given in
        the local object coordinates (rotates with object).

    dimension: array_like, shape (2,), default=`None`
        Dimension (d,h) denote diameter and height of the cylinder in units of mm.

    position: array_like, shape (3,) or (m,3), default=`(0,0,0)`
        Object position(s) in the global coordinates in units of mm. For m>1, the
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
    magnet source: `Cylinder`

    Examples
    --------
    `Cylinder` magnets are magnetic field sources. Below we compute the H-field in kA/m of a
    cylinder magnet with magnetization (100,200,300) in units of mT and 1 mm diameter and height
    at the observer position (1,1,1) given in units of mm:

    >>> import magpylib as magpy
    >>> src = magpy.magnet.Cylinder(magnetization=(100,200,300), dimension=(1,1))
    >>> H = src.getH((1,1,1))
    >>> print(H)
    [4.84991343 3.88317816 2.73973202]

    We rotate the source object, and compute the B-field, this time at a set of observer positions:

    >>> src.rotate_from_angax(45, 'x')
    Cylinder(id=...)
    >>> B = src.getB([(1,1,1), (2,2,2), (3,3,3)])
    >>> print(B)
    [[3.31419501 5.26683023 0.37767015]
     [0.42298405 0.67710536 0.04464932]
     [0.12571523 0.20144503 0.01312389]]

    The same result is obtained when the rotated source moves along a path away from an
    observer at position (1,1,1). Here we use a `Sensor` object as observer.

    >>> sens = magpy.Sensor(position=(1,1,1))
    >>> src.move([(-1,-1,-1), (-2,-2,-2)])
    Cylinder(id=...)
    >>> B = src.getB(sens)
    >>> print(B)
    [[3.31419501 5.26683023 0.37767015]
     [0.42298405 0.67710536 0.04464932]
     [0.12571523 0.20144503 0.01312389]]
    """

    _field_func = staticmethod(magnet_cylinder_field)
    _field_func_kwargs_ndim = {"magnetization": 2, "dimension": 2}
    get_trace = make_Cylinder

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
        """Dimension (d,h) denote diameter and height of the cylinder in units of mm."""
        return self._dimension

    @dimension.setter
    def dimension(self, dim):
        """Set Cylinder dimension (d,h) in units of mm."""
        self._dimension = check_format_input_vector(
            dim,
            dims=(1,),
            shape_m1=2,
            sig_name="Cylinder.dimension",
            sig_type="array_like (list, tuple, ndarray) with shape (2,) with positive values",
            allow_None=True,
            forbid_negative0=True,
        )

    @property
    def _default_style_description(self):
        """Default style description text"""
        if self.dimension is None:
            return "no dimension"
        d = [unit_prefix(d / 1000) for d in self.dimension]
        return f"D={d[0]}m, H={d[1]}m"

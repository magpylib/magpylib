# pylint: disable=too-many-positional-arguments

"""Magnet Cuboid class code"""

from typing import ClassVar

import numpy as np

from magpylib._src.display.traces_core import make_Cuboid
from magpylib._src.fields.field_BH_cuboid import BHJM_magnet_cuboid
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet
from magpylib._src.obj_classes.class_BaseTarget import BaseTarget
from magpylib._src.obj_classes.target_meshing import target_mesh_cuboid
from magpylib._src.utility import unit_prefix


class Cuboid(BaseMagnet, BaseTarget):
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

    meshing: int, array_like, shape (3,), default=`None`
        Parameter that defines the mesh fineness for force computation.
        Must be a positive integer specifying the target mesh size or an
        explicit splitting of the cuboid into regular cubic grid cells with
        shape (n1,n2,n3).

    volume: float
        Read-only. Object physical volume in units of m^3.

    centroid: np.ndarray, shape (3,) or (m,3)
        Read-only. Object centroid in units of m.

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

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> src = magpy.magnet.Cuboid(polarization=(.5,.6,.7), dimension=(.01,.01,.01))
    >>> H = src.getH((.01,.01,.01))
    >>> with np.printoptions(precision=0):
    ...     print(H)
    [16149. 14907. 13665.]
    """

    _field_func = staticmethod(BHJM_magnet_cuboid)
    _force_type = "magnet"
    _field_func_kwargs_ndim: ClassVar[dict[str, int]] = {
        "polarization": 2,
        "dimension": 2,
    }
    get_trace = make_Cuboid

    def __init__(
        self,
        position=(0, 0, 0),
        orientation=None,
        dimension=None,
        polarization=None,
        magnetization=None,
        meshing=None,
        style=None,
        **kwargs,
    ):
        # instance attributes
        self.dimension = dimension

        # init inheritance
        super().__init__(
            position, orientation, magnetization, polarization, style, **kwargs
        )

        # Initialize BaseTarget
        BaseTarget.__init__(self, meshing)

    # Properties
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

    # Methods
    def _get_volume(self):
        """Volume of object in units of m³."""
        if self.dimension is None:
            return 0.0
        return np.prod(self.dimension)

    def _get_centroid(self, squeeze=True):
        """Centroid of object in units of m."""
        if squeeze:
            return self.position
        return self._position

    def _generate_mesh(self):
        """Generate mesh for force computation."""
        return target_mesh_cuboid(self.meshing, self.dimension, self.magnetization)

    def _validate_meshing(self, value):
        """Cuboid meshing must be a positive integer or array_like of shape (3,)."""
        if (isinstance(value, int) and value > 0) or (
            isinstance(value, list | tuple | np.ndarray) and len(value) == 3
        ):
            pass
        else:
            msg = (
                "Cuboid meshing parameter must be positive integer or array_like of shape"
                " (3,) for {self}. Instead got {value}."
            )
            raise ValueError(msg)

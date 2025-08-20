# pylint: disable=too-many-positional-arguments

"""Magnet Cylinder class code"""

from typing import ClassVar

import numpy as np

from magpylib._src.display.traces_core import make_Cylinder
from magpylib._src.fields.field_BH_cylinder import BHJM_magnet_cylinder
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet
from magpylib._src.obj_classes.class_BaseTarget import BaseTarget
from magpylib._src.obj_classes.target_meshing import target_mesh_cylinder
from magpylib._src.utility import unit_prefix


class Cylinder(BaseMagnet, BaseTarget):
    """Cylinder magnet with homogeneous magnetization.

    Can be used as `sources` input for magnetic field computation.

    When `position=(0,0,0)` and `orientation=None` the geometric center of the
    cylinder lies in the origin of the global coordinate system and
    the cylinder axis coincides with the global z-axis.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    position: array_like, shape (3,) or (m,3), default=`(0,0,0)`
        Object position(s) in the global coordinates in units of m. For m>1, the
        `position` and `orientation` attributes together represent an object path.

    orientation: scipy `Rotation` object with length 1 or m, default=`None`
        Object orientation(s) in the global coordinates. `None` corresponds to
        a unit-rotation. For m>1, the `position` and `orientation` attributes
        together represent an object path.

    dimension: array_like, shape (2,), default=`None`
        Dimension (d,h) denote diameter and height of the cylinder in units of m.

    polarization: array_like, shape (3,), default=`None`
        Magnetic polarization vector J = mu0*M in units of T,
        given in the local object coordinates (rotates with object).

    magnetization: array_like, shape (3,), default=`None`
        Magnetization vector M = J/mu0 in units of A/m,
        given in the local object coordinates (rotates with object).

    meshing: int, default=`None`
        Parameter that defines the mesh fineness for force computation.
        Must be a positive integer specifying the target mesh size.

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
    magnet source: `Cylinder`

    Examples
    --------
    `Cylinder` magnets are magnetic field sources. Below we compute the H-field in A/m of a
    cylinder magnet with polarization (.1,.2,.3) in units of T and 0.01 meter diameter and
    height at the observer position (0.01,0.01,0.01) given in units of m:

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> src = magpy.magnet.Cylinder(polarization=(.1,.2,.3), dimension=(.01,.01))
    >>> H = src.getH((.01,.01,.01))
    >>> with np.printoptions(precision=3):
    ...     print(H)
    [4849.913 3883.178 2739.732]
    """

    _field_func = staticmethod(BHJM_magnet_cylinder)
    _force_type = "magnet"
    _field_func_kwargs_ndim: ClassVar[dict[str, int]] = {
        "polarization": 2,
        "dimension": 2,
    }
    get_trace = make_Cylinder

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
        """Dimension (d,h) denote diameter and height of the cylinder in units of m."""
        return self._dimension

    @dimension.setter
    def dimension(self, dim):
        """Set Cylinder dimension (d,h) in units of m."""
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
        d = [unit_prefix(d) for d in self.dimension]
        return f"D={d[0]}m, H={d[1]}m"

    # Methods
    def _get_volume(self):
        """Volume of object in units of m³."""
        if self.dimension is None:
            return 0.0

        d, h = self.dimension
        return d**2 * np.pi * h / 4

    def _get_centroid(self, squeeze=True):
        """Centroid of object in units of m."""
        if squeeze:
            return self.position
        return self._position

    def _generate_mesh(self):
        """Generate mesh for force computation."""
        # Tests in getFT ensure that meshing, dimension and excitation are set
        d, h = self.dimension
        return target_mesh_cylinder(
            0, d / 2, h, 0, 360, self.meshing, self.magnetization
        )

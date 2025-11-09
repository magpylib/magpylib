"""Magnet Cuboid class code"""

# pylint: disable=too-many-positional-arguments

from typing import ClassVar

import numpy as np

from magpylib._src.display.traces_core import make_Cuboid
from magpylib._src.fields.field_BH_cuboid import _BHJM_magnet_cuboid
from magpylib._src.input_checks import check_format_input_numeric
from magpylib._src.obj_classes.class_BaseExcitations import BaseMagnet
from magpylib._src.obj_classes.class_BaseProperties import (
    BaseDipoleMoment,
    BaseVolume,
)
from magpylib._src.obj_classes.class_BaseTarget import BaseTarget
from magpylib._src.obj_classes.target_meshing import _target_mesh_cuboid
from magpylib._src.utility import unit_prefix


class Cuboid(BaseMagnet, BaseTarget, BaseVolume, BaseDipoleMoment):
    """Cuboid magnet with homogeneous magnetization.

    Can be used as ``sources`` input for magnetic field computation and ``target``
    input for force computation.

    When ``position=(0, 0, 0)`` and ``orientation=None`` the Cuboid sides are parallel
    to the global coordinate basis vectors and the geometric center of the Cuboid
    is located in the origin.

    SI units are used for all inputs and outputs.

    Parameters
    ----------
    position : array-like, shape (3,) or (p, 3), default (0, 0, 0)
        Object position(s) in global coordinates in units (m). ``position`` and
        ``orientation`` attributes define the object path.
    orientation : Rotation | None, default None
        Object orientation(s) in global coordinates as a scipy Rotation. Rotation can
        have length 1 or p. ``None`` generates a unit-rotation.
    dimension : None | array-like, shape (3,) or (p, 3), default None
        Lengths of the cuboid sides (a, b, c) in units (m). Can be a path.
    polarization : None | array-like, shape (3,) or (p, 3), default None
        Magnetic polarization vector J = mu0*M in units (T), given in the
        local object coordinates. Sets also ``magnetization``. Can be a path.
    magnetization : None | array-like, shape (3,) or (p, 3), default None
        Magnetization vector M = J/mu0 in units (A/m), given in the local
        object coordinates. Sets also ``polarization``. Can be a path.
    meshing : None | int | array-like, shape (3,), default None
        Mesh fineness for force computation. Must be a positive integer specifying
        the target mesh size or an explicit splitting of the cuboid into regular
        cubic grid cells with shape (n1, n2, n3).
    style : dict | None, default None
        Style dictionary. Can also be provided via style underscore magic, e.g.
        ``style_color='red'``.

    Attributes
    ----------
    position : ndarray, shape (3,) or (p, 3)
        Same as constructor parameter ``position``.
    orientation : Rotation
        Same as constructor parameter ``orientation``.
    dimension : None | ndarray, shape (3,) or (p, 3)
        Same as constructor parameter ``dimension``.
    polarization : None | ndarray, shape (3,) or (p, 3)
        Same as constructor parameter ``polarization``.
    magnetization : None | ndarray, shape (3,) or (p, 3)
        Same as constructor parameter ``magnetization``.
    meshing : int | None
        Same as constructor parameter ``meshing``.
    centroid : ndarray, shape (3,) or (p, 3)
        Read-only. Object centroid in units (m) in global coordinates.
        Can be a path.
    dipole_moment : ndarray, shape (3,)
        Read-only. Object dipole moment (A·m²) in local object coordinates.
    volume : float
        Read-only. Object physical volume in units (m³).
    parent : Collection or None
        Parent collection of the object.
    style : dict
        Style dictionary defining visual properties.

    Notes
    -----
    Returns (0, 0, 0) on edges and corners.

    Examples
    --------
    ``Cuboid`` magnets are magnetic field sources. Below we compute the H-field in
    (A/m) of a cubical magnet with magnetic polarization (0.5, 0.6, 0.7) in
    units (T) and side lengths (0.01, 0.01, 0.01) at the observer position
    (0.01, 0.01, 0.01) (m):

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> src = magpy.magnet.Cuboid(polarization=(0.5, 0.6, 0.7), dimension=(0.01, 0.01, 0.01))
    >>> H = src.getH((0.01, 0.01, 0.01))
    >>> with np.printoptions(precision=0):
    ...     print(H)
    [16149. 14907. 13665.]
    """

    _field_func = staticmethod(_BHJM_magnet_cuboid)
    _force_type = "magnet"
    _field_func_kwargs_ndim: ClassVar[dict[str, int]] = {
        "polarization": 2,
        "dimension": 2,
    }
    _path_properties = ("dimension",)  # also inherits from parent class
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
        # init inheritance - dimension is passed via kwargs for path sync
        super().__init__(
            position,
            orientation,
            magnetization=magnetization,
            polarization=polarization,
            dimension=dimension,
            style=style,
            **kwargs,
        )

        # Initialize BaseTarget
        BaseTarget.__init__(self, meshing)

    # Properties
    @property
    def dimension(self):
        """Cuboid side lengths (a, b, c) in units (m)."""
        return np.squeeze(self._dimension) if self._dimension is not None else None

    @dimension.setter
    def dimension(self, dim):
        """Set cuboid side lengths.

        Parameters
        ----------
        dim : None or array-like, shape (3,) or (p, 3)
            Side lengths (a, b, c) in units (m). Can be a path.
        """
        self._dimension = check_format_input_numeric(
            dim,
            dtype=float,
            shapes=((3,), (None, 3)),
            name="Cuboid.dimension",
            allow_None=True,
            reshape=(-1, 3),
            value_conditions=[("gt", 0, "all")],
        )
        self._sync_all_paths(propagate=False)

    @property
    def _default_style_description(self):
        """Default style description text"""
        if self.dimension is None:
            return "no dimension"
        # Handle path dimensions similar to BaseCurrent
        dims = self._dimension
        if len(dims) == 1 or np.all(dims == dims[0], axis=0).all():
            # Single dimension or all dimensions are the same
            d = [unit_prefix(v) for v in dims[0]]
            return f"{d[0]}m|{d[1]}m|{d[2]}m"
        # Multiple different dimensions - show range only for varying axes
        dmin, dmax = np.nanmin(dims, axis=0), np.nanmax(dims, axis=0)
        parts = []
        for i in range(3):
            if np.allclose(dmin[i], dmax[i]):
                # No variation in this dimension
                parts.append(f"{unit_prefix(dmin[i])}m")
            else:
                # Variation in this dimension
                parts.append(f"{unit_prefix(dmin[i])}m↔{unit_prefix(dmax[i])}m")
        return "|".join(parts)

    # Methods
    def _get_volume(self):
        """Volume of object in units (m³)."""
        if self.dimension is None:
            return 0.0
        return np.prod(self.dimension)

    def _get_centroid(self, squeeze=True):
        """Centroid of object in units (m)."""
        if squeeze:
            return self.position
        return self._position

    def _get_dipole_moment(self, squeeze=True):  # noqa: ARG002
        """Magnetic moment of object in units (A*m²)."""
        # test init
        if self.magnetization is None or self.dimension is None:
            return np.array((0.0, 0.0, 0.0))
        return self.magnetization * self.volume

    def _generate_mesh(self):
        """Generate mesh for force computation."""
        return _target_mesh_cuboid(self.meshing, self.dimension, self.magnetization)

    def _validate_meshing(self, value):
        """Cuboid meshing must be a positive integer or array-like of shape (3,)."""
        if (isinstance(value, int) and value > 0) or (
            isinstance(value, list | tuple | np.ndarray) and len(value) == 3
        ):
            pass
        else:
            msg = (
                f"Input meshing of {self} must be positive integer or array-like of shape "
                f"(3,); instead received {value}."
            )
            raise ValueError(msg)

"""Magnetic field computation functional interface"""

import numbers
from collections.abc import Callable

import numpy as np
from scipy.spatial.transform import Rotation as R

from magpylib._src.exceptions import MagpylibBadUserInput
from magpylib._src.fields.field_BH_circle import _BHJM_circle
from magpylib._src.fields.field_BH_cuboid import _BHJM_magnet_cuboid
from magpylib._src.fields.field_BH_current_sheet import _BHJM_current_sheet
from magpylib._src.fields.field_BH_cylinder import _BHJM_magnet_cylinder
from magpylib._src.fields.field_BH_cylinder_segment import _BHJM_cylinder_segment
from magpylib._src.fields.field_BH_dipole import _BHJM_dipole
from magpylib._src.fields.field_BH_polyline import _BHJM_current_polyline
from magpylib._src.fields.field_BH_sphere import _BHJM_magnet_sphere
from magpylib._src.fields.field_BH_tetrahedron import _BHJM_magnet_tetrahedron
from magpylib._src.fields.field_BH_triangle import _BHJM_triangle
from magpylib._src.utility import get_registered_sources, has_parameter


def circle_field(
    field,
    observers,
    diameters,
    currents,
    positions=(0, 0, 0),
    orientations=None,
    squeeze=True,
):
    """Return B- or H-field of circular current loops for i instances.

    With ``positions=(0, 0, 0)`` and ``orientations=None`` a loop lies in the
    ``z=0`` plane with its center at the origin. Positive current flows in the
    mathematically positive (counter-clockwise) direction.

    Parameters
    ----------
    field : {'B', 'H'}
        Select which field is returned.
    observers : array-like, shape (3,) or (i, 3)
        Points where the field is evaluated in units (m).
    diameters : float | array-like, shape (i,)
        Loop diameters in units (m).
    currents : float | array-like, shape (i,)
        Electric currents in units (A).
    positions : array-like, shape (3,) or (i, 3), default (0, 0, 0)
        Loop centers in units (m).
    orientations : None | Rotation, default None
        Loop orientations. If ``None``, the unit rotation is applied.
    squeeze : bool, default True
        If ``True``, squeeze singleton axis when ``i=1``.

    Returns
    -------
    ndarray, shape (3,) or (i, 3)
        Field at the observer locations.

    Notes
    -----
    Returns (0, 0, 0) on the loop.

    Examples
    --------
    >>> import numpy as np
    >>> from magpylib.func import circle_field
    >>> B = circle_field(
    ...     field='B',
    ...     observers=(0.2, 0.3, 0.1),
    ...     diameters=(1.0, 1.5),
    ...     currents=1.0
    ... )
    >>> with np.printoptions(precision=3):
    ...     print(B)
    [[3.863e-07 5.795e-07 1.604e-06]
     [6.756e-08 1.013e-07 9.694e-07]]
    """
    params = {
        "observers": observers,
        "diameter": diameters,
        "current": currents,
        "position": positions,
        "orientation": orientations,
    }
    return _getBH_func(_BHJM_circle, field, params, squeeze)


def polyline_field(
    field,
    observers,
    segments_start,
    segments_end,
    currents,
    positions=(0, 0, 0),
    orientations=None,
    squeeze=True,
):
    """Return B- or H-field of straight current segments for i instances.

    With ``positions=(0, 0, 0)`` and ``orientations=None`` local and global
    coordinates coincide. Current flows in straight lines from segment start
    to end positions. The field is set to (0, 0, 0) on the segments.

    Parameters
    ----------
    field : {'B', 'H'}
        Select which field is returned.
    observers : array-like, shape (3,) or (i, 3)
        Points where the field is evaluated in units (m).
    segments_start : array-like, shape (3,) or (i, 3)
        Segment start points in units (m).
    segments_end : array-like, shape (3,) or (i, 3)
        Segment end points in units (m).
    currents : float | array-like, shape (i,)
        Electric currents in units (A).
    positions : array-like, shape (3,) or (i, 3), default (0, 0, 0)
        Local coordinate origin in units (m).
    orientations : None | Rotation, default None
        Local coordinate orientations. If ``None``, the unit rotation is
        applied.
    squeeze : bool, default True
        If ``True``, squeeze singleton axis when ``i=1``.

    Notes
    -----
    Returns (0, 0, 0) on the line segments.

    Returns
    -------
    ndarray, shape (3,) or (i, 3)
        Field at the observer locations.

    Examples
    --------
    >>> import numpy as np
    >>> from magpylib.func import polyline_field
    >>> B = polyline_field(
    ...     field='B',
    ...     observers=(0.2, 0.3, 0.1),
    ...     segments_start=[(-0.5, -1.0, 0), (-1.0, -1.0, 0)],
    ...     segments_end=[(0.5, 1.0, 0), (1.0, 1.0, 0)],
    ...     currents=1e6
    ... )
    >>> with np.printoptions(precision=3):
    ...     print(B)
    [[ 1.481 -0.741 -0.741]
     [ 0.939 -0.939  0.939]]
    """
    params = {
        "observers": observers,
        "segment_start": segments_start,
        "segment_end": segments_end,
        "current": currents,
        "position": positions,
        "orientation": orientations,
    }
    return _getBH_func(_BHJM_current_polyline, field, params, squeeze)


def cuboid_field(
    field,
    observers,
    dimensions,
    polarizations,
    positions=(0, 0, 0),
    orientations=None,
    squeeze=True,
):
    """Return B- or H-field of cuboid magnets for i instances.

    With ``positions=(0, 0, 0)`` and ``orientations=None`` the cuboid sides are
    parallel to the coordinate axes and the geometric center lies at the origin.

    Parameters
    ----------
    field : {'B', 'H'}
        Select which field is returned.
    observers : array-like, shape (3,) or (i, 3)
        Points where the field is evaluated in units (m).
    dimensions : array-like, shape (3,) or (i, 3)
        Cuboid sides (a, b, c) in units (m).
    polarizations : array-like, shape (3,) or (i, 3)
        Magnetic polarization in units (T).
    positions : array-like, shape (3,) or (i, 3), default (0, 0, 0)
        Cuboid centers in units (m).
    orientations : None | Rotation, default None
        Cuboid orientations. If ``None``, the unit rotation is applied.
    squeeze : bool, default True
        If ``True``, squeeze singleton axis when ``i=1``.

    Returns
    -------
    ndarray, shape (3,) or (i, 3)
        Field at the observer locations.

    Notes
    -----
    Returns (0, 0, 0) on edges and corners.

    Examples
    --------
    >>> import numpy as np
    >>> from magpylib.func import cuboid_field
    >>> B = cuboid_field(
    ...     field='B',
    ...     observers=(0.2, 1.3, 1.1),
    ...     dimensions=[(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)],
    ...     polarizations=(0.0, 0.0, 1.0),
    ... )
    >>> with np.printoptions(precision=3):
    ...     print(B)
    [[ 0.003  0.023  0.004]
     [ 0.017  0.206 -0.006]]
    """
    params = {
        "observers": observers,
        "dimension": dimensions,
        "polarization": polarizations,
        "position": positions,
        "orientation": orientations,
    }
    shapes = {"dimension": (3,)}
    return _getBH_func(_BHJM_magnet_cuboid, field, params, squeeze, shapes)


def cylinder_field(
    field,
    observers,
    dimensions,
    polarizations,
    positions=(0, 0, 0),
    orientations=None,
    squeeze=True,
):
    """Return B- or H-field of solid cylinder magnets for i instances.

    With ``positions=(0, 0, 0)`` and ``orientations=None`` the cylinder axis
    coincides with the global z-axis and the geometric center lies at the
    origin.

    Parameters
    ----------
    field : {'B', 'H'}
        Select which field is returned.
    observers : array-like, shape (3,) or (i, 3)
        Points where the field is evaluated in units (m).
    dimensions : array-like, shape (2,) or (i, 2)
        Cylinder dimensions (diameter, height) in units (m).
    polarizations : array-like, shape (3,) or (i, 3)
        Magnetic polarization in units (T).
    positions : array-like, shape (3,) or (i, 3), default (0, 0, 0)
        Cylinder centers in units (m).
    orientations : None | Rotation, default None
        Cylinder orientations. If ``None``, the unit rotation is applied.
    squeeze : bool, default True
        If ``True``, squeeze singleton axis when ``i=1``.

    Returns
    -------
    ndarray, shape (3,) or (i, 3)
        Field at the observer locations.

    Notes
    -----
    Returns (0, 0, 0) on edges.

    Examples
    --------
    >>> import numpy as np
    >>> from magpylib.func import cylinder_field
    >>> B = cylinder_field(
    ...     field='B',
    ...     observers=(0.2, 1.3, 1.1),
    ...     dimensions=[(1.0, 1.0), (2.0, 2.0)],
    ...     polarizations=(0.0, 0.0, 1.0),
    ... )
    >>> with np.printoptions(precision=3):
    ...     print(B)
    [[ 0.003  0.018  0.002]
     [ 0.026  0.17  -0.006]]
    """
    params = {
        "observers": observers,
        "dimension": dimensions,
        "polarization": polarizations,
        "position": positions,
        "orientation": orientations,
    }
    shapes = {"dimension": (2,)}
    return _getBH_func(_BHJM_magnet_cylinder, field, params, squeeze, shapes)


def cylinder_segment_field(
    field,
    observers,
    dimensions,
    polarizations,
    positions=(0, 0, 0),
    orientations=None,
    squeeze=True,
):
    """Return B- or H-field of cylinder-segment magnets for i instances.

    With ``positions=(0, 0, 0)`` and ``orientations=None`` the segment's
    cylinder axis coincides with the global z-axis and the geometric center
    lies at the origin.

    Parameters
    ----------
    field : {'B', 'H'}
        Select which field is returned.
    observers : array-like, shape (3,) or (i, 3)
        Points where the field is evaluated in units (m).
    dimensions : array-like, shape (5,) or (i, 5)
        Segment dimensions (r1, r2, h, φ1, φ2) where r1 < r2 are inner
        and outer radii in units (m), h is the height in units (m), and
        φ1 < φ2 are azimuth section angles in radians (rad).
    polarizations : array-like, shape (3,) or (i, 3)
        Magnetic polarization in units (T).
    positions : array-like, shape (3,) or (i, 3), default (0, 0, 0)
        Cylinder centers in units (m).
    orientations : None | Rotation, default None
        Magnet orientations. If ``None``, the unit rotation is applied.
    squeeze : bool, default True
        If ``True``, squeeze singleton axis when ``i=1``.

    Returns
    -------
    ndarray, shape (3,) or (i, 3)
        Field at the observer locations.

    Notes
    -----
    Returns (0, 0, 0) on surface, edges, and corners.

    Examples
    --------
    >>> import numpy as np
    >>> from magpylib.func import cylinder_segment_field
    >>> B = cylinder_segment_field(
    ...     field='B',
    ...     observers=(0.2, 0.3, 0.1),
    ...     dimensions=[(1.0, 2.0, 1.0, 45, 225), (1.0, 2.0, 1.0, 90, 270)],
    ...     polarizations=(0.0, 0.0, 1.0),
    ... )
    >>> with np.printoptions(precision=3):
    ...    print(B)
    [[ 0.006 -0.017 -0.122]
     [ 0.011 -0.004 -0.084]]
    """
    params = {
        "observers": observers,
        "dimension": dimensions,
        "polarization": polarizations,
        "position": positions,
        "orientation": orientations,
    }
    shapes = {"dimension": (5,)}
    return _getBH_func(_BHJM_cylinder_segment, field, params, squeeze, shapes)


def sphere_field(
    field,
    observers,
    diameters,
    polarizations,
    positions=(0, 0, 0),
    orientations=None,
    squeeze=True,
):
    """Return B- or H-field of sphere magnets for i instances.

    With ``positions=(0, 0, 0)`` and ``orientations=None`` local and global
    coordinates coincide and the sphere center lies at the origin.

    Parameters
    ----------
    field : {'B', 'H'}
        Select which field is returned.
    observers : array-like, shape (3,) or (i, 3)
        Points where the field is evaluated in units (m).
    diameters : float | array-like, shape (i,)
        Sphere diameters in units (m).
    polarizations : array-like, shape (3,) or (i, 3)
        Magnetic polarization in units (T).
    positions : array-like, shape (3,) or (i, 3), default (0, 0, 0)
        Sphere centers in units (m).
    orientations : None | Rotation, default None
        Magnet orientations. If ``None``, the unit rotation is applied.
    squeeze : bool, default True
        If ``True``, squeeze singleton axis when ``i=1``.

    Returns
    -------
    ndarray, shape (3,) or (i, 3)
        Field at the observer locations.

    Examples
    --------
    >>> import numpy as np
    >>> from magpylib.func import sphere_field
    >>> B = sphere_field(
    ...     field='B',
    ...     observers=(1.2, 0.3, 0.1),
    ...     diameters=(1.0, 1.5),
    ...     polarizations=(0.0, 0.0, 1.0),
    ... )
    >>> with np.printoptions(precision=3):
    ...     print(B)
    [[ 0.005  0.001 -0.021]
     [ 0.017  0.004 -0.072]]
    """
    params = {
        "observers": observers,
        "diameter": diameters,
        "polarization": polarizations,
        "position": positions,
        "orientation": orientations,
    }
    return _getBH_func(_BHJM_magnet_sphere, field, params, squeeze)


def tetrahedron_field(
    field,
    observers,
    vertices,
    polarizations,
    positions=(0, 0, 0),
    orientations=None,
    squeeze=True,
):
    """Return B- or H-field of tetrahedron magnets for i instances.

    With ``positions=(0, 0, 0)`` and ``orientations=None`` local and global
    coordinates coincide. The tetrahedron is defined by four vertices in the
    local coordinate system.

    Parameters
    ----------
    field : {'B', 'H'}
        Select which field is returned.
    observers : array-like, shape (3,) or (i, 3)
        Points where the field is evaluated in units (m).
    vertices : array-like, shape (4, 3) or (i, 4, 3)
        Vertices of the tetrahedra in units (m).
    polarizations : array-like, shape (3,) or (i, 3)
        Magnetic polarization in units (T).
    positions : array-like, shape (3,) or (i, 3), default (0, 0, 0)
        Position of local coordinate origin in units (m).
    orientations : None | Rotation, default None
        Orientation of local coordinates. If ``None``, the unit
        rotation is applied.
    squeeze : bool, default True
        If ``True``, squeeze singleton axis when ``i=1``.

    Returns
    -------
    ndarray, shape (3,) or (i, 3)
        Field at the observer locations.

    Notes
    -----
    Returns (0, 0, 0) on corners.

    Examples
    --------
    >>> import numpy as np
    >>> from magpylib.func import tetrahedron_field
    >>> B = tetrahedron_field(
    ...     field='B',
    ...     observers=(-.2, 0.3, 0.1),
    ...     vertices=[
    ...         ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ...         ((0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2)),
    ...     ],
    ...     polarizations=(0.0, 0.0, 1.0),
    ... )
    >>> with np.printoptions(precision=3):
    ...     print(B)
    [[ 0.065 -0.007 -0.075]
     [ 0.146  0.026 -0.09 ]]
    """
    params = {
        "observers": observers,
        "vertices": vertices,
        "polarization": polarizations,
        "position": positions,
        "orientation": orientations,
    }
    shapes = {"vertices": (4, 3)}
    return _getBH_func(_BHJM_magnet_tetrahedron, field, params, squeeze, shapes)


def dipole_field(
    field,
    observers,
    moments,
    positions=(0, 0, 0),
    orientations=None,
    squeeze=True,
):
    """Return B- or H-field of magnetic dipoles for i instances.

    With ``positions=(0, 0, 0)`` and ``orientations=None`` local and global
    coordinates coincide and the dipole lies at the origin.

    Parameters
    ----------
    field : {'B', 'H'}
        Select which field is returned.
    observers : array-like, shape (3,) or (i, 3)
        Points where the field is evaluated in units (m).
    moments : array-like, shape (3,) or (i, 3)
        Magnetic dipole moments in units (A m²).
    positions : array-like, shape (3,) or (i, 3), default (0, 0, 0)
        Dipole positions in units (m).
    orientations : None | Rotation, default None
        Orientation of local coordinates. If ``None``, the unit
        rotation is applied.
    squeeze : bool, default True
        If ``True``, squeeze singleton axis when ``i=1``.

    Returns
    -------
    ndarray, shape (3,) or (i, 3)
        Field at the observer locations.

    Notes
    -----
    Returns inf or nan at the dipole position.

    Examples
    --------
    >>> import numpy as np
    >>> from magpylib.func import dipole_field
    >>> B = dipole_field(
    ...     field='B',
    ...     observers=(1.2, 0.3, 0.1),
    ...     moments=[(1e6, 0, 0), (0, 1e6, 0)]
    ... )
    >>> with np.printoptions(precision=3):
    ...     print(B)
    [[ 0.094  0.037  0.012]
     [ 0.037 -0.043  0.003]]
    """
    params = {
        "observers": observers,
        "moment": moments,
        "position": positions,
        "orientation": orientations,
    }
    return _getBH_func(_BHJM_dipole, field, params, squeeze)


def triangle_charge_field(
    field,
    observers,
    vertices,
    polarizations,
    positions=(0, 0, 0),
    orientations=None,
    squeeze=True,
):
    """Return B- or H-field of magnetically charged triangles for i instances.

    With ``positions=(0, 0, 0)`` and ``orientations=None`` local and global
    coordinates coincide. Triangles are defined by three vertices in the
    local coordinates.

    Parameters
    ----------
    field : {'B', 'H'}
        Select which field is returned.
    observers : array-like, shape (3,) or (i, 3)
        Points where the field is evaluated in units (m).
    vertices : array-like, shape (3, 3) or (i, 3, 3)
        Triangle vertices [(V1a, V1b, V1c), (V2a, V2b, V2c), ...] in
        units (m).
    polarizations : array-like, shape (3,) or (i, 3)
        Magnetic polarization in units (T).
    positions : array-like, shape (3,) or (i, 3), default (0, 0, 0)
        Position of local coordinate origin in units (m).
    orientations : None | Rotation, default None
        Orientation of local coordinates. If ``None``, the unit
        rotation is applied.
    squeeze : bool, default True
        If ``True``, squeeze singleton axis when ``i=1``.

    Returns
    -------
    ndarray, shape (3,) or (i, 3)
        Field at the observer locations.

    Notes
    -----
    Returns (0, 0, 0) on corners.

    Examples
    --------
    >>> import numpy as np
    >>> from magpylib.func import triangle_charge_field
    >>> B = triangle_charge_field(
    ...     field='B',
    ...     observers=(1.2, 0.3, 0.1),
    ...     vertices=[
    ...         ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
    ...         ((0, 0, 0), (2, 0, 0), (0, 2, 0)),
    ...         ],
    ...     polarizations=[0, 0, 1],
    ... )
    >>> with np.printoptions(precision=3):
    ...     print(B)
    [[ 0.06   0.009  0.01 ]
     [ 0.11  -0.074  0.403]]
    """
    params = {
        "observers": observers,
        "vertices": vertices,
        "polarization": polarizations,
        "position": positions,
        "orientation": orientations,
    }
    shapes = {"vertices": (3, 3)}
    return _getBH_func(_BHJM_triangle, field, params, squeeze, shapes)


def triangle_current_field(
    field,
    observers,
    vertices,
    current_densities,
    positions=(0, 0, 0),
    orientations=None,
    squeeze=True,
):
    """Return B- or H-field of triangular current sheets for i instances.

    With ``positions=(0, 0, 0)`` and ``orientations=None`` local and global
    coordinates coincide. Triangles are defined by three vertices in the
    local coordinates.

    Parameters
    ----------
    field : {'B', 'H'}
        Select which field is returned.
    observers : array-like, shape (3,) or (i, 3)
        Points where the field is evaluated in units (m).
    vertices : array-like, shape (3, 3) or (i, 3, 3)
        Triangle vertices in units (m).
    current_densities : array-like, shape (3,) or (i, 3)
        Surface current densities in units (A/m²).
    positions : array-like, shape (3,) or (i, 3), default (0, 0, 0)
        Position of local coordinate origin in units (m).
    orientations : None | Rotation, default None
        Orientation of local coordinates. If ``None``, the unit
        rotation is applied.
    squeeze : bool, default True
        If ``True``, squeeze singleton axis when ``i=1``.

    Returns
    -------
    ndarray, shape (3,) or (i, 3)
        Field at the observer locations.

    Notes
    -----
    Returns (0, 0, 0) on a sheet.

    Examples
    --------
    >>> import numpy as np
    >>> from magpylib.func import triangle_current_field
    >>> B = triangle_current_field(
    ...     field='B',
    ...     observers=(1.2, 0.3, 0.1),
    ...     vertices=[
    ...         ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
    ...         ((0, 0, 0), (2, 0, 0), (0, 2, 0)),
    ...         ],
    ...     current_densities=(1e6, 1e6, 1e6),
    ... )
    >>> with np.printoptions(precision=3):
    ...     print(B)
    [[ 0.012 -0.012 -0.065]
     [ 0.506 -0.506 -0.231]]
    """
    params = {
        "observers": observers,
        "vertices": vertices,
        "current_densities": current_densities,
        "position": positions,
        "orientation": orientations,
    }
    shapes = {"vertices": (3, 3)}
    return _getBH_func(_BHJM_current_sheet, field, params, squeeze, shapes)


DIM = {
    "position": 2,
    "orientation": 2,
    "observers": 2,
    "diameter": 1,
    "current": 1,
    "segment_start": 2,
    "segment_end": 2,
    "polarization": 2,
    "dimension": 2,
    "moment": 2,
    "vertices": 3,
    "current_densities": 2,
}
SHAPE = {
    "position": (3,),
    "orientation": (4,),
    "observers": (3,),
    "diameter": (),
    "current": (),
    "segment_start": (3,),
    "segment_end": (3,),
    "polarization": (3,),
    "moment": (3,),
    "current_densities": (3,),
}


def _getBH_func(field_func, field, params, squeeze, shapes=None):
    """Functional interface for magnetic field computation

    field_func: callable
    field: {"B", "H"}
    params: dict
        contains all kwargs required for field_func() call
    squeeze: bool
    shapes: dict | None, default None
        extra shapes that are class specific (e.g. dimension can
        have multiple shapes).
    """
    # Check field input
    if field not in {"B", "H"}:
        msg = f"Input field must be one of {'B', 'H'}; instead received {field!r}."
        raise ValueError(msg)

    # Check orientation input
    if params["orientation"] is None:
        params["orientation"] = R.identity()
    if not isinstance(params["orientation"], R):
        msg = (
            "Input orientation must be a SciPy Rotation instance or None; "
            f"instead received type {type(params['orientation']).__name__}."
        )
        raise TypeError(msg)

    # Transform Rotation to Quat
    params["orientation"] = params["orientation"].as_quat()

    # Transform all inputs to ndarray
    for key, val in params.items():
        try:
            if not isinstance(val, np.ndarray):
                params[key] = np.array(val, dtype=float)
        except ValueError as err:
            msg = f"Input {key} must be array-like; instead received {val!r}."
            raise ValueError(msg) from err

    # Tile missing ndims, Find maxlength
    nmax = 1
    for key, val in params.items():
        if val.ndim < DIM[key]:
            params[key] = np.expand_dims(val, axis=0)
        if val.ndim > DIM[key]:
            msg = (
                f"Input {key} must have at most {DIM[key]} dimensions; "
                f"instead received ndim {val.ndim}."
            )
            raise ValueError(msg)
        # store maxlength
        n = params[key].shape[0]
        nmax = n if n > nmax else nmax

    # Check if shapes are correct, Tile to maxlength
    # include extra shapes
    SHAPES = SHAPE if shapes is None else SHAPE | shapes
    for key, val in params.items():
        if val.shape[1:] != SHAPES[key]:
            msg = (
                f"Input {key} must have shape {SHAPES[key]} for ndim > 0; "
                f"instead received shape {val.shape[1:]}"
            )
            raise ValueError(msg)
        if val.shape[0] not in (1, nmax):
            msg = (
                f"Input {key} must have 1 or {nmax} instances; "
                f"instead received {val.shape[0]}."
            )
            raise ValueError(msg)
        # tile up to nmax if only 1 instance is given
        if nmax > 1 and val.shape[0] == 1:
            params[key] = np.tile(val, [nmax] + [1] * (val.ndim - 1))

    # Transform Quat to Rotation object
    params["orientation"] = R.from_quat(params["orientation"])

    # Call to level1, squeeze, return
    field = _getBH_level1(field_func=field_func, field=field, **params)

    if squeeze:
        return np.squeeze(field)
    return field


# REMOVE IN FUTURE VERSIONS ############################################
# REMOVE IN FUTURE VERSIONS ############################################
# REMOVE IN FUTURE VERSIONS ############################################


def _getBH_dict_level2(
    source_type,
    observers,
    *,
    field: str,
    position=(0, 0, 0),
    orientation=None,
    squeeze=True,
    in_out="auto",
    **kwargs: dict,
) -> np.ndarray:
    """Functional interface access to vectorized computation

    Parameters
    ----------
    kwargs: dict that describes the computation.

    Returns
    -------
    field: ndarray, shape (o, 3), field at obs_pos in (T) or (A/m)

    Info
    ----
    - check inputs

    - secures input types (list/tuple -> ndarray)
    - test if mandatory inputs are there
    - sets default input variables (e.g. pos, rot) if missing
    - tiles 1D inputs vectors to correct dimension
    """
    # pylint: disable=protected-access
    # pylint: disable=too-many-branches

    # generate dict of secured inputs for auto-tiling ---------------
    #  entries in this dict will be tested for input length, and then
    #  be automatically tiled up and stored back into kwargs for calling
    #  _getBH_level1().
    #  To allow different input dimensions, the ndim argument is also given
    #  which tells the program which dimension it should tile up.

    # pylint: disable=import-outside-toplevel
    if orientation is None:
        orientation = R.identity()
    try:
        source_classes = get_registered_sources()
        field_func = source_classes[source_type]._field_func
        field_func_kwargs_ndim = {"position": 2, "orientation": 2, "observers": 2}
        field_func_kwargs_ndim.update(
            source_classes[source_type]._field_func_kwargs_ndim
        )
    except KeyError as err:
        msg = (
            f"Input source_type must be one of {list(source_classes)}; "
            f"instead received {source_type!r}."
        )
        raise MagpylibBadUserInput(msg) from err

    kwargs["observers"] = observers
    kwargs["position"] = position

    # change orientation to Rotation NumPy array for tiling
    kwargs["orientation"] = orientation.as_quat()

    # evaluation vector lengths
    vec_lengths = {}
    ragged_seq = {}
    for key, val_item in kwargs.items():
        val = val_item
        try:
            if (
                not isinstance(val, numbers.Number)
                and not isinstance(val[0], numbers.Number)
                and any(len(o) != len(val[0]) for o in val)
            ):
                ragged_seq[key] = True
                val = np.array([np.array(v, dtype=float) for v in val], dtype="object")
            else:
                ragged_seq[key] = False
                val = np.array(val, dtype=float)
        except TypeError as err:
            msg = f"Input {key} must be array-like; instead received {val!r}."
            raise MagpylibBadUserInput(msg) from err
        expected_dim = field_func_kwargs_ndim.get(key, 1)
        if val.ndim == expected_dim or ragged_seq[key]:
            if len(val) == 1:
                val = np.squeeze(val)
            else:
                vec_lengths[key] = len(val)

        kwargs[key] = val

    if len(set(vec_lengths.values())) > 1:
        msg = (
            "Input arrays must have length 1 or the same length; "
            f"instead received lengths {vec_lengths}."
        )
        raise MagpylibBadUserInput(msg)
    vec_len = max(vec_lengths.values(), default=1)
    # tile 1D inputs and replace original values in kwargs
    for key, val in kwargs.items():
        expected_dim = field_func_kwargs_ndim.get(key, 1)
        if val.ndim < expected_dim and not ragged_seq[key]:
            kwargs[key] = np.tile(val, (vec_len, *[1] * (expected_dim - 1)))

    # change orientation back to Rotation object
    kwargs["orientation"] = R.from_quat(kwargs["orientation"])

    # compute and return B
    B = _getBH_level1(field=field, field_func=field_func, in_out=in_out, **kwargs)

    if B is not None and squeeze:
        return np.squeeze(B)
    return B


def _getBH_level1(
    *,
    field_func: Callable,
    field: str,
    position: np.ndarray,
    orientation: np.ndarray,
    observers: np.ndarray,
    **kwargs: dict,
) -> np.ndarray:
    """
    COPY FROM field_BH TO AVOID CIRCULAR IMPORT

    Vectorized field computation

    - applies spatial transformations global CS <-> source CS
    - selects the correct Bfield_XXX function from input

    Args
    ----
    kwargs: dict of shape (N, x) input vectors that describes the computation.

    Returns
    -------
    field: ndarray, shape (N, 3)

    """

    # transform obs_pos into source CS
    pos_rel_rot = orientation.apply(observers - position, inverse=True)

    # filter arguments
    if not has_parameter(field_func, "in_out"):  # in_out passed only to magnets
        kwargs.pop("in_out", None)

    # compute field
    BH = field_func(field=field, observers=pos_rel_rot, **kwargs)

    # transform field back into global CS
    if BH is not None:  # catch non-implemented field_func a level above
        BH = orientation.apply(BH)

    return BH

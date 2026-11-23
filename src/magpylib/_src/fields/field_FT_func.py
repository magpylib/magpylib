"""Force and torque computation functional interface"""

import numpy as np
from array_api_compat import (
    array_namespace,
    device,
    is_array_api_obj,
    is_jax_namespace,
    is_numpy_namespace,
    is_torch_namespace,
)
from scipy.constants import mu_0 as MU0
from scipy.spatial.transform import Rotation as R

from magpylib._src.array_api_utils import (
    constant,
    input_namespace,
    rotate,
    xp_promote,
)
from magpylib._src.obj_classes.class_BaseTarget import (
    check_meshing_count,
    check_meshing_count_or_triple,
)
from magpylib._src.obj_classes.target_meshing import (
    _circle_segments,
    _cuboid_cells,
    _cylinder_segment_cells,
    _tetrahedron_cells,
)


def _to_namespace(*values):
    """Return the namespace of the inputs and the inputs as its floating
    arrays, None staying None: lists, scalars and NumPy arrays follow the
    arrays of another library, as in the field functions."""
    xp = input_namespace(*values)
    values = [
        val if val is None or is_array_api_obj(val) else np.array(val, dtype=float)
        for val in values
    ]
    if is_numpy_namespace(xp):
        return xp, [val if val is None else xp_promote(val, xp=xp) for val in values]
    arrays = [
        val for val in values if val is not None and not isinstance(val, np.ndarray)
    ]
    promoted = xp_promote(*arrays, xp=xp)
    dtype = (promoted if len(arrays) == 1 else promoted[0]).dtype

    def convert(val):
        if val is None:
            return None
        if isinstance(val, np.ndarray):
            return xp.asarray(val, dtype=dtype, device=device(arrays[0]))
        return xp.astype(val, dtype, copy=False)

    return xp, [convert(val) for val in values]


def _create_eps_vector(eps):
    """Create a vector of shape (7, 3) for finite difference calculations with step size eps."""
    return np.array(
        [
            (0, 0, 0),
            (eps, 0, 0),
            (-eps, 0, 0),
            (0, eps, 0),
            (0, -eps, 0),
            (0, 0, eps),
            (0, 0, -eps),
        ]
    )


def _check_eps(eps):
    """
    check FD step
    """
    if not isinstance(eps, float):
        msg = f"Input eps must be a float; instead received {eps!r}."
        raise TypeError(msg)
    if eps <= 0:
        msg = f"Input eps must be a positive float; instead received {eps!r}."
        raise ValueError(msg)


def _field_gradient(field, points, eps):
    """B-field and its gradient at the given points.

    JAX and PyTorch arrays are differentiated in forward mode, one direction
    for all points at once, which takes that the field at a point depends on
    this point only. Other arrays get central finite differences from one
    field call.

    Parameters
    ----------
    field : callable
        Returns the B-field at points of shape (..., n, 3), with shape
        (..., n, 3) and possibly more leading axes.
    points : array, shape (..., n, 3)
        Points where the field and its gradient are evaluated.
    eps : float
        Finite difference step, only used for NumPy arrays.

    Returns
    -------
    tuple[array, array]
        The field, shape (..., n, 3), and its gradient, shape (..., n, 3, 3),
        with the derivative along axis i of the field component k at [i, k].
    """
    xp = array_namespace(points)
    if is_jax_namespace(xp) or is_torch_namespace(xp):
        units = [
            xp.broadcast_to(unit, points.shape)
            for unit in xp.unstack(xp.eye(3, dtype=points.dtype, device=device(points)))
        ]
        if is_jax_namespace(xp):
            import jax  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

            B, jvp = jax.linearize(field, points)
            return B, xp.stack([jvp(unit) for unit in units], axis=-2)
        # torch.func.linearize cannot trace the loops of the field kernels
        out = [xp.func.jvp(field, (points,), (unit,)) for unit in units]
        return out[0][0], xp.stack([dB for _, dB in out], axis=-2)

    obs = points[..., xp.newaxis, :] + _create_eps_vector(eps)
    B = field(xp.reshape(obs, (*points.shape[:-2], -1, 3)))
    B = xp.reshape(B, (*B.shape[:-2], points.shape[-2], 7, 3))
    return B[..., 0, :], (B[..., 1::2, :] - B[..., 2::2, :]) / (2 * eps)


def _cell_force_torque(field, points, *, moments=None, cvecs=None, pivot=None, eps):
    """Force and torque on each cell of a target, see ``force_torque``.

    ``points``, ``moments``, ``cvecs`` and ``pivot`` are arrays of one
    namespace that broadcast to shape (..., n, 3). The force and torque have
    the shape of the field, and are not summed over the cells.
    """
    xp = array_namespace(points)
    if moments is not None:
        B, dB = _field_gradient(field, points, eps)
        F = xp.matmul(dB, moments[..., xp.newaxis])[..., 0]
        T = xp.linalg.cross(moments, B)
    else:
        F = xp.linalg.cross(cvecs, field(points))
        T = xp.zeros_like(F)
    if pivot is not None:
        T = T + xp.linalg.cross(points - pivot, F)
    return F, T


def force_torque(
    field,
    points,
    *,
    moments=None,
    current_vectors=None,
    pivot=None,
    eps=1e-5,
):
    """Return force and torque on a target that is cut into n cells.

    A magnet is given by the magnetic moments of its cells and a current by
    the current vectors of its segments, as the mesh functions of
    ``magpylib.func`` return them. With JAX or PyTorch arrays the computation
    runs in that library, the gradient of the field comes from automatic
    differentiation, and the result can be differentiated and compiled.

    Parameters
    ----------
    field : callable
        Returns the B-field in units (T) at the observers given as array of
        shape (n, 3), with shape (n, 3). The field at an observer must depend
        on this observer only, as it does for the field functions of
        ``magpylib.func``.
    points : array-like, shape (n, 3)
        Cell positions in units (m).
    moments : None | array-like, shape (n, 3), default None, keyword-only
        Magnetic moments of the cells of a magnet in units (A*m²).
    current_vectors : None | array-like, shape (n, 3), default None, keyword-only
        Currents times the segment vectors of a current in units (A*m).
    pivot : None | array-like, shape (3,), default None, keyword-only
        Pivot point through which the force contributes to the torque. If
        ``None``, no pivot is applied (may yield nonphysical results).
    eps : float, default 1e-5, keyword-only
        Finite-difference step size for the gradient of the field at magnet
        cells. Only used with NumPy arrays.

    Returns
    -------
    tuple[array, array]
        Force in units (N) and torque in units (N*m), each of shape (3,), as
        arrays of the library of the inputs.

    Notes
    -----
    The force and torque on a magnet are F = (gradB) · MOM and
    T = MOM x B + r x F, and on a current F = CVEC x B and T = r x F, summed
    over the cells, with r the cell position seen from the pivot.

    Examples
    --------
    The force on a magnetic moment on the axis of a current loop:

    >>> import numpy as np
    >>> import magpylib as magpy
    >>> def field(observers):
    ...     return magpy.func.circle_field('B', observers, 2.0, 1e3)
    >>> F, T = magpy.func.force_torque(
    ...     field,
    ...     points=[(0.0, 0.0, 1.0)],
    ...     moments=[(0.0, 0.0, 1e3)],
    ... )
    >>> print(f'force: {np.round(F, decimals=4) + 0.0} N')
    force: [ 0.      0.     -0.3332] N
    """
    if (moments is None) == (current_vectors is None):
        msg = (
            "Input moments or current_vectors of force_torque must be given, and "
            "only one of them."
        )
        raise ValueError(msg)
    _check_eps(eps)
    _, (points, moments, current_vectors, pivot) = _to_namespace(
        points, moments, current_vectors, pivot
    )
    if points.ndim != 2 or points.shape[1] != 3:
        msg = (
            "Input points of force_torque must have shape (n, 3); instead "
            f"received shape {tuple(points.shape)}."
        )
        raise ValueError(msg)
    F, T = _cell_force_torque(
        field, points, moments=moments, cvecs=current_vectors, pivot=pivot, eps=eps
    )
    xp = array_namespace(F)
    return xp.sum(F, axis=-2), xp.sum(T, axis=-2)


# MESHES ######################################################################


def _check_shapes(func, shapes, **inputs):
    """Check the shapes of the inputs of a mesh function."""
    for (name, value), shape in zip(inputs.items(), shapes, strict=True):
        if tuple(value.shape) != shape:
            msg = (
                f"Input {name} of {func} must have shape {shape}; instead "
                f"received shape {tuple(value.shape)}."
            )
            raise ValueError(msg)


def _place(points, vectors, position, orientation):
    """Return the cell positions and the moments or current vectors of a mesh
    in the local frame of its target at the position and orientation of the
    target."""
    if orientation is not None:
        if not isinstance(orientation, R):
            msg = (
                "Input orientation must be a SciPy Rotation instance or None; "
                f"instead received type {type(orientation).__name__}."
            )
            raise TypeError(msg)
        if orientation.as_quat().size != 4:
            msg = (
                "Input orientation of a mesh must be a single rotation; instead "
                f"received {len(orientation)} rotations."
            )
            raise ValueError(msg)
        quat = constant(orientation.as_quat().reshape(1, 4), points)
        points, vectors = rotate(quat, points), rotate(quat, vectors)
    return points + position, vectors


def _magnet_mesh(cells, polarization, position, orientation):
    """Return the placed cell centroids and magnetic moments of magnet cells."""
    moments = cells["volumes"][:, np.newaxis] * (polarization / MU0)
    return _place(cells["centroids"], moments, position, orientation)


def cuboid_mesh(dimension, polarization, meshing, position=(0, 0, 0), orientation=None):
    """Return the cells of a cuboid magnet for the force computation.

    The cuboid is cut into cuboid cells. With ``position=(0, 0, 0)`` and
    ``orientation=None`` its sides are parallel to the coordinate axes and
    its geometric center lies at the origin.

    Parameters
    ----------
    dimension : array-like, shape (3,)
        Cuboid sides (a, b, c) in units (m).
    polarization : array-like, shape (3,)
        Magnetic polarization in units (T).
    meshing : int | tuple of int, shape (3,)
        Target number of cells, or the number of divisions (n1, n2, n3) along
        the sides. An integer gives cells that are as close to cubes as
        possible.
    position : array-like, shape (3,), default (0, 0, 0)
        Cuboid center in units (m).
    orientation : None | Rotation, default None
        Cuboid orientation. If ``None``, the unit rotation is applied.

    Returns
    -------
    tuple[array, array]
        Cell positions in units (m) and magnetic moments of the cells in
        units (A*m²), each of shape (n, 3), as inputs of ``force_torque``.

    Notes
    -----
    With JAX or PyTorch arrays the mesh is computed in that library and can be
    differentiated with respect to the inputs. An integer ``meshing`` takes
    the number of divisions from the values of ``dimension``, which have to
    be known: under ``jax.jit`` give the divisions.

    Examples
    --------
    >>> import magpylib as magpy
    >>> points, moments = magpy.func.cuboid_mesh(
    ...     dimension=(1.0, 1.0, 2.0),
    ...     polarization=(0.0, 0.0, 1.0),
    ...     meshing=(1, 1, 2),
    ... )
    >>> print(points)
    [[ 0.   0.  -0.5]
     [ 0.   0.   0.5]]
    >>> print(moments.round())
    [[     0.      0. 795775.]
     [     0.      0. 795775.]]
    """
    meshing = check_meshing_count_or_triple(meshing, "cuboid_mesh")
    _, (dimension, polarization, position) = _to_namespace(
        dimension, polarization, position
    )
    _check_shapes(
        "cuboid_mesh",
        ((3,), (3,), (3,)),
        dimension=dimension,
        polarization=polarization,
        position=position,
    )
    cells = _cuboid_cells(dimension, meshing)
    return _magnet_mesh(cells, polarization, position, orientation)


def cylinder_mesh(
    dimension, polarization, meshing, position=(0, 0, 0), orientation=None
):
    """Return the cells of a cylinder magnet for the force computation.

    The cylinder is cut into cylinder-segment cells in rings and layers. With
    ``position=(0, 0, 0)`` and ``orientation=None`` its axis coincides with
    the z-axis and its geometric center lies at the origin.

    Parameters
    ----------
    dimension : array-like, shape (2,)
        Cylinder dimension (diameter, height) in units (m).
    polarization : array-like, shape (3,)
        Magnetic polarization in units (T).
    meshing : int | tuple of int, shape (3,)
        Target number of cells, or the divisions (nphi, nr, nh) along the
        circumference at half the radius, the radius and the height. An
        integer gives cells that are as close to cubes as possible.
    position : array-like, shape (3,), default (0, 0, 0)
        Cylinder center in units (m).
    orientation : None | Rotation, default None
        Cylinder orientation. If ``None``, the unit rotation is applied.

    Returns
    -------
    tuple[array, array]
        Cell positions in units (m), which are the cell centroids, and
        magnetic moments of the cells in units (A*m²), each of shape (n, 3),
        as inputs of ``force_torque``.

    Notes
    -----
    With JAX or PyTorch arrays the mesh is computed in that library and can be
    differentiated with respect to the inputs. The number of cells in each
    ring is taken from the values of ``dimension``, which have to be known:
    the mesh cannot be created under ``jax.jit``.

    Examples
    --------
    >>> import magpylib as magpy
    >>> points, moments = magpy.func.cylinder_mesh(
    ...     dimension=(1.0, 2.0),
    ...     polarization=(0.0, 0.0, 1.0),
    ...     meshing=(1, 1, 2),
    ... )
    >>> print(points.round(3) + 0.0)
    [[ 0.   0.  -0.5]
     [ 0.   0.   0.5]]
    >>> print(moments.round())
    [[     0.      0. 625000.]
     [     0.      0. 625000.]]
    """
    meshing = check_meshing_count_or_triple(meshing, "cylinder_mesh")
    xp, (dimension, polarization, position) = _to_namespace(
        dimension, polarization, position
    )
    _check_shapes(
        "cylinder_mesh",
        ((2,), (3,), (3,)),
        dimension=dimension,
        polarization=polarization,
        position=position,
    )
    zero = dimension[0] * 0
    segment = xp.stack([zero, dimension[0] / 2, dimension[1], zero, zero + 360])
    cells = _cylinder_segment_cells(segment, meshing)
    return _magnet_mesh(cells, polarization, position, orientation)


def cylinder_segment_mesh(
    dimension, polarization, meshing, position=(0, 0, 0), orientation=None
):
    """Return the cells of a cylinder-segment magnet for the force computation.

    The segment is cut into cylinder-segment cells in rings and layers. With
    ``position=(0, 0, 0)`` and ``orientation=None`` the cylinder axis
    coincides with the z-axis and the segment lies between z = -h/2 and h/2.

    Parameters
    ----------
    dimension : array-like, shape (5,)
        Segment dimension (r1, r2, h, φ1, φ2) where r1 < r2 are inner and
        outer radii in units (m), h is the height in units (m), and φ1 < φ2
        are azimuth section angles in units (deg).
    polarization : array-like, shape (3,)
        Magnetic polarization in units (T).
    meshing : int | tuple of int, shape (3,)
        Target number of cells, or the divisions (nphi, nr, nh) along the arc
        at the mean radius, the radius and the height. An integer gives cells
        that are as close to cubes as possible.
    position : array-like, shape (3,), default (0, 0, 0)
        Position of the cylinder axis at half the height in units (m).
    orientation : None | Rotation, default None
        Segment orientation. If ``None``, the unit rotation is applied.

    Returns
    -------
    tuple[array, array]
        Cell positions in units (m), which are the cell centroids, and
        magnetic moments of the cells in units (A*m²), each of shape (n, 3),
        as inputs of ``force_torque``.

    Notes
    -----
    With JAX or PyTorch arrays the mesh is computed in that library and can be
    differentiated with respect to the inputs. The number of cells in each
    ring is taken from the values of ``dimension``, which have to be known:
    the mesh cannot be created under ``jax.jit``.

    Examples
    --------
    >>> import magpylib as magpy
    >>> points, moments = magpy.func.cylinder_segment_mesh(
    ...     dimension=(1.0, 2.0, 1.0, 0.0, 90.0),
    ...     polarization=(0.0, 0.0, 1.0),
    ...     meshing=(2, 1, 1),
    ... )
    >>> print(points.round(3))
    [[1.4  0.58 0.  ]
     [0.58 1.4  0.  ]]
    """
    meshing = check_meshing_count_or_triple(meshing, "cylinder_segment_mesh")
    _, (dimension, polarization, position) = _to_namespace(
        dimension, polarization, position
    )
    _check_shapes(
        "cylinder_segment_mesh",
        ((5,), (3,), (3,)),
        dimension=dimension,
        polarization=polarization,
        position=position,
    )
    cells = _cylinder_segment_cells(dimension, meshing)
    return _magnet_mesh(cells, polarization, position, orientation)


def tetrahedron_mesh(
    vertices, polarization, meshing, position=(0, 0, 0), orientation=None
):
    """Return the cells of a tetrahedron magnet for the force computation.

    The tetrahedron is cut into tetrahedron cells of equal volume. With
    ``position=(0, 0, 0)`` and ``orientation=None`` local and global
    coordinates coincide.

    Parameters
    ----------
    vertices : array-like, shape (4, 3)
        Tetrahedron vertices in units (m).
    polarization : array-like, shape (3,)
        Magnetic polarization in units (T).
    meshing : int
        Target number of cells. The tetrahedron is cut into the cube number
        of cells that is closest to it.
    position : array-like, shape (3,), default (0, 0, 0)
        Local coordinate origin in units (m).
    orientation : None | Rotation, default None
        Local coordinate orientation. If ``None``, the unit rotation is
        applied.

    Returns
    -------
    tuple[array, array]
        Cell positions in units (m), which are the cell centroids, and
        magnetic moments of the cells in units (A*m²), each of shape (n, 3),
        as inputs of ``force_torque``.

    Notes
    -----
    With JAX or PyTorch arrays the mesh is computed in that library and can be
    differentiated with respect to the inputs, also under ``jax.jit``.

    Examples
    --------
    >>> import magpylib as magpy
    >>> points, moments = magpy.func.tetrahedron_mesh(
    ...     vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
    ...     polarization=(0.0, 0.0, 1.0),
    ...     meshing=8,
    ... )
    >>> print(points.shape)
    (8, 3)
    >>> print(moments.sum(axis=0).round())
    [     0.      0. 132629.]
    """
    meshing = check_meshing_count(meshing, "tetrahedron_mesh")
    _, (vertices, polarization, position) = _to_namespace(
        vertices, polarization, position
    )
    _check_shapes(
        "tetrahedron_mesh",
        ((4, 3), (3,), (3,)),
        vertices=vertices,
        polarization=polarization,
        position=position,
    )
    cells = _tetrahedron_cells(vertices, meshing)
    return _magnet_mesh(cells, polarization, position, orientation)


def circle_mesh(diameter, current, meshing, position=(0, 0, 0), orientation=None):
    """Return the segments of a circular current loop for the force computation.

    The loop is cut into straight segments, the sides of the regular polygon
    with the area of the circle. With ``position=(0, 0, 0)`` and
    ``orientation=None`` the loop lies in the ``z=0`` plane with its center
    at the origin.

    Parameters
    ----------
    diameter : float | array-like, shape ()
        Loop diameter in units (m).
    current : float | array-like, shape ()
        Electric current in units (A).
    meshing : int
        Number of segments, at least 4.
    position : array-like, shape (3,), default (0, 0, 0)
        Loop center in units (m).
    orientation : None | Rotation, default None
        Loop orientation. If ``None``, the unit rotation is applied.

    Returns
    -------
    tuple[array, array]
        Segment midpoints in units (m) and current vectors, the current times
        the segment vectors, in units (A*m), each of shape (n, 3), as inputs
        of ``force_torque``.

    Notes
    -----
    With JAX or PyTorch arrays the mesh is computed in that library and can be
    differentiated with respect to the inputs, also under ``jax.jit``.

    Examples
    --------
    >>> import magpylib as magpy
    >>> points, current_vectors = magpy.func.circle_mesh(
    ...     diameter=2.0,
    ...     current=1.0,
    ...     meshing=4,
    ... )
    >>> print(points.round(3) + 0.0)
    [[ 0.627  0.627  0.   ]
     [-0.627  0.627  0.   ]
     [-0.627 -0.627  0.   ]
     [ 0.627 -0.627  0.   ]]
    """
    meshing = check_meshing_count(meshing, "circle_mesh", minimum=4)
    _, (diameter, current, position) = _to_namespace(diameter, current, position)
    _check_shapes(
        "circle_mesh",
        ((), (), (3,)),
        diameter=diameter,
        current=current,
        position=position,
    )
    points, cvecs = _circle_segments(diameter / 2, current, meshing)
    return _place(points, cvecs, position, orientation)

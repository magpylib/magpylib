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

from magpylib._src.array_api_utils import input_namespace, xp_promote


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
            for unit in xp.unstack(
                xp.eye(3, dtype=points.dtype, device=device(points))
            )
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

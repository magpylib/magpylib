"""Helpers for field code written against the Python array API standard."""

import numpy as np
from array_api_compat import (
    array_namespace,
    device,
    is_array_api_obj,
    is_jax_namespace,
    is_numpy_namespace,
    is_torch_namespace,
)


def input_namespace(*args):
    """Return the namespace of the array inputs among lists, scalars and arrays.

    NumPy arrays, lists and scalars follow the arrays of another library, so
    that NumPy is the namespace only when no other one is given.
    """
    arrays = [
        arg
        for arg in args
        if is_array_api_obj(arg) and not isinstance(arg, np.ndarray | np.generic)
    ]
    return array_namespace(*arrays) if arrays else array_namespace(np.empty(0))


def xp_promote(*args, xp):
    """Promote arrays to their common floating dtype.

    Integer arrays become the default floating dtype of the namespace, so
    that a kernel can divide and take roots of its inputs.
    """
    floats = [arg.dtype for arg in args if xp.isdtype(arg.dtype, "real floating")]
    dtype = xp.result_type(*floats) if floats else xp.asarray(1.0).dtype
    out = tuple(xp.astype(arg, dtype, copy=False) for arg in args)
    return out[0] if len(out) == 1 else out


def constant(values, like):
    """Return NumPy values as an array of the namespace and device of ``like``,
    floating values in its dtype."""
    values = np.asarray(values)
    dtype = like.dtype if values.dtype.kind == "f" else None
    xp = array_namespace(like)
    return xp.asarray(values, dtype=dtype, device=device(like))


def concrete(value, name):
    """Return the values of an input array as a NumPy array, detached from
    automatic differentiation: for decisions that cannot be traced, such as
    a number of cells.

    The input has no values while ``jax.jit`` or a ``vmap`` traces it, which
    is an error."""
    xp = array_namespace(value)
    try:
        if is_jax_namespace(xp):
            import jax  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

            value = jax.lax.stop_gradient(value)
        elif is_torch_namespace(xp):
            value = value.detach().tolist()
        return np.asarray(value, dtype=float)
    except (TypeError, RuntimeError) as err:
        msg = (
            f"Input {name} has no values while it is traced, as under jax.jit or "
            "a vmap, but the cells of the mesh are laid out from them. Create "
            "the mesh outside of the traced function, or use a mesh whose "
            "layout is fixed by the meshing input, such as a cuboid with the "
            "divisions (n1, n2, n3)."
        )
        raise ValueError(msg) from err


def require_numpy(xp, name):
    """Raise for a kernel that is only implemented for NumPy input."""
    if not is_numpy_namespace(xp):
        msg = (
            f"{name} is only implemented for NumPy arrays: it needs incomplete "
            "elliptic integrals from SciPy, which other array libraries lack; "
            f"instead received arrays of {xp.__name__}."
        )
        raise NotImplementedError(msg)


def while_loop(cond, body, state, max_iter):
    """Repeat ``state = body(state)`` for as long as ``cond(state)`` is true,
    at most ``max_iter`` times.

    ``state`` is a tuple of arrays. When JAX traces the loop, the condition
    has no value yet. The loop then always runs ``max_iter`` times and keeps
    the state once the condition is false, because JAX cannot differentiate
    a loop of unknown length in reverse mode.
    """
    xp = array_namespace(*state)
    if is_jax_namespace(xp):
        import jax  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

        def step(_, state):
            go = cond(state)
            return tuple(
                xp.where(go, new, old)
                for new, old in zip(body(state), state, strict=True)
            )

        try:
            bool(cond(state))
        except jax.errors.TracerBoolConversionError:
            return jax.lax.fori_loop(0, max_iter, step, state)
    for _ in range(max_iter):
        if not bool(cond(state)):
            break
        state = body(state)
    return state


def rotate(quat, vectors, *, inverse=False):
    """Rotate vectors, shape (i, 3), by unit quaternions, shape (i, 4).

    The quaternions are scalar-last, as given by SciPy's
    ``Rotation.as_quat()``. SciPy's ``Rotation`` itself only handles other
    array libraries when ``SCIPY_ARRAY_API=1`` is set before it is imported.
    """
    xp = array_namespace(quat, vectors)
    x, y, z, w = xp.unstack(quat, axis=-1)
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w

    # rotation matrices as in SciPy's Rotation.as_matrix, shape (i, 3, 3)
    matrix = xp.stack(
        (
            xp.stack((x2 - y2 - z2 + w2, 2 * (xy - zw), 2 * (xz + yw)), axis=-1),
            xp.stack((2 * (xy + zw), -x2 + y2 - z2 + w2, 2 * (yz - xw)), axis=-1),
            xp.stack((2 * (xz - yw), 2 * (yz + xw), -x2 - y2 + z2 + w2), axis=-1),
        ),
        axis=-2,
    )
    if inverse:
        matrix = xp.matrix_transpose(matrix)
    # an infinite field component times a zero of the matrix is NaN, as with
    # SciPy's Rotation, which does not warn about it
    with np.errstate(invalid="ignore"):
        return xp.matmul(matrix, vectors[..., xp.newaxis])[..., 0]

"""Helpers for field code written against the Python array API standard."""

import functools
import inspect
import sys

import array_api_compat
import numpy as np
from array_api_compat import (
    is_array_api_obj,
    is_jax_namespace,
    is_numpy_namespace,
    is_torch_namespace,
)

_NUMPY = array_api_compat.array_namespace(np.empty(0))
_FLOAT64 = np.dtype(float)


def array_namespace(*arrays):
    """Return the array namespace of the arrays, as ``array_api_compat``
    does, which takes 2 microseconds per call. NumPy arrays are recognized
    without it: a kernel asks for the namespace in each of its functions,
    which adds up to much of the time of a call with few observers.
    """
    for array in arrays:
        # not isinstance: subclasses like masked arrays are left to the lookup
        if type(array) is not np.ndarray:  # pylint: disable=unidiomatic-typecheck
            return array_api_compat.array_namespace(*arrays)
    return _NUMPY


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
    if xp is _NUMPY and all(arg.dtype == _FLOAT64 for arg in args):
        return args[0] if len(args) == 1 else args  # nothing to promote
    floats = [arg.dtype for arg in args if xp.isdtype(arg.dtype, "real floating")]
    dtype = xp.result_type(*floats) if floats else xp.asarray(1.0).dtype
    out = tuple(xp.astype(arg, dtype, copy=False) for arg in args)
    return out[0] if len(out) == 1 else out


def some(mask):
    """Whether any element of a mask is true, to skip a branch of a kernel
    that no instance needs.

    Also true when that cannot be known, because JAX or PyTorch is tracing:
    the branch is then computed for all instances and selected with
    ``where``, as tracing requires.
    """
    if type(mask) is np.ndarray:  # pylint: disable=unidiomatic-typecheck
        return bool(mask.any())
    xp = array_namespace(mask)
    # torch.jit.trace would record the branch that its example input takes
    if is_torch_namespace(xp) and sys.modules["torch"].jit.is_tracing():
        return True
    try:
        return bool(xp.any(mask))
    except (TypeError, RuntimeError):  # of JAX and of torch.func
        return True


def dot(vectors, others):
    """Inner product of vectors of three components along the last axis.

    For NumPy it is written out, which is about three times faster there
    than a sum over a short axis. Other libraries are faster with the sum.
    """
    xp = array_namespace(vectors, others)
    if xp is not _NUMPY:
        return xp.sum(vectors * others, axis=-1)
    return (
        vectors[..., 0] * others[..., 0]
        + vectors[..., 1] * others[..., 1]
        + vectors[..., 2] * others[..., 2]
    )


def cross(vectors, others):
    """Cross product of vectors of three components along the last axis.

    For NumPy it is written out, which is twice as fast for few vectors as
    its own cross product.
    """
    xp = array_namespace(vectors, others)
    if xp is not _NUMPY:
        return xp.linalg.cross(vectors, others)
    a0, a1, a2 = vectors[..., 0], vectors[..., 1], vectors[..., 2]
    b0, b1, b2 = others[..., 0], others[..., 1], others[..., 2]
    return xp.stack((a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0), axis=-1)


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


def padded_size(n):
    """Round a number of rows up to one of four sizes per power of two.

    That is at most a quarter more rows: 5, 6, 7, 8, 10, 12, 14, 16, 20, ...
    """
    shift = max(n.bit_length() - 3, 0)
    return -(-n >> shift) << shift


def pad_rows(arrays, n, xp):
    """Give every array of a dict ``padded_size(n)`` rows, for JAX.

    The arrays have ``n`` rows or one. JAX compiles each operation once for
    every array shape, also without ``jax.jit``, so that every new number of
    rows costs a compilation of the whole computation. With the rows padded,
    by repeating the last one, only the padded sizes are compiled. This is
    done with NumPy, because JAX would compile the padding for every shape as
    well.

    Arrays that JAX is tracing (``jit``, ``grad``, ``vmap``) are not padded,
    only tiled to ``n`` rows: there the user is in charge of the shapes.
    """
    import jax  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

    if any(isinstance(val, jax.core.Tracer) for val in arrays.values()):
        return {
            key: val
            if val.shape[0] == n
            else xp.tile(val, (n,) + (1,) * (val.ndim - 1))
            for key, val in arrays.items()
        }
    size = padded_size(n)
    out = {}
    for key, val in arrays.items():
        out[key] = val
        if val.shape[0] != size:
            # ponytail: arrays on an accelerator are copied to the host and
            # back, pad with jax.numpy.pad there if that copy costs more than
            # compiling the padding once per size
            rows = np.asarray(val)
            last = np.broadcast_to(rows[-1:], (size - rows.shape[0], *rows.shape[1:]))
            out[key] = xp.asarray(np.concatenate((rows, last)))
    return out


def cut_rows(array, n, xp, axis=0):
    """Cut the rows that ``pad_rows`` added from a result, with NumPy."""
    if array.shape[axis] == n:
        return array
    array = np.asarray(array)
    return xp.asarray(array[:n] if axis == 0 else array[:, :n])


def jax_padded(out_axis=0):  # 0 or 1
    """Pad the inputs of a kernel when it is called with JAX arrays.

    All inputs of the kernel are arrays with the instances along the first
    axis, and its output has them along ``out_axis``, see ``pad_rows``.
    """

    def decorator(func):
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if array_namespace(*args, *kwargs.values()) is _NUMPY:
                return func(*args, **kwargs)
            arrays = signature.bind(*args, **kwargs).arguments
            xp = input_namespace(*arrays.values())
            sizes = {
                val.shape[0] if getattr(val, "ndim", 0) else None
                for val in arrays.values()
            }
            n = sizes.pop()
            if not is_jax_namespace(xp) or sizes or not n:
                return func(*args, **kwargs)
            return cut_rows(func(**pad_rows(arrays, n, xp)), n, xp, out_axis)

        return wrapper

    return decorator


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

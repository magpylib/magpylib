"""Helpers for field code written against the Python array API standard."""

import numpy as np
from array_api_compat import array_namespace, is_array_api_obj, is_numpy_namespace


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


def require_numpy(xp, name):
    """Raise for a kernel that is only implemented for NumPy input."""
    if not is_numpy_namespace(xp):
        msg = (
            f"{name} is only implemented for NumPy arrays: it needs incomplete "
            "elliptic integrals from SciPy, which other array libraries lack; "
            f"instead received arrays of {xp.__name__}."
        )
        raise NotImplementedError(msg)


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

import numpy as np
import numpy.typing as npt
from array_api_compat import (
    is_numpy_namespace as is_numpy,
    is_torch_namespace as is_torch,
    array_namespace,
)
from typing import Any, TypeAlias, Literal
from types import ModuleType

Array: TypeAlias = Any  # To be changed to a Protocol later (see array-api#589)
ArrayLike: TypeAlias = Array | npt.ArrayLike


def _check_finite(array: Array, xp: ModuleType) -> None:
    """Check for NaNs or Infs."""
    if not xp.all(xp.isfinite(array)):
        msg = "array must not contain infs or NaNs"
        raise ValueError(msg)


def _asarray(
    array: ArrayLike,
    dtype: Any = None,
    order: Literal["K", "A", "C", "F"] | None = None,
    copy: bool | None = None,
    *,
    xp: ModuleType | None = None,
    check_finite: bool = False,
    subok: bool = False,
) -> Array:
    """SciPy-specific replacement for `np.asarray` with `order`, `check_finite`, and
    `subok`.

    Memory layout parameter `order` is not exposed in the Array API standard.
    `order` is only enforced if the input array implementation
    is NumPy based, otherwise `order` is just silently ignored.

    `check_finite` is also not a keyword in the array API standard; included
    here for convenience rather than that having to be a separate function
    call inside SciPy functions.

    `subok` is included to allow this function to preserve the behaviour of
    `np.asanyarray` for NumPy based inputs.
    """
    if xp is None:
        xp = array_namespace(array)
    if is_numpy(xp):
        # Use NumPy API to support order
        if copy is True:
            array = np.array(array, order=order, dtype=dtype, subok=subok)
        elif subok:
            array = np.asanyarray(array, order=order, dtype=dtype)
        else:
            array = np.asarray(array, order=order, dtype=dtype)
    else:
        try:
            array = xp.asarray(array, dtype=dtype, copy=copy)
        except TypeError:
            coerced_xp = array_namespace(xp.asarray(3))
            array = coerced_xp.asarray(array, dtype=dtype, copy=copy)

    if check_finite:
        _check_finite(array, xp)

    return array


def xp_default_dtype(xp):
    """Query the namespace-dependent default floating-point dtype."""
    if is_torch(xp):
        # historically, we allow pytorch to keep its default of float32
        return xp.get_default_dtype()
    else:
        # we default to float64
        return xp.float64


def xp_result_type(*args, force_floating=False, xp):
    """
    Returns the dtype that results from applying type promotion rules
    (see Array API Standard Type Promotion Rules) to the arguments. Augments
    standard `result_type` in a few ways:

    - There is a `force_floating` argument that ensures that the result type
      is floating point, even when all args are integer.
    - When a TypeError is raised (e.g. due to an unsupported promotion)
      and `force_floating=True`, we define a custom rule: use the result type
      of the default float and any other floats passed. See
      https://github.com/scipy/scipy/pull/22695/files#r1997905891
      for rationale.
    - This function accepts array-like iterables, which are immediately converted
      to the namespace's arrays before result type calculation. Consequently, the
      result dtype may be different when an argument is `1.` vs `[1.]`.

    Typically, this function will be called shortly after `array_namespace`
    on a subset of the arguments passed to `array_namespace`.
    """
    args = [
        (_asarray(arg, subok=True, xp=xp) if np.iterable(arg) else arg) for arg in args
    ]
    args_not_none = [arg for arg in args if arg is not None]
    if force_floating:
        args_not_none.append(1.0)

    if is_numpy(xp) and xp.__version__ < "2.0":
        # Follow NEP 50 promotion rules anyway
        args_not_none = [
            arg.dtype if getattr(arg, "size", 0) == 1 else arg for arg in args_not_none
        ]
        return xp.result_type(*args_not_none)

    try:  # follow library's preferred promotion rules
        return xp.result_type(*args_not_none)
    except TypeError:  # mixed type promotion isn't defined
        if not force_floating:
            raise
        # use `result_type` of default floating point type and any floats present
        # This can be revisited, but right now, the only backends that get here
        # are array-api-strict (which is not for production use) and PyTorch
        # (due to data-apis/array-api-compat#279).
        float_args = []
        for arg in args_not_none:
            arg_array = xp.asarray(arg) if np.isscalar(arg) else arg
            dtype = getattr(arg_array, "dtype", arg)
            if xp.isdtype(dtype, ("real floating", "complex floating")):
                float_args.append(arg)
        return xp.result_type(*float_args, xp_default_dtype(xp))


def xp_promote(*args, broadcast=False, force_floating=False, xp):
    """
    Promotes elements of *args to result dtype, ignoring `None`s.
    Includes options for forcing promotion to floating point and
    broadcasting the arrays, again ignoring `None`s.
    Type promotion rules follow `xp_result_type` instead of `xp.result_type`.

    Typically, this function will be called shortly after `array_namespace`
    on a subset of the arguments passed to `array_namespace`.

    This function accepts array-like iterables, which are immediately converted
    to the namespace's arrays before result type calculation. Consequently, the
    result dtype may be different when an argument is `1.` vs `[1.]`.

    See Also
    --------
    xp_result_type
    """
    args = [
        (_asarray(arg, subok=True, xp=xp) if np.iterable(arg) else arg) for arg in args
    ]  # solely to prevent double conversion of iterable to array

    dtype = xp_result_type(*args, force_floating=force_floating, xp=xp)

    args = [
        (_asarray(arg, dtype=dtype, subok=True, xp=xp) if arg is not None else arg)
        for arg in args
    ]

    if not broadcast:
        return args[0] if len(args) == 1 else tuple(args)

    args_not_none = [arg for arg in args if arg is not None]

    # determine result shape
    shapes = {arg.shape for arg in args_not_none}
    try:
        shape = (
            np.broadcast_shapes(*shapes) if len(shapes) != 1 else args_not_none[0].shape
        )
    except ValueError as e:
        message = "Array shapes are incompatible for broadcasting."
        raise ValueError(message) from e

    out = []
    for arg in args:
        if arg is None:
            out.append(arg)
            continue

        # broadcast only if needed
        # Even if two arguments need broadcasting, this is faster than
        # `broadcast_arrays`, especially since we've already determined `shape`
        if arg.shape != shape:
            kwargs = {"subok": True} if is_numpy(xp) else {}
            arg = xp.broadcast_to(arg, shape, **kwargs)

        # This is much faster than xp.astype(arg, dtype, copy=False)
        if arg.dtype != dtype:
            arg = xp.astype(arg, dtype)

        out.append(arg)

    return out[0] if len(out) == 1 else tuple(out)

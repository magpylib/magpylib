from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import partial, wraps
from types import ModuleType
from typing import Any, Literal, ParamSpec, TypeAlias, cast

import array_api_compat as _compat
import numpy as np
import numpy.typing as npt
from array_api_compat import (
    array_namespace,
    is_dask_namespace,
    is_jax_namespace,
    is_numpy_array,
)
from array_api_compat import (
    is_numpy_namespace as is_numpy,
)
from array_api_compat import (
    is_torch_namespace as is_torch,
)
from typing_extensions import TypeIs

Array: TypeAlias = Any  # To be changed to a Protocol later (see array-api#589)
DType: TypeAlias = Any  # To be changed to a Protocol later (see array-api#589)
ArrayLike: TypeAlias = Array | npt.ArrayLike

P = ParamSpec("P")


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


def _is_jax_jit_enabled(xp: ModuleType) -> bool:  # numpydoc ignore=PR01,RT01
    """Return True if this function is being called inside ``jax.jit``."""
    import jax  # pylint: disable=import-outside-toplevel

    x = xp.asarray(False)
    try:
        return bool(x)
    except jax.errors.TracerBoolConversionError:
        return True


def is_python_scalar(x: object) -> TypeIs[complex]:  # numpydoc ignore=PR01,RT01
    """Return True if `x` is a Python scalar, False otherwise."""
    # isinstance(x, float) returns True for np.float64
    # isinstance(x, complex) returns True for np.complex128
    # bool is a subclass of int
    return isinstance(x, int | float | complex) and not is_numpy_array(x)


def lazy_while(  # type: ignore[valid-type]  # numpydoc ignore=GL07,SA04
    func: Callable[P, Array | ArrayLike | Sequence[Array | ArrayLike]],
    cond: Callable[P, bool],
    args: Array | complex | None,
    shape: tuple[int | None, ...] | Sequence[tuple[int | None, ...]] | None = None,
    dtype: DType | Sequence[DType] | None = None,
    as_numpy: bool = False,
    xp: ModuleType | None = None,
    **kwargs: P.kwargs,  # pyright: ignore[reportGeneralTypeIssues]
) -> Array | tuple[Array, ...]:
    args_not_none = [arg for arg in args if arg is not None]
    array_args = [arg for arg in args_not_none if not is_python_scalar(arg)]
    if not array_args:
        msg = "Must have at least one argument array"
        raise ValueError(msg)
    if xp is None:
        xp = array_namespace(*args)

    # Normalize and validate shape and dtype
    shapes: list[tuple[int | None, ...]]
    dtypes: list[DType]
    multi_output = False

    if shape is None:
        shapes = [np.broadcast_shapes(*(arg.shape for arg in array_args))]
    elif all(isinstance(s, int | None) for s in shape):
        # Do not test for shape to be a tuple
        # https://github.com/data-apis/array-api/issues/891#issuecomment-2637430522
        shapes = [cast(tuple[int | None, ...], shape)]
    else:
        shapes = list(shape)  # type: ignore[arg-type]  # pyright: ignore[reportAssignmentType]
        multi_output = True

    if dtype is None:
        dtypes = [xp.result_type(*args_not_none)] * len(shapes)
    elif multi_output:
        if not isinstance(dtype, Sequence):
            msg = "Got multiple shapes but only one dtype"
            raise ValueError(msg)
        dtypes = list(dtype)  # pyright: ignore[reportUnknownArgumentType]
    else:
        if isinstance(dtype, Sequence):
            msg = "Got single shape but multiple dtypes"
            raise ValueError(msg)

        dtypes = [dtype]

    if len(shapes) != len(dtypes):
        msg = f"Got {len(shapes)} shapes and {len(dtypes)} dtypes"
        raise ValueError(msg)
    del shape
    del dtype
    # End of shape and dtype parsing

    # Backend-specific branches
    if is_dask_namespace(xp):
        import dask
        import dask.array

        metas: list[Array] = [arg._meta for arg in array_args]  # pylint: disable=protected-access    # pyright: ignore[reportAttributeAccessIssue]
        meta_xp = array_namespace(*metas)

        def dask_wrapper(args):
            while cond(args):
                args = func(args)
            return args

        wrapped = dask.delayed(  # type: ignore[attr-defined]  # pyright: ignore[reportPrivateImportUsage]
            _lazy_apply_wrapper(dask_wrapper, as_numpy, multi_output, meta_xp),
            pure=True,
        )
        # This finalizes each arg, which is the same as arg.rechunk(-1).
        # Please read docstring above for why we're not using
        # dask.array.map_blocks or dask.array.blockwise!
        delayed_out = dask.array.map_blocks(wrapped, args, **kwargs)

        out = tuple(
            xp.from_delayed(
                delayed_out[i],  # pyright: ignore[reportIndexIssue]
                # Dask's unknown shapes diverge from the Array API specification
                shape=tuple(math.nan if s is None else s for s in shape),
                dtype=dtype,
                meta=metas[0],
            )
            for i, (shape, dtype) in enumerate(zip(shapes, dtypes, strict=True))
        )

    elif is_jax_namespace(xp) and _is_jax_jit_enabled(xp):
        # Delay calling func with jax.pure_callback, which will forward to func eager
        # JAX arrays. Do not use jax.pure_callback when running outside of the JIT,
        # as it does not support raising exceptions:
        # https://github.com/jax-ml/jax/issues/26102
        import jax

        if any(None in shape for shape in shapes):
            msg = "Output shape must be fully known when running inside jax.jit"
            raise ValueError(msg)

        # Shield kwargs from being coerced into JAX arrays.
        # jax.pure_callback calls jax.jit under the hood, but without the chance of
        # passing static_argnames / static_argnums.
        wrapped = _lazy_apply_wrapper(
            partial(func, **kwargs), as_numpy, multi_output, xp
        )

        # suppress unused-ignore to run mypy in -e lint as well as -e dev
        out = cast(  # type: ignore[bad-cast,unused-ignore]
            tuple[Array, ...],
            jax.lax.while_loop(
                cond,
                wrapped,
                init_val=args,
            ),
        )

    else:
        # Eager backends, including non-jitted JAX
        # wrapped = _lazy_apply_wrapper(func, as_numpy, multi_output, xp)
        while cond(args):
            args = func(args)
        out = args

    return out


def traverse_args(args):
    if not isinstance(args, Sequence):
        yield args
    else:
        for a in args:
            yield from traverse_args(a)


def _lazy_apply_wrapper(  # type: ignore[explicit-any]  # numpydoc ignore=PR01,RT01
    func: Callable[..., Array | ArrayLike | Sequence[Array | ArrayLike]],
    as_numpy: bool,
    multi_output: bool,
    xp: ModuleType,
) -> Callable[..., tuple[Array, ...]]:
    """
    Helper of `lazy_apply`.

    Given a function that accepts one or more arrays as positional arguments and returns
    a single array-like or a sequence of array-likes, return a function that accepts the
    same number of Array API arrays and always returns a tuple of Array API array.

    Any keyword arguments are passed through verbatim to the wrapped function.
    """

    # On Dask, @wraps causes the graph key to contain the wrapped function's name
    @wraps(func)
    def wrapper(  # type: ignore[decorated-any,explicit-any]
        *args: Array | complex | None, **kwargs: Any
    ) -> tuple[Array, ...]:  # numpydoc ignore=GL08
        args_list = []
        device = None
        for a in args:
            for arg in a:
                if arg is not None and not is_python_scalar(arg):
                    if device is None:
                        device = _compat.device(arg)
                    if as_numpy:
                        import numpy as np

                        arg = cast(Array, np.asarray(arg))  # type: ignore[bad-cast]  # noqa: PLW2901
                args_list.append(arg)
        # assert device is not None

        out = func(tuple(args_list), **kwargs)

        return tuple(xp.asarray(o, device=device) for o in out)

    return wrapper
    return wrapper

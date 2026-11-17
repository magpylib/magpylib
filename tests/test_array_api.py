"""The core functions and the functional interface on JAX and PyTorch arrays.

The values are checked by test_regression_reference.py, which runs its core
and functional interface cases on these libraries too.
"""

from pathlib import Path

import numpy as np
import pytest

import magpylib as magpy

DATA = Path(__file__).parent / "testdata" / ("regression_reference" + ".npz")


@pytest.fixture(scope="module")
def reference():
    with np.load(DATA) as data:
        return dict(data)


@pytest.fixture(scope="module", params=["jax", "torch"])
def xp(request):
    """The array namespace of the library, computing in float64."""
    if request.param == "jax":
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        return jax.numpy
    return pytest.importorskip("torch")


def _is_array(xp, val):
    return isinstance(val, xp.Tensor if xp.__name__ == "torch" else xp.ndarray)


@pytest.mark.parametrize(
    "name", sorted(set(magpy.func.__all__) - {"cylinder_segment_field"})
)
def test_backend_output(name, xp, reference):
    """The functional interface returns an array of the input library, also
    when only one input is such an array and the others are lists."""
    prefix = f"func/{name}/B/broadcast|in|"
    inputs = {
        key.removeprefix(prefix): val.tolist()
        for key, val in reference.items()
        if key.startswith(prefix)
    }
    del inputs["orientations"]
    inputs["observers"] = xp.asarray(np.array(inputs["observers"]))
    out = getattr(magpy.func, name)("B", **inputs)
    assert _is_array(xp, out)
    assert str(out.dtype).endswith("float64")


@pytest.mark.parametrize(
    "name", sorted(set(magpy.core.__all__) - {"magnet_cylinder_segment_Hfield"})
)
def test_backend_output_core(name, xp, reference):
    """A core function returns an array of the input library."""
    prefix = f"core/{name}|in|"
    inputs = {
        key.removeprefix(prefix): xp.asarray(val)
        for key, val in reference.items()
        if key.startswith(prefix)
    }
    out = getattr(magpy.core, name)(**inputs)
    assert _is_array(xp, out)
    assert str(out.dtype).endswith("float64")


def test_backend_float32_warns(xp):
    """Other precision than float64 follows the input arrays, with a warning."""
    observers = xp.asarray([[1.0, 2.0, 3.0]], dtype=xp.float32)
    with pytest.warns(UserWarning, match="float32 instead of float64"):
        out = magpy.func.dipole_field("B", observers, (1.0, 0.0, 0.0), squeeze=False)
    assert str(out.dtype).endswith("float32")
    expected = magpy.func.dipole_field("B", np.asarray(observers), (1.0, 0.0, 0.0))
    np.testing.assert_allclose(np.asarray(out)[0], expected, rtol=1e-5)


def test_backend_gradient(xp):
    """The field can be differentiated with respect to the observer."""
    position = [0.3, 0.4, 0.5]

    def field_z(observer):
        return magpy.func.sphere_field("B", observer, 1.0, (0.0, 0.0, 1.0))[2]

    if xp.__name__ == "torch":
        observer = xp.asarray(position, dtype=xp.float64).requires_grad_()
        (grad,) = xp.autograd.grad(field_z(observer), observer)
    else:
        import jax  # noqa: PLC0415

        grad = jax.grad(field_z)(xp.asarray(position))
    step = 1e-6 * np.eye(3)
    expected = [
        (field_z(np.array(position) + d) - field_z(np.array(position) - d)) / 2e-6
        for d in step
    ]
    np.testing.assert_allclose(np.asarray(grad), expected, rtol=1e-6)

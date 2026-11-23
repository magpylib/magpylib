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


JAX_INPUTS = {
    "circle_field": {"diameters": [2.0, 1.5], "currents": [1.0, 3.0]},
    "cuboid_field": {
        "dimensions": [(1.0, 2.0, 3.0), (2.0, 1.0, 1.0)],
        "polarizations": [(0.1, 0.2, 1.0), (1.0, 0.0, 0.3)],
    },
    "cylinder_field": {
        "dimensions": [(1.0, 2.0), (2.0, 1.0)],
        "polarizations": [(0.1, 0.2, 1.0), (1.0, 0.0, 0.3)],
    },
    "dipole_field": {"moments": [(0.1, 0.2, 1.0), (1.0, 0.0, 0.3)]},
    "polyline_field": {
        "segments_start": [(0.0, 0.0, 0.0), (0.1, 0.2, 0.3)],
        "segments_end": [(1.0, 0.0, 0.0), (0.5, -0.4, 0.2)],
        "currents": [1.0, 3.0],
    },
    "sphere_field": {
        "diameters": [1.0, 2.0],
        "polarizations": [(0.1, 0.2, 1.0), (1.0, 0.0, 0.3)],
    },
    "tetrahedron_field": {
        "vertices": [[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]] * 2,
        "polarizations": [(0.1, 0.2, 1.0), (1.0, 0.0, 0.3)],
    },
    "triangle_charge_field": {
        "vertices": [[(0, 0, 0), (1, 0, 0), (0, 1, 0)]] * 2,
        "polarizations": [(0.1, 0.2, 1.0), (1.0, 0.0, 0.3)],
    },
    "triangle_current_field": {
        "vertices": [[(0, 0, 0), (1, 0, 0), (0, 1, 0)]] * 2,
        "current_densities": [(1.0, 2.0, 0.0), (-2.0, 1.0, 0.0)],
    },
}
OBSERVERS = np.array([(0.3, 2.1, 0.4), (-1.7, 0.6, 1.2)])


def test_jax_inputs_cover_func():
    assert set(JAX_INPUTS) == set(magpy.func.__all__) - {"cylinder_segment_field"}


@pytest.mark.parametrize("name", JAX_INPUTS)
def test_jax_transformations(name):
    """The functional interface can be compiled and differentiated by the
    user: under jax.jit it gives the NumPy field, and the compiled gradient
    with respect to the observers is the finite difference of the field.
    Not on the axis of a circle or cylinder, where the gradient is NaN."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    func = getattr(magpy.func, name)
    inputs = {key: np.array(val, dtype=float) for key, val in JAX_INPUTS[name].items()}
    position = np.array((0.1, -0.2, 0.3))

    def field(observers, inputs):
        return func("B", observers, **inputs, positions=position)

    def norm(observers, inputs):
        return (field(observers, inputs) ** 2).sum()

    expected = field(OBSERVERS, inputs)
    arrays = {key: jax.numpy.asarray(val) for key, val in inputs.items()}
    observers = jax.numpy.asarray(OBSERVERS)
    np.testing.assert_allclose(
        jax.jit(field)(observers, arrays), expected, rtol=1e-10, atol=1e-20
    )

    # the observers are independent, so one step moves all of them at once
    slope = np.empty((2, 3))
    for axis, step in enumerate(1e-6 * np.eye(3)):
        ahead, behind = field(OBSERVERS + step, inputs), field(OBSERVERS - step, inputs)
        slope[:, axis] = ((ahead**2).sum(axis=1) - (behind**2).sum(axis=1)) / 2e-6
    for grad in (jax.grad(norm), jax.jit(jax.grad(norm))):
        np.testing.assert_allclose(
            grad(observers, arrays), slope, rtol=1e-5, atol=1e-9 * np.abs(slope).max()
        )


def test_jax_jit_above_chunk_rows(monkeypatch):
    """Inputs of more rows than a chunk are traced in one piece."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    monkeypatch.setattr("magpylib._src.fields.chunks.CHUNK_ROWS", 4)
    observers = np.random.default_rng(0).uniform(1, 2, (11, 3))

    def field(observers):
        return magpy.func.dipole_field("B", observers, (0.1, 0.2, 1.0))

    out = jax.jit(field)(jax.numpy.asarray(observers))
    np.testing.assert_allclose(out, field(observers), rtol=1e-12)


def test_backend_float32_elliptic(xp):
    """The iteration of the elliptic integrals converges in float32."""
    rng = np.random.default_rng(0)
    z0, r, z = rng.uniform(0.3, 2, (3, 37))
    expected = magpy.core.magnet_cylinder_axial_Bfield(z0, r, z)
    out = magpy.core.magnet_cylinder_axial_Bfield(
        *(xp.asarray(val, dtype=xp.float32) for val in (z0, r, z))
    )
    np.testing.assert_allclose(np.asarray(out), expected, atol=1e-6)


def test_elliptic_iteration_ends():
    """Input that never converges gets NaN instead of an endless loop: NaN
    itself, and an observer on a current loop that is not caught as one."""
    b_z = magpy.core.magnet_cylinder_axial_Bfield(
        np.ones(3), np.array([0.5, 2.0, np.nan]), np.ones(3)
    )[2]
    np.testing.assert_array_equal(np.isnan(b_z), [False, False, True])
    with np.errstate(all="ignore"):
        h = magpy.func.circle_field("H", (1.0, 0.0, 1e-170), 2.0, 1.0)
    assert np.isnan(h).all()

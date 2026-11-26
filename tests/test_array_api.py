"""The core functions and the functional interface on JAX and PyTorch arrays.

The values are checked by test_regression_reference.py, which runs its core
and functional interface cases on these libraries too.
"""

from pathlib import Path

import numpy as np
import pytest

import magpylib as magpy
from magpylib._src.array_api_utils import cross, dot, padded_size
from magpylib._src.fields.special_cel import _cel

DATA = Path(__file__).parent / "testdata" / ("regression_reference" + ".npz")


@pytest.fixture(scope="module")
def reference():
    with np.load(DATA) as data:
        return dict(data)


@pytest.fixture(scope="module", params=["jax", "torch", "array_api_strict"])
def xp(request):
    """The array namespace of the library, computing in float64."""
    if request.param == "jax":
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        return jax.numpy
    return pytest.importorskip(request.param)


def _is_array(xp, val):
    return isinstance(val, type(xp.asarray(0.0)))


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


# torch looks for a Metal GPU on its first backward pass and warns on the
# virtual macOS machines of CI, which have none
@pytest.mark.filterwarnings("ignore:Skipping device Apple Paravirtual:UserWarning")
def test_backend_gradient(xp):
    """The field can be differentiated with respect to the observer."""
    if xp.__name__ == "array_api_strict":
        pytest.skip("array_api_strict has no automatic differentiation")
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


@pytest.mark.parametrize("scale", [1e-10, 1e-9, 1.0, 1e9])
@pytest.mark.filterwarnings("ignore:Field computation in:UserWarning")
def test_backend_float32_scale(xp, scale):
    """Sphere and dipole keep the precision of float32 from atoms to
    planets: no power of the distance underflows or overflows."""
    observers = np.array([(0.6, 0.7, 0.4)]) * scale
    moments = np.array([(1.0, 2.0, 3.0)]) * scale**3
    diameters = np.array([1.0]) * scale
    polarizations = np.array([(0.3, 0.2, 1.0)])

    def single(*arrays):
        return (xp.asarray(val, dtype=xp.float32) for val in arrays)

    out = magpy.func.dipole_field("B", *single(observers, moments))
    expected = magpy.func.dipole_field("B", observers, moments)
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5)
    out = magpy.func.sphere_field("B", *single(observers, diameters, polarizations))
    expected = magpy.func.sphere_field("B", observers, diameters, polarizations)
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5)


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


def test_padded_size():
    """Row counts are rounded up to four sizes per power of two."""
    sizes = [padded_size(n) for n in (1, 2, 7, 8, 9, 10, 11, 17, 1000, 1025)]
    assert sizes == [1, 2, 7, 8, 10, 10, 12, 20, 1024, 1280]
    assert all(n <= padded_size(n) <= 1.25 * n for n in range(1, 5000))


@pytest.mark.parametrize("n", [1, 2, 9, 11, 33])
@pytest.mark.parametrize(
    "name", sorted(set(magpy.core.__all__) - {"magnet_cylinder_segment_Hfield"})
)
def test_jax_padded_rows(name, n, reference):
    """JAX computes with rows padded to a size it has compiled before, and
    returns the NumPy result for the rows that were given."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    prefix = f"core/{name}|in|"
    inputs = {
        key.removeprefix(prefix): val[:n]
        for key, val in reference.items()
        if key.startswith(prefix)
    }
    func = getattr(magpy.core, name)
    expected = func(**inputs)
    out = func(**{key: jax.numpy.asarray(val) for key, val in inputs.items()})
    assert isinstance(out, jax.Array)
    np.testing.assert_allclose(out, expected, rtol=1e-9, atol=1e-12)


# Rows that take a special branch of their kernel, next to a row that takes
# the general one: (function, inputs of the rows).
BRANCH_ROWS = {
    "cylinder_field": {
        "observers": [(0.3, 0.2, 0.4), (0.01, 0.0, 0.4), (2.0, 0.1, 0.3), (0, 0, 3.0)],
        "dimensions": [(1.0, 2.0)] * 4,
        "polarizations": [(0, 0, 1.0), (1.0, 0, 0), (0.3, 0.2, 1.0), (0, 0, 0)],
    },
    "circle_field": {
        "observers": [(0.3, 0.2, 0.4), (1.0, 0.0, 0.0), (0.0, 0.0, 0.5)],
        "diameters": [2.0, 2.0, 0.0],
        "currents": [1.0, 2.0, 3.0],
    },
    "dipole_field": {
        "observers": [(0.3, 0.2, 0.4), (0.0, 0.0, 0.0)],
        "moments": [(1.0, 2.0, 3.0), (1.0, 0.0, -3.0)],
    },
    "triangle_charge_field": {
        # the second observer lies on the extension of an edge
        "observers": [(0.3, 0.2, 0.4), (2.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        "vertices": [[(0, 0, 0), (1, 0, 0), (0, 1, 0)]] * 3,
        "polarizations": [(0.1, 0.2, 1.0)] * 3,
    },
    "triangle_current_field": {
        # in the plane of the triangle: beside it, and beside each edge
        "observers": [
            (0.3, 0.2, 0.4),
            (2.0, 3.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (-1.0, 2.0, 0.0),
        ],
        "vertices": [[(0, 0, 0), (1, 0, 0), (0, 1, 0)]] * 5,
        "current_densities": [(1.0, 2.0, 0.0)] * 5,
    },
}


@pytest.mark.parametrize("name", BRANCH_ROWS)
def test_branches_skipped_or_selected(name):
    """A branch of a kernel that no row needs is skipped, and one that some
    rows need is computed for all rows and selected. Both give the same
    field: every row alone, all rows together, and all rows under jax.jit,
    where nothing can be skipped."""
    func = getattr(magpy.func, name)
    inputs = {key: np.array(val, dtype=float) for key, val in BRANCH_ROWS[name].items()}
    with np.errstate(all="ignore"):
        together = func("B", **inputs)
        alone = [
            func("B", **{key: val[i : i + 1] for key, val in inputs.items()})
            for i in range(len(together))
        ]
    assert np.isfinite(together[0]).all()
    assert np.abs(together[0]).max() > 0
    np.testing.assert_allclose(together, alone, rtol=1e-12, atol=1e-20)

    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    arrays = {key: jax.numpy.asarray(val) for key, val in inputs.items()}
    compiled = jax.jit(lambda arrays: func("B", **arrays))(arrays)
    np.testing.assert_allclose(compiled, together, rtol=1e-10, atol=1e-20)


def test_cel_shared_iteration(xp):
    """Several integrals of one kc are computed in one call, below and above
    the size where NumPy switches from scalars to arrays."""
    for n in (3, 40):
        rng = np.random.default_rng(n)
        kc = rng.uniform(0.05, 2, n)
        p = np.stack((np.ones(n), rng.uniform(0.1, 3, n), rng.uniform(-3, 0, n)))
        c = rng.uniform(-1, 1, (3, n))
        s = rng.uniform(-1, 1, (3, n))
        expected = [_cel(kc, p[i], c[i], s[i]) for i in range(3)]
        np.testing.assert_allclose(_cel(kc, p, c, s), expected, rtol=1e-12)
        out = _cel(*(xp.asarray(val) for val in (kc, p, c, s)))
        np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-12)


def test_dot_cross(xp):
    """The products of 3-vectors that NumPy gets written out."""
    a, b = np.random.default_rng(0).uniform(-1, 1, (2, 5, 4, 3))
    np.testing.assert_allclose(dot(a, b), np.sum(a * b, axis=-1), rtol=1e-14)
    np.testing.assert_allclose(cross(a, b), np.cross(a, b), rtol=1e-14, atol=1e-16)
    out = dot(xp.asarray(a), xp.asarray(b)), cross(xp.asarray(a), xp.asarray(b))
    np.testing.assert_allclose(np.asarray(out[0]), dot(a, b), rtol=1e-14)
    np.testing.assert_allclose(np.asarray(out[1]), cross(a, b), rtol=1e-14, atol=1e-16)


@pytest.mark.filterwarnings("ignore:`torch.jit.trace` is deprecated:FutureWarning")
def test_torch_trace_keeps_branches():
    """torch.jit.trace records the operations of its example input. A branch
    that this input does not need is still recorded, for other input."""
    torch = pytest.importorskip("torch")
    observers = torch.tensor([(0.3, 0.2, 0.4), (1.0, 2.0, 3.0)], dtype=torch.float64)
    moments = torch.tensor([(1.0, 0.0, -3.0)] * 2, dtype=torch.float64)
    traced = torch.jit.trace(magpy.core.dipole_Hfield, (observers, moments))
    at_origin = traced(torch.zeros_like(observers), moments)
    np.testing.assert_array_equal(at_origin, [(np.inf, 0.0, -np.inf)] * 2)

"""The core functions and the functional interface on JAX and PyTorch arrays.

The values are checked by test_regression_reference.py, which runs its core
and functional interface cases on these libraries too.
"""

from pathlib import Path

import numpy as np
import pytest
from array_api_compat import array_namespace
from scipy.spatial.transform import Rotation as R

import magpylib as magpy

pytestmark = [
    # torch looks for a Metal GPU on its first backward pass and warns on the
    # virtual macOS machines of CI, which have none
    pytest.mark.filterwarnings("ignore:Skipping device Apple Paravirtual:UserWarning"),
    # torch.func.jvp loads decompositions that torch compiles with its own
    # deprecated torch.jit.script
    pytest.mark.filterwarnings("ignore:`torch.jit.script` is deprecated:FutureWarning"),
]

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


FIELD_FUNCS = sorted(
    name
    for name in magpy.func.__all__
    if name.endswith("_field") and name != "cylinder_segment_field"
)


@pytest.mark.parametrize("name", FIELD_FUNCS)
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


def _jacobian(xp, func, point):
    """Jacobian of func at a point, by reverse mode differentiation."""
    if xp.__name__ == "torch":
        return np.asarray(xp.func.jacrev(func)(xp.asarray(point)))
    import jax  # noqa: PLC0415

    return np.asarray(jax.jacrev(func)(xp.asarray(point)))


def _jacobian_fd(func, point, step=1e-5):
    """Jacobian of func at a point, by central finite differences."""
    return np.stack(
        [(func(point + d) - func(point - d)) / (2 * step) for d in step * np.eye(3)],
        axis=-1,
    )


@pytest.mark.parametrize("radius", [0.0, 1e-12, 1e-7])
@pytest.mark.parametrize(
    "source",
    [
        ("circle_field", 2.0, 1e3),
        ("cylinder_field", (1.0, 2.0), (0.0, 0.0, 1.0)),
        ("cylinder_field", (1.0, 2.0), (0.3, -0.7, 0.0)),
        ("cylinder_field", (1.0, 2.0), (0.3, -0.7, 0.5)),
    ],
)
def test_backend_jacobian_on_axis(source, radius, xp):
    """The field of a circle and a cylinder can be differentiated on the axis
    and next to it, in- and outside of the cylinder, where the cylindrical
    coordinates of the observer cannot."""
    name, *inputs = source
    position = np.array((0.1, -0.2, 0.3))
    rotation = R.from_rotvec((0.3, -0.2, 0.5))

    def field(observer):
        return getattr(magpy.func, name)(
            "B", observer, *inputs, positions=position, orientations=rotation
        )

    for z in (-2.5, -0.4, 0.7):
        point = position + rotation.apply((radius, 0.0, z))
        expected = _jacobian_fd(field, point)
        np.testing.assert_allclose(
            _jacobian(xp, field, point), expected, atol=1e-8 * np.abs(expected).max()
        )


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
    assert set(JAX_INPUTS) == set(FIELD_FUNCS)


@pytest.mark.parametrize("name", JAX_INPUTS)
def test_jax_transformations(name):
    """The functional interface can be compiled and differentiated by the
    user: under jax.jit it gives the NumPy field, and the compiled gradient
    with respect to the observers is the finite difference of the field."""
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


# FORCE AND TORQUE ------------------------------------------------------------


def _loop_field(observers):
    return magpy.func.circle_field("B", observers, 2.0, 1e3)


CELLS = np.random.default_rng(0).uniform((-0.5, -0.5, 0.5), (0.5, 0.5, 1.5), (2, 7, 3))
PIVOT = (0.1, 0.2, 0.3)


def test_force_torque_is_getFT():
    """With NumPy arrays force_torque is the computation of getFT."""
    loop = magpy.current.Circle(diameter=2.0, current=1e3)
    dipole = magpy.misc.Dipole(moment=(1.0, 2.0, 3.0), position=(0.1, 0.2, 1.0))
    out = magpy.func.force_torque(
        _loop_field, [dipole.position], moments=[dipole.moment], pivot=PIVOT
    )
    np.testing.assert_allclose(out, magpy.getFT(loop, dipole, pivot=PIVOT), rtol=1e-14)


def test_force_torque_bad_input():
    with pytest.raises(ValueError, match="moments or current_vectors"):
        magpy.func.force_torque(_loop_field, CELLS[0])
    with pytest.raises(ValueError, match="moments or current_vectors"):
        magpy.func.force_torque(
            _loop_field, CELLS[0], moments=CELLS[1], current_vectors=CELLS[1]
        )
    with pytest.raises(ValueError, match="shape"):
        magpy.func.force_torque(_loop_field, CELLS[0, 0], moments=CELLS[1, 0])
    with pytest.raises(ValueError, match="eps"):
        magpy.func.force_torque(_loop_field, CELLS[0], moments=CELLS[1], eps=0.0)


@pytest.mark.parametrize("kind", ["moments", "current_vectors"])
def test_force_torque_backend(kind, xp):
    """Force and torque are arrays of the input library. On a magnet they
    have the gradient of the field from automatic differentiation, which the
    finite differences of NumPy match within their error."""
    points, vectors = CELLS
    expected = magpy.func.force_torque(
        _loop_field, points, **{kind: vectors}, pivot=PIVOT
    )
    out = magpy.func.force_torque(
        _loop_field, xp.asarray(points), **{kind: vectors}, pivot=PIVOT
    )
    assert all(_is_array(xp, val) for val in out)
    rtol = 1e-8 if kind == "moments" else 1e-13
    np.testing.assert_allclose([np.asarray(val) for val in out], expected, rtol=rtol)


POSITION = (0.3, -0.2, 1.1)
ROTATION = R.from_rotvec((0.2, 0.5, -0.3))
POL = (0.1, 0.2, 1.0)
# mesh function: target class, inputs, meshing values
MESHES = {
    "cuboid_mesh": (
        magpy.magnet.Cuboid,
        {"dimension": (1.0, 0.7, 0.4), "polarization": POL},
        [30, (2, 3, 4)],
    ),
    "cylinder_mesh": (
        magpy.magnet.Cylinder,
        {"dimension": (1.0, 0.7), "polarization": POL},
        [200, (5, 2, 3)],
    ),
    "cylinder_segment_mesh": (
        magpy.magnet.CylinderSegment,
        {"dimension": (0.3, 1.0, 0.7, 20.0, 250.0), "polarization": POL},
        [30, (5, 2, 3)],
    ),
    "tetrahedron_mesh": (
        magpy.magnet.Tetrahedron,
        {
            "vertices": [(0, 0, 0), (1, 0, 0), (0.2, 1, 0), (0.1, 0.3, 0.9)],
            "polarization": POL,
        },
        [30],
    ),
    "circle_mesh": (magpy.current.Circle, {"diameter": 1.2, "current": 2.0}, [31]),
}


def test_meshes_cover_func():
    meshes = {name for name in magpy.func.__all__ if name.endswith("_mesh")}
    assert set(MESHES) == meshes


def _to_xp(xp, inputs):
    """Inputs as arrays of the library, face indices staying a list."""
    return {
        key: val if key == "faces" else xp.asarray(np.array(val, dtype=float))
        for key, val in inputs.items()
    }


@pytest.mark.parametrize("name", MESHES)
def test_mesh_is_target_mesh(name):
    """With NumPy arrays a mesh function gives the mesh of its target in
    getFT."""
    cls, inputs, meshings = MESHES[name]
    target = cls(**inputs, position=POSITION, orientation=ROTATION)
    for meshing in meshings:
        with pytest.warns(DeprecationWarning, match="return_mesh"):
            (expected,) = magpy.getFT(target, target, meshing=meshing, return_mesh=True)
        out = getattr(magpy.func, name)(
            **inputs, meshing=meshing, position=POSITION, orientation=ROTATION
        )
        for val, exp in zip(out, expected.values(), strict=True):
            np.testing.assert_allclose(val, exp, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("name", MESHES)
def test_mesh_backend(name, xp):
    """A mesh function returns arrays of the input library with the values
    of NumPy."""
    _, inputs, meshings = MESHES[name]
    func = getattr(magpy.func, name)
    for meshing in meshings:
        expected = func(
            **inputs, meshing=meshing, position=POSITION, orientation=ROTATION
        )
        out = func(
            **_to_xp(xp, inputs),
            meshing=meshing,
            position=POSITION,
            orientation=ROTATION,
        )
        for val, exp in zip(out, expected, strict=True):
            assert _is_array(xp, val)
            np.testing.assert_allclose(
                np.asarray(val), exp, rtol=1e-13, atol=1e-15 * np.abs(exp).max()
            )


def test_mesh_bad_input():
    with pytest.raises(ValueError, match="meshing of cuboid_mesh"):
        magpy.func.cuboid_mesh((1, 1, 1), POL, meshing=1.5)
    with pytest.raises(ValueError, match="dimension of cuboid_mesh"):
        magpy.func.cuboid_mesh((1, 1), POL, meshing=8)
    with pytest.raises(TypeError, match="orientation"):
        magpy.func.cuboid_mesh((1, 1, 1), POL, meshing=8, orientation=(0, 0, 1))
    with pytest.raises(ValueError, match="single rotation"):
        magpy.func.cuboid_mesh(
            (1, 1, 1), POL, meshing=8, orientation=R.from_rotvec([(0, 0, 1)] * 2)
        )
    with pytest.raises(ValueError, match="polarization of cuboid_mesh"):
        magpy.func.cuboid_mesh((1, 1, 1), 1.0, meshing=8)
    with pytest.raises(ValueError, match="position of circle_mesh"):
        magpy.func.circle_mesh(1.0, 1.0, meshing=8, position=1.0)
    with pytest.raises(ValueError, match="0 <= r1 < r2"):
        magpy.func.cylinder_segment_mesh((2, 1, 1, 0, 90), POL, meshing=8)


@pytest.mark.parametrize("name", MESHES)
def test_mesh_gradient(name, xp):
    """A mesh can be differentiated with respect to its geometry."""
    _, inputs, meshings = MESHES[name]
    key = next(iter(inputs))
    geometry = np.array(inputs[key], dtype=float)

    def scalar(geometry):
        points, vectors = getattr(magpy.func, name)(
            **(_to_xp(array_namespace(geometry), inputs) | {key: geometry}),
            meshing=meshings[0],
            position=POSITION,
            orientation=ROTATION,
        )
        return (points**2).sum() + 1e-10 * (vectors**2).sum()

    if xp.__name__ == "torch":
        arg = xp.asarray(geometry).requires_grad_()
        (grad,) = xp.autograd.grad(scalar(arg), arg)
    else:
        import jax  # noqa: PLC0415

        grad = jax.grad(scalar)(xp.asarray(geometry))
    expected = np.empty(geometry.size)
    for ind in range(geometry.size):
        step = np.zeros(geometry.size)
        step[ind] = 1e-6
        step = step.reshape(geometry.shape)
        expected[ind] = (scalar(geometry + step) - scalar(geometry - step)) / 2e-6
    np.testing.assert_allclose(
        np.asarray(grad).ravel(),
        expected,
        rtol=1e-6,
        atol=1e-8 * np.abs(expected).max(),
    )


def _cuboid_force(dimension, position, meshing=(2, 3, 2)):
    """Force of a current loop on a cuboid magnet."""
    points, moments = magpy.func.cuboid_mesh(dimension, POL, meshing, position)
    return magpy.func.force_torque(_loop_field, points, moments=moments)[0]


@pytest.mark.parametrize("meshing", [12, (2, 3, 2)])
def test_force_gradient_geometry(meshing, xp):
    """The force can be differentiated with respect to the position and the
    dimension of the target, also when the number of cells comes from the
    values of the dimension."""
    dimension, position = np.array((1.0, 0.7, 0.4)), np.array(POSITION)

    def force_z(dimension, position):
        return _cuboid_force(dimension, position, meshing)[2]

    if xp.__name__ == "torch":
        args = [xp.asarray(val).requires_grad_() for val in (dimension, position)]
        grads = xp.autograd.grad(force_z(*args), args)
    else:
        import jax  # noqa: PLC0415

        grads = jax.grad(force_z, argnums=(0, 1))(
            xp.asarray(dimension), xp.asarray(position)
        )
    for arg, grad in enumerate(grads):
        expected = np.empty(3)
        for axis, step in enumerate(1e-4 * np.eye(3)):
            ahead, behind = [dimension, position], [dimension, position]
            ahead[arg], behind[arg] = ahead[arg] + step, behind[arg] - step
            expected[axis] = (force_z(*ahead) - force_z(*behind)) / 2e-4
        np.testing.assert_allclose(np.asarray(grad), expected, rtol=1e-6)


def test_jax_jit_force():
    """Mesh and force can be compiled when the meshing fixes the number of
    cells. A number of cells that comes from the values of the dimension
    cannot be traced, which is an error."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    dimension, position = np.array((1.0, 0.7, 0.4)), np.array(POSITION)
    out = jax.jit(_cuboid_force)(jax.numpy.asarray(dimension), position)
    np.testing.assert_allclose(out, _cuboid_force(dimension, position), rtol=1e-8)
    with pytest.raises(ValueError, match="has no values while it is traced"):
        jax.jit(lambda dim: _cuboid_force(dim, position, 12))(
            jax.numpy.asarray(dimension)
        )


@pytest.mark.parametrize(
    "target",
    [("cuboid_mesh", (1.0, 1.0, 1.0), (3, 3, 3)), ("cylinder_mesh", (1.0, 1.0), 200)],
)
@pytest.mark.parametrize(
    "source",
    [
        ("circle_field", 2.0, 1e3),
        ("cylinder_field", (2.0, 1.0), (0.3, 0.0, 1.0)),
    ],
)
def test_force_coaxial(source, target, xp):
    """The force on a magnet that shares the axis of a current loop or a
    cylinder magnet is finite and the one of NumPy: it has cells on that
    axis (cuboid) or within rounding of it (cylinder)."""
    name, *inputs = source
    mesh, dimension, meshing = target
    position = (0.0, 0.0, 2.0)

    def field(observers):
        return getattr(magpy.func, name)("B", observers, *inputs)

    def force_torque(dimension):
        points, moments = getattr(magpy.func, mesh)(
            dimension, (0.0, 0.0, 1.0), meshing, position
        )
        radius = np.hypot(*np.asarray(points)[:, :2].T)
        assert (radius < 1e-15).any()
        assert (radius == 0).any() == (mesh == "cuboid_mesh")
        return magpy.func.force_torque(field, points, moments=moments, pivot=position)

    expected = force_torque(np.array(dimension))
    out = [np.asarray(val) for val in force_torque(xp.asarray(np.array(dimension)))]
    scale = np.abs(expected[0]).max()
    np.testing.assert_allclose(out, expected, rtol=1e-7, atol=1e-9 * scale)

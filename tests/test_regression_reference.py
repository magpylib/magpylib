"""Regression tests against reference outputs frozen from the NumPy backend.

Every case calls the public API (``magpylib.core``, ``magpylib.func``, the
object interface, ``getFT``) with inputs stored in
``testdata/regression_reference.npz`` and compares with the output stored next
to them. The inputs are stored, not redrawn, because NumPy does not promise
the same random stream across versions.

Regenerate the reference, on purpose only, with

    uv run python tests/test_regression_reference.py

which also prints how much of each tolerance the rounding noise of the current
code uses up (see ``_noise``).
"""

import sys
import warnings
from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import magpylib as magpy

DATA = Path(__file__).parent / "testdata" / "regression_reference.npz"

# A case passes if every field vector is off by at most RTOL of its length,
# plus ATOL_REL of the largest field of the case for the vectors that cancel
# to (almost) zero. A rewrite reorders floating-point operations, so the bar
# is rounding noise, not bits: see _noise for how the numbers were chosen.
RTOL = 1e-9
ATOL_REL = 1e-12
# Looser (rtol, atol_rel) for the kernels with more rounding noise: with the
# default, their noise is up to 0.05 of the tolerance, against 0.01 and less
# for all others. First match wins.
TOL = {
    # observers next to a singularity: kept only if one ulp moves the field by
    # less than 1e-6 (_steady); a point on the wrong side of a band is far off
    "*/near": (1e-4, ATOL_REL),
    # the current code loses digits to cancellation with the distance: noise
    # of 1e-6 at 300 times the size of a tetrahedron
    "*/far": (1e-4, ATOL_REL),
    "*etrahedron*": (1e-8, ATOL_REL),
    "*ylinder*": (1e-8, ATOL_REL),
    "func/triangle_charge_field/*": (1e-8, ATOL_REL),
    # finite differences of the field with step 1e-5 amplify its rounding
    "force/*": (1e-7, 1e-10),
    "scene/groups/*": (1e-8, ATOL_REL),  # holds cylinders and tetrahedra
}
# The largest field that ATOL_REL refers to is taken separately along these
# axes, where one case holds fields of very different size: per source, per
# force and torque of each target, per observer next to a singularity.
ATOL_AXES = {
    "force/*": (0, -2),
    "scene/[BHJM]": (0,),
    "scene/groups/*": (0,),
    "*/near": (0,),
    "*/far": (0,),
}
# These return the components along the first axis, shape (3, i).
COMPONENTS_FIRST = (
    "core/current_circle_Hfield",
    "core/magnet_cylinder_axial_Bfield",
    "core/magnet_cylinder_diametral_Hfield",
)

GET = {"B": magpy.getB, "H": magpy.getH, "J": magpy.getJ, "M": magpy.getM}
POL = (0.1, 0.2, 1.0)
N_CORE, N_FUNC, N_OBS = 300, 200, 40

# name -> (run, make, nudge): run(**inputs) gives the output, make(rng) the
# inputs as a dict of arrays, and nudge says if the inputs are generic points,
# where a few ulp cannot move an observer across a surface.
CASES = {}


def _case(name, run, make=lambda _: {}, nudge=True):
    CASES[name] = (run, make, nudge)


def _quiet(run, inputs, keys):
    """The inputs without the rows, along the arrays named by keys, where the
    current code warns or is not finite: those have no reference."""

    def quiet(row):
        one = {
            key: val[row : row + 1].copy() if key in keys else val.copy()
            for key, val in inputs.items()
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                return bool(np.all(np.isfinite(run(**one))))
            except Warning:
                return False

    keep = [row for row in range(len(inputs[keys[0]])) if quiet(row)]
    return {key: val[keep] if key in keys else val for key, val in inputs.items()}


def _steady(run, inputs, rng, limit=1e-6):
    """The inputs without the observers whose field vector moves by more than
    limit of its length when they move by one ulp: so close to a singularity
    the current code returns noise, which is no reference."""
    inputs = _quiet(run, inputs, ("observers",))
    observers = inputs["observers"]
    ref = run(**inputs)
    steady = np.ones(len(observers), dtype=bool)
    eps = np.finfo(float).eps
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for _ in range(5):
            nudged = observers * (1 + eps * rng.choice((-1.0, 1.0), observers.shape))
            moved = np.linalg.norm(run(**inputs | {"observers": nudged}) - ref, axis=1)
            steady &= moved <= limit * np.linalg.norm(ref, axis=1)  # false for NaN
    return inputs | {"observers": observers[steady]}


def _uniform(rng, low, high, *shape):
    return rng.uniform(low, high, shape)


TRIANGLE = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
TETRAHEDRON = [*TRIANGLE, (0, 0, 1)]


def _shape(rng, vertices, *s):
    """Random triangles or tetrahedra that are far from degenerate, where
    the field is ill-conditioned: the given one with each vertex moved."""
    vertices = np.array(vertices, dtype=float)
    return vertices + _uniform(rng, -0.3, 0.3, *s, *vertices.shape)


# CORE -----------------------------------------------------------------------


def _core_segment(rng):
    n = N_CORE
    r1 = _uniform(rng, 0.2, 1, n)
    phi1 = _uniform(rng, -np.pi, 0, n)
    return {
        "observers": np.stack(
            [
                _uniform(rng, 0.1, 4, n),
                _uniform(rng, -np.pi, np.pi, n),
                _uniform(rng, -3, 3, n),
            ],
            axis=1,
        ),
        "dimensions": np.stack(
            [
                r1,
                r1 + _uniform(rng, 0.2, 2, n),
                phi1,
                phi1 + _uniform(rng, 0.3, 3, n),
                _uniform(rng, -1.5, -0.1, n),
                _uniform(rng, 0.1, 1.5, n),
            ],
            axis=1,
        ),
        "magnetizations": np.stack(
            [
                _uniform(rng, 1e5, 1e6, n),
                _uniform(rng, -np.pi, np.pi, n),
                _uniform(rng, 0, np.pi, n),
            ],
            axis=1,
        ),
    }


CORE = {
    "current_circle_Hfield": lambda rng: {
        "r0": _uniform(rng, 0.5, 2, N_CORE),
        "r": _uniform(rng, 0.05, 5, N_CORE),
        "z": _uniform(rng, -3, 3, N_CORE),
        "i0": _uniform(rng, -5, 5, N_CORE),
    },
    "current_polyline_Hfield": lambda rng: {
        "observers": _uniform(rng, -3, 3, N_CORE, 3),
        "segments_start": _uniform(rng, -1, 1, N_CORE, 3),
        "segments_end": _uniform(rng, -1, 1, N_CORE, 3),
        "currents": _uniform(rng, -5, 5, N_CORE),
    },
    "current_sheet_Hfield": lambda rng: {
        "observers": _uniform(rng, -3, 3, N_CORE, 3),
        "vertices": _shape(rng, TRIANGLE, N_CORE),
        "current_densities": _uniform(rng, -5, 5, N_CORE, 3),
    },
    "dipole_Hfield": lambda rng: {
        "observers": _uniform(rng, -3, 3, N_CORE, 3),
        "moments": _uniform(rng, -5, 5, N_CORE, 3),
    },
    "magnet_cuboid_Bfield": lambda rng: {
        "observers": _uniform(rng, -3, 3, N_CORE, 3),
        "dimensions": _uniform(rng, 0.2, 2, N_CORE, 3),
        "polarizations": _uniform(rng, -1, 1, N_CORE, 3),
    },
    "magnet_cylinder_axial_Bfield": lambda rng: {
        "z0": _uniform(rng, 0.2, 3, N_CORE),
        "r": _uniform(rng, 0.01, 4, N_CORE),
        "z": _uniform(rng, -4, 4, N_CORE),
    },
    "magnet_cylinder_diametral_Hfield": lambda rng: {
        "z0": _uniform(rng, 0.2, 3, N_CORE),
        "r": _uniform(rng, 0.01, 4, N_CORE),
        "z": _uniform(rng, -4, 4, N_CORE),
        "phi": _uniform(rng, -np.pi, np.pi, N_CORE),
    },
    "magnet_cylinder_segment_Hfield": _core_segment,
    "magnet_sphere_Bfield": lambda rng: {
        "observers": _uniform(rng, -3, 3, N_CORE, 3),
        "diameters": _uniform(rng, 0.2, 3, N_CORE),
        "polarizations": _uniform(rng, -1, 1, N_CORE, 3),
    },
    "triangle_Bfield": lambda rng: {
        "observers": _uniform(rng, -3, 3, N_CORE, 3),
        "vertices": _shape(rng, TRIANGLE, N_CORE),
        "polarizations": _uniform(rng, -1, 1, N_CORE, 3),
    },
}
for _name, _make in CORE.items():
    _case(f"core/{_name}", getattr(magpy.core, _name), _make)

# the series of the diametral cylinder for small r, which the rows above
# only touch
_case(
    "core/magnet_cylinder_diametral_Hfield/small_r",
    magpy.core.magnet_cylinder_diametral_Hfield,
    lambda rng: {
        "z0": _uniform(rng, 0.2, 3, N_CORE),
        "r": _uniform(rng, 1e-3, 0.05, N_CORE),
        "z": _uniform(rng, -4, 4, N_CORE),
        "phi": _uniform(rng, -np.pi, np.pi, N_CORE),
    },
)


def _core_segment_cases(_):
    """Observers in the planes, on the cylinders and on the axis of the faces
    of a section with and without hole: together they reach every case
    function, which random observers never do."""
    grid = np.array(
        [
            (r, phi, z, r1)
            for r in (0, 0.5, 1, 2, 3)
            for phi in (0, np.pi / 2, np.pi, -np.pi / 2, 0.3, 2)
            for z in (-1, 1, 0.3, 3)
            for r1 in (0, 1)
        ],
        dtype=float,
    )
    n = len(grid)
    inputs = {
        "observers": grid[:, :3],
        "dimensions": np.column_stack(
            [grid[:, 3], *np.broadcast_to((2, 0, np.pi / 2, -1, 1), (n, 5)).T]
        ),
        "magnetizations": np.broadcast_to((1e5, 0.4, 1.1), (n, 3)),
    }
    return _quiet(magpy.core.magnet_cylinder_segment_Hfield, inputs, tuple(inputs))


_case(
    "core/magnet_cylinder_segment_Hfield/cases",
    magpy.core.magnet_cylinder_segment_Hfield,
    _core_segment_cases,
    nudge=False,
)


# FUNCTIONAL INTERFACE -------------------------------------------------------
# Source parameters of shape (*s, ...): s is (n,) for one source per observer
# and () for a single source.


def _func_segment_dims(rng, s):
    r1 = _uniform(rng, 0.1, 1, *s)
    phi1 = _uniform(rng, -180, 90, *s)
    return np.stack(
        [
            r1,
            r1 + _uniform(rng, 0.2, 1.5, *s),
            _uniform(rng, 0.3, 2, *s),
            phi1,
            phi1 + _uniform(rng, 20, 340, *s),
        ],
        axis=-1,
    )


FUNC = {
    "circle_field": lambda rng, s: {
        "diameters": _uniform(rng, 0.5, 3, *s),
        "currents": _uniform(rng, -5, 5, *s),
    },
    "polyline_field": lambda rng, s: {
        "segments_start": _uniform(rng, -1, 1, *s, 3),
        "segments_end": _uniform(rng, -1, 1, *s, 3),
        "currents": _uniform(rng, -5, 5, *s),
    },
    "cuboid_field": lambda rng, s: {
        "dimensions": _uniform(rng, 0.3, 2, *s, 3),
        "polarizations": _uniform(rng, -1, 1, *s, 3),
    },
    "cylinder_field": lambda rng, s: {
        "dimensions": _uniform(rng, 0.3, 2, *s, 2),
        "polarizations": _uniform(rng, -1, 1, *s, 3),
    },
    "cylinder_segment_field": lambda rng, s: {
        "dimensions": _func_segment_dims(rng, s),
        "polarizations": _uniform(rng, -1, 1, *s, 3),
    },
    "sphere_field": lambda rng, s: {
        "diameters": _uniform(rng, 0.3, 3, *s),
        "polarizations": _uniform(rng, -1, 1, *s, 3),
    },
    "tetrahedron_field": lambda rng, s: {
        "vertices": _shape(rng, TETRAHEDRON, *s),
        "polarizations": _uniform(rng, -1, 1, *s, 3),
    },
    "dipole_field": lambda rng, s: {"moments": _uniform(rng, -5, 5, *s, 3)},
    "triangle_charge_field": lambda rng, s: {
        "vertices": _shape(rng, TRIANGLE, *s),
        "polarizations": _uniform(rng, -1, 1, *s, 3),
    },
    "triangle_current_field": lambda rng, s: {
        "vertices": _shape(rng, TRIANGLE, *s),
        "current_densities": _uniform(rng, -5, 5, *s, 3),
    },
}

# One source at the origin, unrotated, and the observers where its field is
# special. All are exact in floating point, so that no rounding decides on
# which side of a surface a point lies. The object cases reuse them.
SPECIAL = {
    "circle_field": (
        {"diameters": 2.0, "currents": 1.5},
        [
            (0, 0, 0),  # center
            (0, 0, 1),  # on the axis
            (1, 0, 0),  # on the loop
            (0, -1, 0),  # on the loop
            (0.5, 0, 0),  # in the plane, inside
            (3, 0, 0),  # in the plane, outside
            (0.3, 0.4, 0.5),
        ],
    ),
    "polyline_field": (
        {"segments_start": (0, 0, 0), "segments_end": (1, 0, 0), "currents": 1.5},
        [
            (0.5, 0, 0),  # on the segment
            (0, 0, 0),  # at its start
            (1, 0, 0),  # at its end
            (2, 0, 0),  # on its line, beyond the end
            (-1, 0, 0),  # on its line, before the start
            (0.5, 1, 0),
            (0.3, 0.4, 0.5),
        ],
    ),
    "cuboid_field": (
        {"dimensions": (2, 2, 2), "polarizations": POL},
        [
            (0, 0, 0),  # center
            (0.1, 0.2, 0.3),  # inside
            (1, 0.3, 0.2),  # on a face
            (0.3, -1, 0.2),  # on a face
            (1, 1, 0.3),  # on an edge
            (1, 0.3, -1),  # on an edge
            (1, 1, 1),  # on a corner
            (-1, 1, -1),  # on a corner
            (1, 3, 0),  # in the plane of a face, outside
            (1, 1, 3),  # on the line of an edge, outside
            (2, 3, 4),
        ],
    ),
    "cylinder_field": (
        {"dimensions": (2, 2), "polarizations": POL},
        [
            (0, 0, 0),  # center
            (0.1, 0.2, 0.3),  # inside
            (0, 0, 0.5),  # on the axis, inside
            (0, 0, 3),  # on the axis, outside
            (1, 0, 0.3),  # on the hull
            (0, -1, 0.3),  # on the hull
            (0.25, 0.5, 1),  # on a base
            (0, 0, 1),  # on a base, on the axis
            (1, 0, 1),  # on an edge
            (0, 1, -1),  # on an edge
            (1, 0, 3),  # on the hull's extension
            (3, 0, 1),  # in the plane of a base, outside
            (2, 3, 4),
        ],
    ),
    "cylinder_segment_field": (
        {"dimensions": (1, 2, 2, 0, 90), "polarizations": POL},
        [
            (1.5, 0.5, 0.25),  # inside
            (0, 0, 0),  # on the axis
            (0, 0, 3),  # on the axis, outside
            (0.5, 0.5, 0),  # in the hole
            (1, 0, 0.5),  # on the inner hull, on the φ1 face
            (0, 2, 0.5),  # on the outer hull, on the φ2 face
            (1.5, 0, 0.5),  # on the φ1 face
            (0, 1.5, 0.5),  # on the φ2 face
            (1.5, 0, 1),  # on an edge
            (0, 2, -1),  # on a corner
            (0.75, 1, 1),  # on the top
            (-1.5, 0.5, 0.25),  # in the ring, outside the section
            (3, 0, 1),  # in the planes of the top and of the φ1 face
            (3, 0, 0.5),  # in the plane of the φ1 face
            (-3, 0, 0.5),  # opposite the φ1 face
            (1, 0, 3),  # on the inner hull's and the φ1 face's extension
            (0, -1, 1),  # on the inner hull's extension, in the top's plane
            (0, -2, 0.3),  # on the outer hull's extension, opposite φ2
            (0.5, 0, 1),  # in the hole, in two planes
            (0, 0, 1),  # on the axis, in the top's plane
            (2, 3, 4),
        ],
    ),
    "sphere_field": (
        {"diameters": 2.0, "polarizations": POL},
        [
            (0, 0, 0),  # center
            (0.1, 0.2, 0.3),  # inside
            (1, 0, 0),  # on the surface
            (0, 0, -1),  # on the surface
            (2, 3, 4),
        ],
    ),
    "tetrahedron_field": (
        {"vertices": TETRAHEDRON, "polarizations": POL},
        [
            (0.125, 0.125, 0.125),  # inside
            (0, 0, 0),  # on a corner
            (0, 0, 1),  # on a corner
            (0.5, 0, 0),  # on an edge
            (0.25, 0.25, 0),  # on a face
            (0.25, 0.25, 0.5),  # on the oblique face
            (2, 2, 0),  # in the plane of a face, outside
            (2, 3, 4),
        ],
    ),
    "dipole_field": (
        {"moments": POL},
        [
            (0, 0, 0),  # at the dipole
            (0, 0, 1),
            (1, 0, 0),
            (2, 3, 4),
        ],
    ),
    "triangle_charge_field": (
        {"vertices": TRIANGLE, "polarizations": POL},
        [
            (0.25, 0.25, 0),  # on the triangle
            (0, 0, 0),  # on a corner
            (0.5, 0, 0),  # on an edge
            (0.5, 0.5, 0),  # on the oblique edge
            (2, 0, 0),  # on the line of an edge
            (2, 2, 0),  # in the plane, outside
            (0.25, 0.25, 1),  # above
            (2, 3, 4),
        ],
    ),
    "triangle_current_field": (
        {"vertices": TRIANGLE, "current_densities": (1.0, 2.0, 3.0)},
        [
            (0.25, 0.25, 0),  # on the sheet
            (0, 0, 0),  # on a corner
            (0.5, 0, 0),  # on an edge
            (0.5, 0.5, 0),  # on the oblique edge
            (2, 0, 0),  # on the line of an edge
            (2, 2, 0),  # in the plane, outside
            (0.25, 0.25, 1),  # above
            (2, 3, 4),
        ],
    ),
}

# Sources without extent, without strength, or with negative or unordered
# dimensions, one per row, seen from a generic point and from the origin. A
# cylinder with negative dimensions has no reference: its field differs from
# the positive one, which looks like a bug.
DEGENERATE = {
    "circle_field": {
        "diameters": [0, 2, 0, -2],
        "currents": [1.5, 0, 0, 1.5],
    },
    "polyline_field": {
        "segments_start": [(0, 0, 0), (1, 2, 3), (0, 0, 0)],
        "segments_end": [(0, 0, 0), (1, 2, 3), (1, 0, 0)],
        "currents": [1.5, 1.5, 0],
    },
    "cuboid_field": {
        "dimensions": [(0, 2, 2), (2, 0, 0), (0, 0, 0), (2, 2, 2), (-2, 2, -2)],
        "polarizations": [POL, POL, POL, (0, 0, 0), POL],
    },
    "cylinder_field": {
        "dimensions": [(2, 0), (2, 2), (2, 2), (2, 2)],
        "polarizations": [POL, (0, 0, 0), (0, 0, 1), (1, 0, 0)],
    },
    "cylinder_segment_field": {
        "dimensions": [
            (1, 2, 2, 0, 90),  # no polarization
            (0, 2, 2, 0, 90),  # no hole
            (1, 2, 2, 0, 360),  # full ring
            (0, 2, 2, 0, 360),  # full cylinder
            (1, 2, 2, 350, 370),  # across 0 (deg)
            (1, 2, 2, -10, 10),  # the same section
            (1, 2, 2, -270, -90),  # negative angles
            (-1, -2, -2, 0, 90),  # negative lengths
            (2, 1, 2, 0, 90),  # r1 > r2
            (1, 2, 2, 90, 0),  # phi1 > phi2
        ],
        "polarizations": [(0, 0, 0), *[POL] * 9],
    },
    "sphere_field": {
        "diameters": [0, 2, -2],
        "polarizations": [POL, (0, 0, 0), POL],
    },
    "tetrahedron_field": {
        "vertices": [
            TETRAHEDRON,
            [TETRAHEDRON[i] for i in (1, 0, 2, 3)],  # other chirality
        ],  # a flat tetrahedron raises LinAlgError, so it has no reference
        "polarizations": [(0, 0, 0), POL],
    },
    "dipole_field": {"moments": [(0, 0, 0), (0, 0, 1), (1, 0, 0)]},
    "triangle_charge_field": {
        "vertices": [
            TRIANGLE,
            TRIANGLE,  # polarization in the plane
        ],  # a triangle without area warns (0/0), so it has no reference
        "polarizations": [(0, 0, 0), (1, 1, 0)],
    },
    "triangle_current_field": {
        "vertices": [
            TRIANGLE,
            TRIANGLE,  # current density normal to the sheet
            [(0, 0, 0), (1, 0, 0), (2, 0, 0)],  # collinear
        ],
        "current_densities": [(0, 0, 0), (0, 0, 3), (1, 2, 3)],
    },
}


def _func(name, field, **fixed):
    func = getattr(magpy.func, name)

    def run(**inputs):
        if "orientations" in inputs:
            inputs["orientations"] = R.from_quat(inputs["orientations"])
        return func(field, **inputs, **fixed)

    return run


def _arrays(inputs):
    return {key: np.array(val, dtype=float) for key, val in inputs.items()}


def _func_random(make):
    """One source per observer, each with its own position and orientation."""
    n = N_FUNC
    return lambda rng: {
        "observers": _uniform(rng, -3, 3, n, 3),
        "positions": _uniform(rng, -0.5, 0.5, n, 3),
        "orientations": rng.normal(size=(n, 4)),
        **make(rng, (n,)),
    }


def _func_broadcast(make):
    """A single source, moved and rotated, for all observers."""
    return lambda rng: {
        "observers": _uniform(rng, -3, 3, N_FUNC, 3),
        "positions": _uniform(rng, -0.5, 0.5, 3),
        "orientations": rng.normal(size=4),
        **make(rng, ()),
    }


def _func_single(make):
    """A single source and a single observer."""
    return lambda rng: {"observers": _uniform(rng, -3, 3, 3), **make(rng, ())}


def _func_degenerate(sources):
    rows = len(next(iter(sources.values())))
    observers = [(0.3, 0.4, 0.5)] * rows + [(0, 0, 0)] * rows
    sources = {key: [*val, *val] for key, val in sources.items()}
    return lambda _: _arrays({"observers": observers, **sources})


# A unit vector that crosses every face, edge and line of SPECIAL, and the
# distances from them: within the bands of 1e-15 that count as on the
# surface, well outside of them, and outside of the widest band (1e-10).
ACROSS = np.array((0.6, 0.48, 0.64))
DISTANCES = (4e-16, 1e-13, 1e-9)


def _func_near(run, source, observers):
    """The special observers, moved off by tiny distances: once away from the
    origin, which crosses the surfaces of the bodies centred there and stays
    in the planes z=0, and once along ACROSS."""
    observers = np.array(observers, dtype=float)
    near = [
        moved
        for dist in DISTANCES
        for sign in (1, -1)
        for moved in (observers * (1 + sign * dist), observers + sign * dist * ACROSS)
    ]
    inputs = _arrays({"observers": np.concatenate(near), **source})
    return lambda rng: _steady(run, inputs, rng)


def _func_far(make):
    """A single source seen from 10 to 300 times its size. Further out the
    current code loses its digits to cancellation."""

    def inputs(rng):
        direction = rng.normal(size=(N_FUNC, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        return {
            "observers": direction * 10 ** _uniform(rng, 1, 2.5, N_FUNC, 1),
            "positions": _uniform(rng, -0.5, 0.5, 3),
            "orientations": rng.normal(size=4),
            **make(rng, ()),
        }

    return inputs


for _name, _make in FUNC.items():
    _source, _observers = SPECIAL[_name]
    _special = _arrays({"observers": _observers, **_source})
    for _field in "BH":
        _run = _func(_name, _field)
        _case(f"func/{_name}/{_field}/random", _run, _func_random(_make))
        _case(f"func/{_name}/{_field}/broadcast", _run, _func_broadcast(_make))
        _case(
            f"func/{_name}/{_field}/special",
            _run,
            lambda _, inputs=_special: inputs,
            nudge=False,
        )
        _case(
            f"func/{_name}/{_field}/degenerate",
            _run,
            _func_degenerate(DEGENERATE[_name]),
            nudge=False,
        )
        _case(
            f"func/{_name}/{_field}/near",
            _run,
            _func_near(_run, _source, _observers),
            nudge=False,
        )
    _case(f"func/{_name}/B/far", _func(_name, "B"), _func_far(_make))
    _case(f"func/{_name}/B/single", _func(_name, "B"), _func_single(_make))
    _case(
        f"func/{_name}/B/single_no_squeeze",
        _func(_name, "B", squeeze=False),
        _func_single(_make),
    )


# OBJECTS --------------------------------------------------------------------

PATH = {
    "position": np.linspace((0, 0, 0), (0.4, -0.2, 0.3), 5),
    "orientation": R.from_euler(
        "xyz", np.linspace((0, 0, 0), (40, -25, 70), 5), degrees=True
    ),
}
CUBE_VERTICES = [
    (-1, -1, -1),
    (-1, -1, 1),
    (-1, 1, -1),
    (-1, 1, 1),
    (1, -1, -1),
    (1, -1, 1),
    (1, 1, -1),
    (1, 1, 1),
]
CUBE_FACES = [
    (6, 0, 2),
    (6, 4, 0),
    (5, 0, 4),
    (5, 1, 0),
    (5, 4, 6),
    (5, 6, 7),
    (3, 2, 0),
    (3, 0, 1),
    (3, 6, 2),
    (3, 7, 6),
    (3, 1, 5),
    (3, 5, 7),
]
SQUARE = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]

# Class name -> (builder, special observers). The builders take the geometry
# of SPECIAL where there is one, so that its observers are special here too.
OBJECTS = {
    "Cuboid": (
        lambda **kw: magpy.magnet.Cuboid(dimension=(2, 2, 2), polarization=POL, **kw),
        SPECIAL["cuboid_field"][1],
    ),
    "Cylinder": (
        lambda **kw: magpy.magnet.Cylinder(dimension=(2, 2), polarization=POL, **kw),
        SPECIAL["cylinder_field"][1],
    ),
    "CylinderSegment": (
        lambda **kw: magpy.magnet.CylinderSegment(
            dimension=(1, 2, 2, 0, 90), polarization=POL, **kw
        ),
        SPECIAL["cylinder_segment_field"][1],
    ),
    "CylinderSegment_ring": (
        lambda **kw: magpy.magnet.CylinderSegment(
            dimension=(1, 2, 2, 0, 360), polarization=POL, **kw
        ),
        SPECIAL["cylinder_segment_field"][1],
    ),
    "CylinderSegment_full": (
        lambda **kw: magpy.magnet.CylinderSegment(
            dimension=(0, 2, 2, 0, 360), polarization=POL, **kw
        ),
        SPECIAL["cylinder_field"][1],
    ),
    "Sphere": (
        lambda **kw: magpy.magnet.Sphere(diameter=2, polarization=POL, **kw),
        SPECIAL["sphere_field"][1],
    ),
    "Tetrahedron": (
        lambda **kw: magpy.magnet.Tetrahedron(
            vertices=TETRAHEDRON, polarization=POL, **kw
        ),
        SPECIAL["tetrahedron_field"][1],
    ),
    "TriangularMesh": (
        lambda **kw: magpy.magnet.TriangularMesh(
            vertices=CUBE_VERTICES, faces=CUBE_FACES, polarization=POL, **kw
        ),
        SPECIAL["cuboid_field"][1],
    ),
    "Circle": (
        lambda **kw: magpy.current.Circle(diameter=2, current=1.5, **kw),
        SPECIAL["circle_field"][1],
    ),
    "Polyline": (
        lambda **kw: magpy.current.Polyline(
            current=1.5,
            vertices=[(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 0), (0, 1, 1)],
            **kw,
        ),
        [*SPECIAL["polyline_field"][1], (1, 0.5, 0), (1, 1, 0)],
    ),
    "TriangleSheet": (
        lambda **kw: magpy.current.TriangleSheet(
            vertices=SQUARE,
            faces=[(0, 1, 2), (1, 3, 2)],
            current_densities=[(1, 2, 3), (-2, 1, 0)],
            **kw,
        ),
        [*SPECIAL["triangle_current_field"][1], (0.75, 0.75, 0), (1, 1, 0)],
    ),
    "TriangleStrip": (
        lambda **kw: magpy.current.TriangleStrip(
            current=1.5, vertices=[*SQUARE, (0, 2, 0.5), (1, 2, 0.5)], **kw
        ),
        [*SPECIAL["triangle_current_field"][1], (0.75, 0.75, 0), (1, 1, 0)],
    ),
    "Dipole": (
        lambda **kw: magpy.misc.Dipole(moment=POL, **kw),
        SPECIAL["dipole_field"][1],
    ),
    "Triangle": (
        lambda **kw: magpy.misc.Triangle(vertices=TRIANGLE, polarization=POL, **kw),
        SPECIAL["triangle_charge_field"][1],
    ),
}


def _obj(build, field, path=False, **kwargs):
    def run(observers):
        source = build(**PATH) if path else build()
        return GET[field](source, observers, **kwargs)

    return run


def _observers(rng):
    return {"observers": _uniform(rng, -2.5, 2.5, N_OBS, 3)}


for _name, (_build, _observers_special) in OBJECTS.items():
    _special = _arrays({"observers": _observers_special})
    for _field in "BHJM":
        _case(f"obj/{_name}/{_field}/path", _obj(_build, _field, path=True), _observers)
        _case(
            f"obj/{_name}/{_field}/special",
            _obj(_build, _field),
            lambda _, inputs=_special: inputs,
            nudge=False,
        )

for _name in ("TriangularMesh", "Tetrahedron"):
    for _in_out in ("inside", "outside"):
        for _field in "BHJM":
            _case(
                f"obj/{_name}/{_field}/in_out_{_in_out}",
                _obj(OBJECTS[_name][0], _field, path=True, in_out=_in_out),
                _observers,
            )


# SCENES ---------------------------------------------------------------------


def _scene(field, collection=False, ragged=False, **kwargs):
    def run(pixel):
        sources = [
            OBJECTS["Cuboid"][0](**PATH),
            magpy.magnet.Cylinder(
                dimension=(1, 2),
                magnetization=(1e5, -2e5, 3e5),
                position=(3, 0.5, -1),
                orientation=R.from_rotvec((0.2, -0.3, 0.5)),
            ),
            OBJECTS["Circle"][0](position=(-2, 1, 0.5)),
            OBJECTS["Dipole"][0](position=(0.5, -3, 1)),
            OBJECTS["TriangularMesh"][0](position=(0.5, 4, 1)),
        ]
        sensors = [
            magpy.Sensor(pixel=pixel, position=(0.5, 0.5, 2)),
            magpy.Sensor(
                pixel=pixel[0] if ragged else pixel,
                position=PATH["position"][::-1] * 3,
                orientation=PATH["orientation"],
                handedness="left",
            ),
        ]
        if collection:
            sources = magpy.Collection(*sources, position=(0.1, 0.2, 0.3))
        return GET[field](sources, sensors, **kwargs)

    return run


def _pixel(rng):
    return {"pixel": _uniform(rng, -1, 1, 2, 3, 3)}


for _field in "BHJM":
    _case(f"scene/{_field}", _scene(_field), _pixel)
_case("scene/B/sumup", _scene("B", sumup=True), _pixel)
_case("scene/B/pixel_agg", _scene("B", pixel_agg="mean"), _pixel)
for _agg in ("max", "std"):  # sensors with different pixel shapes
    _case(f"scene/B/pixel_agg_{_agg}", _scene("B", ragged=True, pixel_agg=_agg), _pixel)
_case("scene/B/no_squeeze", _scene("B", squeeze=False, sumup=True), _pixel)
_case("scene/B/collection", _scene("B", collection=True), _pixel)
_case("scene/H/collection", _scene("H", collection=True), _pixel)


TETRAHEDRON_FACES = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)]
SHORT = {key: val[1:4] for key, val in PATH.items()}  # a path of other length
# A second, different source of every class and how it moves. Next to the one
# of OBJECTS on PATH, the sources of a class share one call of their kernel:
# with other geometry, other strength, another number of segments or faces,
# and a path of another length or none.
SECOND = {
    "Cuboid": (
        lambda **kw: magpy.magnet.Cuboid(
            dimension=(1, 0.5, 3), polarization=(1, -0.3, 0.2), **kw
        ),
        SHORT,
    ),
    "Cylinder": (
        lambda **kw: magpy.magnet.Cylinder(
            dimension=(1, 3), magnetization=(1e5, -2e5, 3e5), **kw
        ),
        {"position": (3, 0.5, -1)},
    ),
    "CylinderSegment": (
        lambda **kw: magpy.magnet.CylinderSegment(
            dimension=(0.5, 1, 3, -60, 200), polarization=(1, -0.3, 0.2), **kw
        ),
        SHORT,
    ),
    "CylinderSegment_ring": (
        lambda **kw: magpy.magnet.CylinderSegment(
            dimension=(0.5, 1, 3, 0, 360), polarization=(1, -0.3, 0.2), **kw
        ),
        {"position": (-3, 0.5, 1)},
    ),
    "CylinderSegment_full": (
        lambda **kw: magpy.magnet.CylinderSegment(
            dimension=(0, 1, 3, -180, 180), polarization=(1, -0.3, 0.2), **kw
        ),
        SHORT,
    ),
    "Sphere": (
        lambda **kw: magpy.magnet.Sphere(diameter=1, polarization=(1, -0.3, 0.2), **kw),
        {"position": (0.5, -3, 1)},
    ),
    "Tetrahedron": (
        lambda **kw: magpy.magnet.Tetrahedron(
            vertices=[(0, 0, 0), (2, 0, 0.5), (0.5, 1, 0), (0.3, 0.2, -1.5)],
            polarization=(1, -0.3, 0.2),
            **kw,
        ),
        SHORT,
    ),
    "TriangularMesh": (
        lambda **kw: magpy.magnet.TriangularMesh(
            vertices=TETRAHEDRON,
            faces=TETRAHEDRON_FACES,
            polarization=(1, -0.3, 0.2),
            **kw,
        ),
        SHORT,
    ),
    "Circle": (
        lambda **kw: magpy.current.Circle(diameter=3, current=-2, **kw),
        {"position": (-2, 1, 0.5)},
    ),
    "Polyline": (
        lambda **kw: magpy.current.Polyline(
            current=-2, vertices=[(0, 0, 0), (0.5, 2, 1), (-1, 0.3, 2)], **kw
        ),
        SHORT,
    ),
    "TriangleSheet": (
        lambda **kw: magpy.current.TriangleSheet(
            vertices=[(0, 0, 1), (2, 0, 1), (0, 1, 1.5)],
            faces=[(0, 1, 2)],
            current_densities=[(1, 0.5, 0.25)],
            **kw,
        ),
        SHORT,
    ),
    "TriangleStrip": (
        lambda **kw: magpy.current.TriangleStrip(
            current=-2, vertices=[(0, 0, 1), (2, 0, 1), (0, 1, 1.5)], **kw
        ),
        {"position": (0.5, 4, 1)},
    ),
    "Dipole": (
        lambda **kw: magpy.misc.Dipole(moment=(1, -0.3, 0.2), **kw),
        SHORT,
    ),
    "Triangle": (
        lambda **kw: magpy.misc.Triangle(
            vertices=[(0, 0, 1), (2, 0, 1), (0, 1, 1.5)],
            polarization=(1, -0.3, 0.2),
            **kw,
        ),
        {"position": (0.1, 0.2, 0.3)},
    ),
}


def _groups(field, **kwargs):
    def run(pixel):
        sources = [
            build(**path)
            for name, (second, short) in SECOND.items()
            for build, path in ((OBJECTS[name][0], PATH), (second, short))
        ]
        sensors = [
            magpy.Sensor(pixel=pixel, position=(0.5, 0.5, 2)),
            magpy.Sensor(
                pixel=pixel,
                position=SHORT["position"][::-1] * 3,
                orientation=SHORT["orientation"],
            ),
        ]
        return GET[field](sources, sensors, **kwargs)

    return run


def _pixel_off(rng):  # away from the sources, which lie around the origin
    return {"pixel": _uniform(rng, 2, 4, 2, 3, 3)}


for _field in "BH":
    _case(f"scene/groups/{_field}", _groups(_field), _pixel_off)


# FORCE ----------------------------------------------------------------------


def _force(path=False):
    def run(position):
        sources = [
            magpy.magnet.Cuboid(
                dimension=(1, 1, 1), polarization=POL, position=position
            ),
            magpy.current.Circle(diameter=3, current=1e3, position=(0, 0, -0.5)),
        ]
        kwargs = PATH if path else {"position": (0.2, 0.1, 1.5)}
        cuboid = magpy.magnet.Cuboid(
            dimension=(0.5, 0.4, 0.6), polarization=(0.3, 1, -0.2), **kwargs
        )
        cylinder = magpy.magnet.Cylinder(
            dimension=(0.5, 0.5), polarization=POL, position=(0, 1.5, 0.5)
        )
        circle = magpy.current.Circle(diameter=1, current=3, position=(0.2, 0, -1.5))
        polyline = magpy.current.Polyline(
            current=3, vertices=[(1, 1, 1), (2, 1, 1), (2, 2, 1.5)]
        )
        sphere = magpy.magnet.Sphere(
            diameter=0.5, polarization=(1, 0, 0), position=(1.5, 0, 0)
        )
        dipole = magpy.misc.Dipole(moment=(0, 0, 1), position=(0, -1.5, 0.3))
        return np.stack(
            magpy.getFT(
                sources,
                [cuboid, cylinder, circle, polyline, sphere, dipole],
                meshing={cuboid: 27, cylinder: 30, circle: 12, polyline: 10},
            )
        )

    return run


def _position(rng):
    return {"position": _uniform(rng, -0.1, 0.1, 3)}


_case("force/static", _force(), _position)
_case("force/path", _force(path=True), _position)


def _force_targets(position):
    """The target classes that _force leaves out."""
    sources = [
        magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=POL, position=position),
        magpy.current.Circle(diameter=3, current=1e3, position=(0, 0, -0.5)),
    ]
    targets = [
        SECOND["CylinderSegment"][0](position=(0, 2, 1)),
        SECOND["Tetrahedron"][0](position=(2, 2, 1)),
        SECOND["TriangularMesh"][0](position=(-2, 1, 1)),
        SECOND["TriangleSheet"][0](position=(1, -2, 1)),
        SECOND["TriangleStrip"][0](position=(-2, -2, 1)),
        magpy.Collection(
            SECOND["Sphere"][0](),
            SECOND["Dipole"][0](position=(0.5, 0, 0)),
            position=(0, -2, 1.5),
        ),
    ]
    return np.stack(magpy.getFT(sources, targets, meshing=20))


_case("force/targets", _force_targets, _position)


# TEST -----------------------------------------------------------------------


def _share(name, out, ref):
    """Error of each finite value as a share of its tolerance: above 1 fails."""
    rtol, atol_rel = next(
        (tol for pattern, tol in TOL.items() if fnmatchcase(name, pattern)),
        (RTOL, ATOL_REL),
    )
    finite = np.isfinite(ref)
    size = np.where(finite, np.abs(ref), 0.0)
    axis = 0 if name.startswith(COMPONENTS_FIRST) else -1
    tol = rtol * np.linalg.norm(size, axis=axis, keepdims=True)
    keep = next(
        (axes for pattern, axes in ATOL_AXES.items() if fnmatchcase(name, pattern)),
        (),
    )
    rest = tuple(set(range(size.ndim)) - {axis % size.ndim for axis in keep})
    tol = tol + atol_rel * size.max(axis=rest, keepdims=True, initial=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.where(finite, np.abs(out - ref), 0.0)
        # an exact value uses up nothing, also of a zero tolerance; a NaN
        # where the reference is finite stays NaN, which fails
        return np.where(err == 0, 0.0, err / tol)


@pytest.fixture(scope="module")
def reference():
    with np.load(DATA) as data:
        return dict(data)


@pytest.mark.parametrize("name", CASES)
def test_regression(name, reference):
    """The output for the stored inputs is the stored output: same shape,
    same dtype, same values within rounding, and NaN and inf in the same
    places."""
    run = CASES[name][0]
    prefix = f"{name}|in|"
    inputs = {
        key.removeprefix(prefix): val
        for key, val in reference.items()
        if key.startswith(prefix)
    }
    ref = reference[f"{name}|out"]
    out = np.asarray(run(**inputs))
    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    special = ~np.isfinite(ref)
    np.testing.assert_array_equal(~np.isfinite(out), special)
    np.testing.assert_array_equal(out[special], ref[special])
    share = _share(name, out, ref)
    worst = np.unravel_index(share.argmax(), share.shape) if share.size else ()
    assert share.max(initial=0.0) <= 1, (
        f"at {worst}: {out[worst]!r} instead of {ref[worst]!r}, "
        f"{share[worst]:.3g} times the tolerance"
    )


@pytest.fixture(scope="module", params=["jax", "torch", "array_api_strict"])
def xp(request):
    """The array namespace of another library, computing in float64."""
    if request.param == "jax":
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        return jax.numpy
    return pytest.importorskip(request.param)


@pytest.mark.parametrize("name", [n for n in CASES if n.startswith(("core/", "func/"))])
def test_regression_backend(name, xp, reference):
    """The core functions and the functional interface give the NumPy
    reference when the stored inputs are arrays of another library."""
    inputs = {
        # the orientation stays a quaternion that becomes a SciPy Rotation
        key: val if key.endswith("|orientations") else xp.asarray(val)
        for key, val in reference.items()
        if key.startswith(f"{name}|in|")
    }
    if "cylinder_segment" in name:  # needs SciPy's incomplete elliptic integrals
        with pytest.raises(NotImplementedError, match="only implemented for NumPy"):
            test_regression(name, reference | inputs)
    else:
        test_regression(name, reference | inputs)


def test_regression_covers_public_api():
    """Every core function, functional interface function and source class
    has cases, so that a new one cannot be added without a reference."""
    classes = [
        name
        for module in (magpy.magnet, magpy.current, magpy.misc)
        for name in module.__all__
        if name != "CustomSource"  # has no field of its own
    ]
    expected = (
        [f"core/{name}" for name in magpy.core.__all__]
        + [f"func/{name}/" for name in magpy.func.__all__]
        + [f"obj/{name}/" for name in classes]
    )
    missing = [exp for exp in expected if not any(c.startswith(exp) for c in CASES)]
    assert not missing


def test_regression_reference_matches_cases(reference):
    """The reference file holds exactly the registered cases: no case without
    a reference and no stale reference."""
    assert {key.split("|")[0] for key in reference} == set(CASES)


# GENERATOR ------------------------------------------------------------------


def _noise(name, run, inputs, ref, rng):
    """Largest share of the tolerance that rounding uses up, when every input
    is moved by up to 4 ulp: a stand-in for the reordered operations of a
    rewrite. The tolerances are set so that this stays below 0.02."""
    eps = 4 * np.finfo(float).eps
    share = 0.0
    for _ in range(5):
        nudged = {
            key: val * (1 + eps * rng.choice((-1.0, 1.0), val.shape))
            for key, val in inputs.items()
        }
        out = np.asarray(run(**nudged))
        share = max(share, _share(name, out, ref).max(initial=0.0))
    return share


def _reference():
    """The content of the reference file and the noise of each case."""
    warnings.simplefilter("error")  # as pytest does: no case may warn
    data, noise = {}, {}
    for name, (run, make, nudge) in CASES.items():
        rng = np.random.default_rng(list(name.encode()))
        stored = {key: np.array(val) for key, val in make(rng).items()}
        # run on copies: some kernels change their input in place
        ref = np.asarray(run(**{key: val.copy() for key, val in stored.items()}))
        data[f"{name}|out"] = ref
        data |= {f"{name}|in|{key}": val for key, val in stored.items()}
        if nudge and stored:
            noise[name] = _noise(name, run, stored, ref, rng)
    return data, noise


def _generate():
    data, noise = _reference()
    np.savez_compressed(DATA, **data)

    size = DATA.stat().st_size / 1e6
    sys.stdout.write(f"{len(CASES)} cases written to {DATA} ({size:.2f} MB)\n")
    sys.stdout.write("share of the tolerance used by rounding noise (largest):\n")
    for name, share in sorted(noise.items(), key=lambda item: -item[1])[:25]:
        flag = "  <-- above 0.02: loosen in TOL" if share > 0.02 else ""
        sys.stdout.write(f"  {share:9.2e}  {name}{flag}\n")


if __name__ == "__main__":
    _generate()

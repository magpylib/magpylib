"""What every source returns exactly on its faces, edges, vertices and wires.

The rules (magpylib#769, magpylib#1073):

- face, a surface the field jumps across (magnet boundary, charge or current
  sheet): the average of both sides, for B, H, J and M. J and M are half.
- singular point, where the face's normal turns, where a sheet's density jumps,
  on a wire or at a dipole: B and H are NaN; J and M are the share of the
  surrounding sphere inside the magnet (1/4 on a right-angle edge, 1/8 on a cube
  corner). A source with zero excitation returns 0 there.
- regular point, where nothing jumps (a face's plane beyond the face, a 360°
  segment's seam, a diagonal between coplanar triangles is a face): the field
  is continuous and the formula finite.
- "on" means within TOL = 100 eps max(size, |observer|), per object.

Cases magpylib does not meet yet are expected failures (XFAIL), each naming
the defect; a fix turns its xfail into a pass, which strict xfail reports.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import magpylib as magpy

J = np.array((0.6, 0.0, 0.8))
EPS = np.finfo(float).eps
FIELDS = "BHJM"


def tol(size, p):
    """The proposed on-feature tolerance."""
    return 100 * EPS * max(size, np.linalg.norm(p))


def cube_mesh(polarization):
    """A 2 m cube as a TriangularMesh; face x = 1 split along y = z."""
    v = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    idx = {tuple(p): i for i, p in enumerate(v.astype(int))}

    def quad(a, b, c, d):
        a, b, c, d = (idx[q] for q in (a, b, c, d))
        return [(a, b, c), (a, c, d)]

    faces = [
        *quad((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)),
        *quad((-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)),
        *quad((-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1)),
        *quad((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)),
        *quad((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)),
        *quad((-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1)),
    ]
    return magpy.magnet.TriangularMesh(
        vertices=v, faces=faces, polarization=polarization
    )


TET = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
TRI = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
STRIP = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0)]

# class: (factory taking the excitation scale, size)
SOURCES = {
    "Cuboid": (
        lambda k=1: magpy.magnet.Cuboid(dimension=(2, 2, 2), polarization=k * J),
        2,
    ),
    "Cylinder": (
        lambda k=1: magpy.magnet.Cylinder(dimension=(2, 2), polarization=k * J),
        2,
    ),
    "CylinderSegment": (
        lambda k=1: magpy.magnet.CylinderSegment(
            dimension=(0.5, 1, 2, 0, 90), polarization=k * J
        ),
        2,
    ),
    "CylinderSegment-r0-90": (
        lambda k=1: magpy.magnet.CylinderSegment(
            dimension=(0, 1, 2, 0, 90), polarization=k * J
        ),
        2,
    ),
    "CylinderSegment-r0-180": (
        lambda k=1: magpy.magnet.CylinderSegment(
            dimension=(0, 1, 2, 0, 180), polarization=k * J
        ),
        2,
    ),
    "CylinderSegment-360": (
        lambda k=1: magpy.magnet.CylinderSegment(
            dimension=(0.5, 1, 2, 0, 360), polarization=k * J
        ),
        2,
    ),
    "Sphere": (lambda k=1: magpy.magnet.Sphere(diameter=2, polarization=k * J), 2),
    "Tetrahedron": (
        lambda k=1: magpy.magnet.Tetrahedron(vertices=TET, polarization=k * J),
        1,
    ),
    "TriangularMesh": (lambda k=1: cube_mesh(k * J), 2),
    "Triangle": (lambda k=1: magpy.misc.Triangle(vertices=TRI, polarization=k * J), 1),
    "Circle": (lambda k=1: magpy.current.Circle(diameter=2, current=k), 2),
    "Polyline": (
        lambda k=1: magpy.current.Polyline(
            vertices=[(0, 0, 0), (1, 0, 0), (1, 1, 0)], current=k
        ),
        1,
    ),
    "TriangleSheet": (
        lambda k=1: magpy.current.TriangleSheet(
            vertices=TRI, faces=[(0, 1, 2)], current_densities=[(k, 0, 0)]
        ),
        1,
    ),
    "TriangleSheet-2": (
        lambda k=1: magpy.current.TriangleSheet(
            vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
            faces=[(0, 1, 2), (1, 3, 2)],
            current_densities=[(k, 0, 0), (k, 0, 0)],
        ),
        1,
    ),
    "TriangleStrip": (
        lambda k=1: magpy.current.TriangleStrip(vertices=STRIP, current=k),
        1,
    ),
    "Dipole": (lambda k=1: magpy.misc.Dipole(moment=(0, 0, k)), 0),
}
MAGNETS = {"Cuboid", "Cylinder", "Sphere", "Tetrahedron", "TriangularMesh"} | {
    k for k in SOURCES if k.startswith("CylinderSegment")
}

S = 1 / np.sqrt(2)
# (source, label, point, outward direction, J share); faces
FACES = [
    ("Cuboid", "x-face", (1, 0.3, 0.2), (1, 0, 0), 0.5),
    ("Cuboid", "y-face", (0.3, 1, 0.2), (0, 1, 0), 0.5),
    ("Cuboid", "z-face", (0.3, 0.2, 1), (0, 0, 1), 0.5),
    ("Cylinder", "side", (1, 0, 0.3), (1, 0, 0), 0.5),
    ("Cylinder", "end", (0.3, 0.2, 1), (0, 0, 1), 0.5),
    ("Cylinder", "end, on axis", (0, 0, 1), (0, 0, 1), 0.5),
    (
        "CylinderSegment",
        "outer side",
        (np.cos(0.6), np.sin(0.6), 0.3),
        (np.cos(0.6), np.sin(0.6), 0),
        0.5,
    ),
    (
        "CylinderSegment",
        "inner side",
        (0.5 * np.cos(0.6), 0.5 * np.sin(0.6), 0.3),
        (-np.cos(0.6), -np.sin(0.6), 0),
        0.5,
    ),
    ("CylinderSegment", "phi face", (0.75, 0, 0.3), (0, -1, 0), 0.5),
    ("CylinderSegment", "end", (0.75 * S, 0.75 * S, 1), (0, 0, 1), 0.5),
    ("CylinderSegment-r0-180", "axis in a flat face", (0, 0, 0.3), (0, -1, 0), 0.5),
    ("Sphere", "surface", (0, 0.6, 0.8), (0, 0.6, 0.8), 0.5),
    ("Tetrahedron", "face", (0.2, 0.2, 0), (0, 0, -1), 0.5),
    ("Tetrahedron", "slanted face", (0.3, 0.3, 0.4), (1, 1, 1), 0.5),
    ("TriangularMesh", "face", (1, 0.3, -0.2), (1, 0, 0), 0.5),
    ("TriangularMesh", "diagonal in a flat face", (1, 0.3, 0.3), (1, 0, 0), 0.5),
    ("Triangle", "on the sheet", (0.2, 0.2, 0), (0, 0, 1), None),
    ("TriangleSheet", "on the sheet", (0.2, 0.2, 0), (0, 0, 1), None),
    ("TriangleSheet-2", "inner edge, same K", (0.5, 0.5, 0), (0, 0, 1), None),
    ("TriangleStrip", "on the strip", (0.3, 0.3, 0), (0, 0, 1), None),
]
# singular points: (source, label, point, J share)
SINGULAR = [
    ("Cuboid", "edge", (1, 1, 0.2), 0.25),
    ("Cuboid", "vertex", (1, 1, 1), 0.125),
    ("Cylinder", "rim", (1, 0, 1), 0.25),
    ("CylinderSegment", "edge phi/outer", (1, 0, 0.3), 0.25),
    ("CylinderSegment", "corner", (1, 0, 1), 0.125),
    ("CylinderSegment-r0-90", "axis", (0, 0, 0.3), 0.25),
    ("Tetrahedron", "edge", (0.4, 0, 0), 0.25),
    ("Tetrahedron", "vertex", (0, 0, 0), 0.125),
    ("TriangularMesh", "edge", (1, 1, 0.2), 0.25),
    ("TriangularMesh", "vertex", (1, 1, 1), 0.125),
    ("Triangle", "edge", (0.4, 0, 0), None),
    ("Triangle", "vertex", (0, 0, 0), None),
    ("Circle", "on the wire", (1, 0, 0), None),
    ("Polyline", "on a segment", (0.5, 0, 0), None),
    ("Polyline", "corner", (1, 0, 0), None),
    ("Polyline", "end", (0, 0, 0), None),
    ("TriangleSheet", "edge", (0.4, 0, 0), None),
    ("TriangleSheet", "vertex", (0, 0, 0), None),
    ("TriangleStrip", "side edge", (0.5, 0, 0), None),
    ("TriangleStrip", "inner edge", (0.5, 0.5, 0), None),
    ("Dipole", "position", (0, 0, 0), None),
]
# regular points: (source, label, point)
REGULAR = [
    ("Cuboid", "face plane, beyond the face", (1, 1.5, 0.2)),
    ("Cuboid", "edge line, beyond the edge", (1, 1, 3)),
    ("Cylinder", "axis, inside", (0, 0, 0.3)),
    ("Cylinder", "end plane, beyond the rim", (2, 0, 1)),
    ("CylinderSegment-360", "seam", (0.75, 0, 0.3)),
    ("Sphere", "centre", (0, 0, 0)),
    ("Triangle", "plane, beyond the edge", (2, 2, 0)),
    ("Circle", "centre", (0, 0, 0)),
    ("Circle", "axis", (0, 0, 0.5)),
    ("Polyline", "line, beyond the end", (-1, 0, 0)),
    ("TriangleSheet", "plane, beyond the edge", (2, 2, 0)),
]

# known defects of today's kernels, by case id
XFAIL = {
    "Cuboid x-face": "outside value with inside J (#1073)",
    "Cuboid y-face": "inside value (#1073)",
    "Cuboid z-face": "inside value (#1073)",
    "Cylinder side": "neither side (#1073)",
    "Cylinder end": "inside value (#1073)",
    "Cylinder end, on axis": "inside value (#1073)",
    "CylinderSegment outer side": "B = H = 0 on every surface (#1073)",
    "CylinderSegment inner side": "B = H = 0 on every surface (#1073)",
    "CylinderSegment phi face": "B = H = 0 on every surface (#1073)",
    "CylinderSegment end": "B = H = 0 on every surface (#1073)",
    "CylinderSegment-r0-180 axis in a flat face": "B = H = 0 on the axis (#1073)",
    "Sphere surface": "inside value (#1073)",
    "Tetrahedron face": "H averages, J counted whole (#1073)",
    "Tetrahedron slanted face": "H averages, J counted whole (#1073)",
    "TriangularMesh face": "H averages, J counted whole (#1073)",
    "TriangularMesh diagonal in a flat face": "wrong near a diagonal between coplanar triangles",
    "TriangleSheet on the sheet": "0, normal component lost (#1073)",
}
XFAIL_SINGULAR = "finite value where the field is infinite (#1073)"
XFAIL_NEAR = {
    "Cuboid edge": "NaN band near edges (#1058, #1066)",
    "CylinderSegment end": "NaN or ValueError within 3e-12..3e-9 m of an end face",
    "CylinderSegment corner": "NaN or ValueError near an end face",
    "Tetrahedron edge": "snaps to the edge value where |r + b/l| < 1e-12 m",
    "TriangularMesh edge": "snaps to the edge value near an edge",
    "TriangularMesh diagonal in a flat face": "wrong up to ~3 µm from a diagonal",
    "TriangleSheet edge": "0 within 1e-10 m, then NaN (num_tol)",
    "TriangleSheet vertex": "0 within 1e-10 m, then NaN (num_tol)",
    "TriangleStrip side edge": "0 within 1e-10 m, then NaN (num_tol)",
    "TriangleStrip inner edge": "0 within 1e-10 m, then NaN (num_tol)",
    "Triangle edge": "snaps to the edge value where |r + b/l| < 1e-12 m",
    "Tetrahedron face": "face value up to ~1e-10 of the size off the face",
    "Tetrahedron slanted face": "face value up to ~1e-10 of the size off the face",
    "TriangularMesh face": "face value up to ~1e-10 of the size off the face",
    "Triangle on the sheet": "face value up to ~1e-10 of the size off the sheet",
    "CylinderSegment-r0-180 axis in a flat face": "divide by zero in arctanh near the axis",
    "CylinderSegment-r0-90 axis": "divide by zero in arctanh near the axis",
    "CylinderSegment edge phi/outer": "divide by zero in arctanh near the edge",
}
XFAIL_ZERO = {
    "TriangleStrip side edge": "zero current divides by zero",
    "TriangleStrip inner edge": "zero current divides by zero",
}


def params(cases, xfail, default_reason=None):
    out = []
    for case in cases:
        cid = f"{case[0]} {case[1]}"
        reason = xfail.get(cid, default_reason)
        marks = [pytest.mark.xfail(reason=reason, strict=True)] if reason else []
        out.append(pytest.param(*case, id=cid, marks=marks))
    return out


def get(src, field, points):
    return np.asarray(
        getattr(magpy, f"get{field}")(src, np.atleast_2d(points)), float
    ).reshape(-1, 3)


def one_sided(src, field, p, n, size):
    """Limit from the side n points to: Richardson from two offsets."""
    d = 1e-5 * size
    a, b = get(src, field, [p + d * n, p + 2 * d * n])
    return 2 * a - b


def fields_of(name):
    return FIELDS if name in MAGNETS else "BH"


@pytest.mark.parametrize(("name", "label", "p", "n", "share"), params(FACES, XFAIL))
def test_face_returns_average_of_both_sides(name, label, p, n, share):
    make, size = SOURCES[name]
    src = make()
    p, n = np.array(p, float), np.array(n, float) / np.linalg.norm(n)
    for field in fields_of(name):
        on = get(src, field, p)[0]
        mean = (
            one_sided(src, field, p, n, size) + one_sided(src, field, p, -n, size)
        ) / 2
        scale = 1 / magpy.mu_0 if field in "HM" else 1
        np.testing.assert_allclose(
            on, mean, atol=1e-7 * scale, err_msg=f"{field} ({label})"
        )
    if share is not None:
        np.testing.assert_allclose(get(src, "J", p)[0], share * J, atol=1e-12)


@pytest.mark.parametrize(
    ("name", "label", "p", "share"), params(SINGULAR, {}, XFAIL_SINGULAR)
)
def test_singular_point_returns_nan(name, label, p, share):
    src = SOURCES[name][0]()
    for field in "BH":
        assert np.isnan(get(src, field, p)).all(), f"{field} at the {label}"
    if share is not None:
        np.testing.assert_allclose(get(src, "J", p)[0], share * J, atol=1e-12)


@pytest.mark.parametrize(("name", "label", "p"), params(REGULAR, {}))
def test_regular_point_is_continuous(name, label, p):
    make, size = SOURCES[name]
    src = make()
    p = np.array(p, float)
    n = np.array((0.36, 0.48, 0.8))
    for field in fields_of(name):
        on = get(src, field, p)[0]
        assert np.isfinite(on).all(), f"{field} at the {label}"
        scale = 1 / magpy.mu_0 if field in "HM" else 1
        for side in (n, -n):
            np.testing.assert_allclose(
                on,
                one_sided(src, field, p, side, size),
                atol=1e-7 * scale,
                err_msg=f"{field} ({label})",
            )


NEAR = [(c[0], c[1], c[2], c[3]) for c in FACES] + [
    (c[0], c[1], c[2], None) for c in SINGULAR if c[0] != "Dipole"
]


@pytest.mark.parametrize(("name", "label", "p", "n"), params(NEAR, XFAIL_NEAR))
def test_near_a_feature_the_formula_is_finite(name, label, p, n):
    """Off the tolerance the full formula applies: finite, no warning; on a face
    the one-sided value, at a singular point a field that keeps growing."""
    make, size = SOURCES[name]
    src = make()
    p = np.array(p, float)
    directions = (
        [np.array(n, float) / np.linalg.norm(n)]
        if n is not None
        else [
            np.array(v, float) / np.linalg.norm(v)
            for v in ((1, 2, 3), (-3, 1, 2), (2, -3, -1))
        ]
    )
    t = tol(size, p)
    for u in directions:
        for side in (u, -u):
            pts = (
                p
                + np.array([1e3 * t, 1e6 * t, 1e-6 * size, 1e-3 * size])[:, None] * side
            )
            for field in "BH":
                values = get(src, field, pts)
                assert np.isfinite(values).all(), f"{field} near the {label}"
                if n is None:
                    grow = np.linalg.norm(values[0]) - np.linalg.norm(values[1])
                    assert grow > 1e-3 * np.linalg.norm(values[1]), (
                        f"{field} stops growing near the {label}"
                    )
                else:
                    limit = one_sided(src, field, p, side, size)
                    scale = 1 / magpy.mu_0 if field == "H" else 1
                    np.testing.assert_allclose(
                        values[:3],
                        np.broadcast_to(limit, (3, 3)),
                        atol=1e-5 * scale,
                        err_msg=f"{field} near the {label}",
                    )


CUTS = {
    "2 cells, shared x-face": (
        [(-0.5, 0, 0), (0.5, 0, 0)],
        (2, 1, 1),
        (0, 0.2, 0.1),
        True,
    ),
    "2 cells, shared z-face": (
        [(0, 0, -0.5), (0, 0, 0.5)],
        (1, 1, 2),
        (0.2, 0.1, 0),
        True,
    ),
    "4 cells, shared edge": (
        [(x, y, 0) for x in (-0.5, 0.5) for y in (-0.5, 0.5)],
        (2, 2, 1),
        (0, 0, 0.1),
        False,
    ),
    "8 cells, shared corner": (
        [(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)],
        (2, 2, 2),
        (0, 0, 0),
        False,
    ),
}


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            k,
            marks=pytest.mark.xfail(
                reason="J counted once per cell (#1073)", strict=True
            ),
        )
        for k in CUTS
    ],
)
def test_touching_cells_add_up_to_the_whole_magnet(case):
    """J always adds up to the whole magnet's; on a shared face B and H too."""
    centres, whole_dim, p, face = CUTS[case]
    cut = magpy.Collection(
        *(
            magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=J, position=c)
            for c in centres
        )
    )
    whole = magpy.magnet.Cuboid(dimension=whole_dim, polarization=J)
    np.testing.assert_allclose(cut.getJ(p), whole.getJ(p), atol=1e-12)
    if face:
        np.testing.assert_allclose(cut.getB(p), whole.getB(p), atol=1e-9)
        np.testing.assert_allclose(cut.getH(p), whole.getH(p), atol=1e-9 / magpy.mu_0)


@pytest.mark.parametrize(
    "seed",
    [
        pytest.param(
            s,
            marks=pytest.mark.xfail(
                reason="face value (#1073) and a tolerance relative to the half side only",
                strict=True,
            ),
        )
        for s in range(3)
    ],
)
def test_rounded_points_count_as_on_the_face(seed):
    """A point meant on a face, placed through a turned parent far from the
    origin, gets the face value."""
    rng = np.random.default_rng(seed)
    rot = R.from_quat(rng.normal(size=4))
    c = magpy.magnet.Cuboid(
        dimension=(2, 2, 2),
        polarization=J,
        position=rng.normal(size=3) * 1e3,
        orientation=rot,
    )
    q = np.array((1, 0.3, 0.2))
    p_local = c.orientation.apply(q) + c.position
    parent = magpy.Collection(c)
    turn, anchor = R.from_quat(rng.normal(size=4)), rng.normal(size=3) * 1e3
    parent.rotate(turn, anchor=anchor)
    p = turn.apply(p_local - anchor) + anchor
    n = c.orientation.apply((1, 0, 0))
    d = 1e-5
    lim = [
        2 * get(c, "B", p + d * s)[0] - get(c, "B", p + 2 * d * s)[0] for s in (n, -n)
    ]
    np.testing.assert_allclose(c.getB(p), (lim[0] + lim[1]) / 2, atol=1e-6)
    np.testing.assert_allclose(c.getJ(p), J / 2, atol=1e-12)


@pytest.mark.parametrize(
    ("name", "label", "p"), params([c[:3] for c in SINGULAR], XFAIL_ZERO)
)
def test_zero_excitation_returns_zero(name, label, p):
    src = SOURCES[name][0](0)
    for field in fields_of(name):
        np.testing.assert_array_equal(
            get(src, field, p), 0, err_msg=f"{field} at the {label}"
        )

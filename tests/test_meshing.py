"""What an integer ``meshing`` means for each target class, as cells (#1009)."""

import numpy as np
import pytest

import magpylib as magpy
from magpylib import getFT

POL = (0.1, 0.2, 0.3)
TETRA = [(0, 0, -0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
BOX = [(x, y, z) for x in (-1.5, 1.5) for y in (-1, 1) for z in (-0.5, 0.5)]


def magnets():
    """One magnet of each meshed class, at the origin and unrotated."""
    return [
        magpy.magnet.Cuboid(dimension=(3, 2, 1), polarization=POL),
        magpy.magnet.Cylinder(dimension=(2, 1), polarization=POL),
        magpy.magnet.CylinderSegment(dimension=(1, 2, 1, 45, 123), polarization=POL),
        magpy.magnet.Tetrahedron(vertices=TETRA, polarization=POL),
        magpy.magnet.TriangularMesh.from_ConvexHull(points=BOX, polarization=POL),
    ]


def class_name(obj):
    """Test id of a parametrized object."""
    return type(obj).__name__


@pytest.mark.parametrize("meshing", [1, 50])
@pytest.mark.parametrize("magnet", magnets(), ids=class_name)
def test_cells_fill_the_magnet(magnet, meshing):
    """The cells add up to the magnet, and the force mesh is their centroids and
    volumes."""
    mesh = magnet._generate_mesh(magnet._validate_meshing(meshing))
    cells = mesh["cells"]
    volumes = cells["volumes"]
    np.testing.assert_allclose(volumes.sum(), magnet.volume)
    np.testing.assert_allclose(
        np.average(cells["centroids"], weights=volumes, axis=0),
        magnet.centroid,
        atol=1e-9,
    )
    np.testing.assert_allclose(mesh["pts"], cells["centroids"])
    np.testing.assert_allclose(
        mesh["moments"], volumes[:, np.newaxis] * magnet.magnetization
    )


@pytest.mark.parametrize(
    ("magnet", "meshing", "n_cells", "cell_type"),
    [
        (magpy.magnet.Cuboid(dimension=(1, 1, 1)), 2000, 2197, "cuboid"),
        (magpy.magnet.Cuboid(dimension=(3, 2, 1)), 100, 120, "cuboid"),
        (magpy.magnet.Cuboid(dimension=(3, 2, 1)), (4, 3, 2), 24, "cuboid"),
        (magpy.magnet.Cylinder(dimension=(1, 1)), 200, 168, "cylinder_segment"),
        (
            magpy.magnet.CylinderSegment(dimension=(1, 2, 1, 45, 123)),
            (4, 2, 3),
            24,
            "cylinder_segment",
        ),
        (
            magpy.magnet.CylinderSegment(dimension=(0.95, 1, 0.5, 0, 360)),
            1,
            12,
            "cylinder_segment",
        ),
        (magpy.magnet.Tetrahedron(vertices=TETRA), 20, 27, "tetrahedron"),
        (magpy.magnet.Tetrahedron(vertices=TETRA), 84, 64, "tetrahedron"),
        (magpy.magnet.Tetrahedron(vertices=TETRA), 286, 343, "tetrahedron"),
        (magpy.magnet.TriangularMesh.from_ConvexHull(points=BOX), 1, 1, "cuboid"),
    ],
    ids=[
        "Cuboid-2000",
        "Block-100",
        "Cuboid-triple",
        "Cylinder-200",
        "Segment-triple",
        "ThinTube-1",
        "Tetrahedron-20",
        "Tetrahedron-84",
        "Tetrahedron-286",
        "TriangularMesh-1",
    ],
)
def test_meshing_cell_counts(magnet, meshing, n_cells, cell_type):
    """An integer meshing is a target number of cells, split by one rule per class.
    Cuboids get near-cubic cells, as up to 5.2: 13 x 13 x 13 and 8 x 5 x 3 here."""
    magnet.polarization = POL
    cells = magnet._generate_mesh(magnet._validate_meshing(meshing))["cells"]
    assert cells["type"] == cell_type
    assert len(cells["volumes"]) == n_cells


def test_cylinder_inner_ring_stays_a_cylinder():
    """A full cylinder with at least 3 rings keeps one cell per layer in the middle,
    stored as a segment with r1 = 0 over 360 degrees and centered on the axis."""
    cylinder = magpy.magnet.Cylinder(dimension=(1, 1), polarization=POL)
    cells = cylinder._generate_mesh(200)["cells"]
    dims = cells["dimensions"]
    inner = (dims[:, 0] == 0) & (dims[:, 4] - dims[:, 3] == 360)
    assert inner.sum() == 6
    np.testing.assert_allclose(cells["centroids"][inner, :2], 0, atol=1e-15)


def test_tetrahedron_force_converges():
    """More cells give a better force, monotonically (#1003)."""
    source = magpy.magnet.Cuboid(
        dimension=(0.01,) * 3, polarization=(0, 0, 1), position=(0, 0, -0.0055)
    )
    vertices = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4]]) * 1e-3
    tetra = magpy.magnet.Tetrahedron(
        vertices=vertices, polarization=(0, 0, 1), position=(-1e-3, -1e-3, 0.5e-3)
    )
    forces = [
        getFT(source, tetra, meshing=meshing)[0][2]
        for meshing in (1, 8, 27, 64, 343, 8000)
    ]
    errors = np.abs(np.array(forces[:-1]) / forces[-1] - 1)
    assert np.all(np.diff(errors) < 0)
    assert errors[-1] < 1e-4


@pytest.mark.parametrize(
    ("target", "meshing"),
    [
        *[
            (magpy.magnet.Cuboid(dimension=(1, 1, 1)), bad)
            for bad in [
                0,
                -1,
                2.0,
                True,
                "bad",
                (1, 2),
                (-1, 2, 2),
                (0, 0, 0),
                (1, 2.0, 3),
                np.full((3, 3), 2),
            ]
        ],
        (magpy.magnet.Cylinder(dimension=(1, 1)), (2, 0, 2)),
        (magpy.magnet.CylinderSegment(dimension=(1, 2, 1, 0, 90)), (2, 2)),
        (magpy.current.Circle(diameter=1, current=1), 3),
        (magpy.current.Polyline(vertices=[(0, 0, 0), (1, 0, 0)], current=1), True),
        (
            magpy.current.TriangleStrip(
                vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)], current=1
            ),
            1,
        ),
        (
            magpy.current.TriangleSheet(
                vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
                faces=[(0, 1, 2), (1, 3, 2)],
                current_densities=[(1, 0, 0), (1, 0, 0)],
            ),
            1,
        ),
    ],
)
def test_meshing_rejects_invalid_values(target, meshing):
    """A meshing is a positive integer, not a bool, above the class minimum; a
    Cuboid, Cylinder or CylinderSegment also takes a triple of them (#1004)."""
    with pytest.raises(ValueError, match="Input meshing of"):
        target._validate_meshing(meshing)


@pytest.mark.parametrize(
    ("target", "meshing", "expected"),
    [
        (magpy.magnet.Cuboid(dimension=(1, 1, 1)), np.int64(8), 8),
        (magpy.magnet.Cuboid(dimension=(1, 1, 1)), [2, 2, 2], (2, 2, 2)),
        (magpy.magnet.Cuboid(dimension=(1, 1, 1)), np.array([2, 3, 4]), (2, 3, 4)),
        (magpy.magnet.Cylinder(dimension=(1, 1)), [8, 3, 2], (8, 3, 2)),
        (magpy.current.Circle(diameter=1, current=1), np.int64(4), 4),
    ],
)
def test_meshing_is_normalised_to_ints(target, meshing, expected):
    """Valid meshing inputs become an int or a tuple of 3 ints."""
    value = target._validate_meshing(meshing)
    assert value == expected
    assert type(value) is type(expected)


def test_return_mesh_is_deprecated():
    """getFT(..., return_mesh=True) still returns the mesh, with a warning."""
    src = magpy.misc.Dipole(moment=(0, 0, 1), position=(0, 0, 2))
    cube = magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=POL)
    with pytest.warns(DeprecationWarning, match="return_mesh"):
        (mesh,) = getFT(src, [cube], meshing=8, return_mesh=True)
    assert mesh["pts"].shape == (8, 3)


def test_polyline_meshing_below_segments_warns():
    """A Polyline with fewer mesh points than segments warns and uses one per
    segment."""
    cube = magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=POL)
    poly = magpy.current.Polyline(vertices=[(0, 0, 0), (1, 1, 1), (2, 2, 2)], current=1)
    with pytest.warns(UserWarning, match="Setting one point per segment"):
        getFT(cube, poly, meshing=1)


def test_meshing_is_an_argument_of_getFT():
    """``meshing`` moved from the targets to ``getFT`` (#1017): an integer for all
    targets, a dict with one value per target, a Collection key for its members;
    the old keyword and attribute point to the call."""
    src = magpy.misc.Dipole(moment=(0, 0, 1e3), position=(0, 0, 2))
    cube = magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=POL)
    loop = magpy.current.Circle(diameter=1, current=1, position=(0, 0, -1))
    sphere = magpy.magnet.Sphere(diameter=1, polarization=POL, position=(0, 0, -2))
    coll = magpy.Collection(cube, loop)

    with pytest.raises(TypeError, match="argument of getFT"):
        magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=POL, meshing=8)
    assert not hasattr(cube, "meshing")
    with pytest.raises(TypeError, match=r"unexpected keyword argument 'meshing'$"):
        magpy.Sensor(meshing=3)  # not a target: the ordinary error, no pointer
    assert "meshing" not in cube.describe(return_string=True)

    F_int, _ = getFT(src, [cube, loop], meshing=8)
    F_dict, _ = getFT(src, [cube, loop], meshing={cube: (2, 2, 2), loop: 8})
    F_coll, _ = getFT(src, coll, meshing={coll: 8})
    F_mixed, _ = getFT(src, coll, meshing={coll: 8, cube: (2, 2, 2)})
    F_both, _ = getFT(src, [cube, coll], meshing={coll: 8})  # the key covers both
    cyl = magpy.magnet.Cylinder(dimension=(1, 1), polarization=POL)
    F_tri, _ = getFT(src, [cube, cyl], meshing=(4, 2, 2))  # every target cuts it
    F_tri_dict, _ = getFT(src, [cube, cyl], meshing={cube: (4, 2, 2), cyl: (4, 2, 2)})
    np.testing.assert_allclose(F_int, F_dict)
    np.testing.assert_allclose(F_coll, F_int.sum(axis=0))
    np.testing.assert_allclose(F_mixed, F_dict.sum(axis=0))
    np.testing.assert_allclose(F_both, [F_int[0], F_int.sum(axis=0)])
    np.testing.assert_allclose(F_tri, F_tri_dict)
    assert getFT(src, sphere)[0].shape == (3,)  # exact targets need no meshing
    assert getFT(src, [sphere, cube], meshing={cube: 1})[0].shape == (2, 3)

    with pytest.raises(ValueError, match="Missing meshing input for target"):
        getFT(src, cube)
    with pytest.raises(ValueError, match="Missing meshing input for target"):
        getFT(src, [cube, loop], meshing={cube: 8})
    with pytest.raises(ValueError, match="keys that are not targets"):
        getFT(src, cube, meshing={cube: 8, loop: 8})
    other = cube.copy()
    with pytest.raises(ValueError, match="keys that are not targets"):
        getFT(
            src,
            magpy.Collection(other, magpy.Sensor()),
            meshing={other: 8, magpy.Sensor(): 1},
        )
    with pytest.raises(ValueError, match="needs no meshing"):
        getFT(src, [sphere, cube], meshing={sphere: 5, cube: 1})
    with pytest.raises(ValueError, match="only cuts a Cuboid, Cylinder"):
        getFT(src, coll, meshing={coll: (2, 2, 2)})
    with pytest.raises(ValueError, match="only cuts a Cuboid, Cylinder"):
        getFT(src, [cube, loop], meshing=(2, 2, 2))
    with pytest.raises(ValueError, match="Input meshing of"):
        getFT(src, loop, meshing=True)
    with pytest.raises(ValueError, match="Input meshing of"):
        getFT(src, loop, meshing={loop: 3})  # a Circle needs at least 4


def test_strip_meshing_minimum_counts_faces_not_path_steps():
    """A TriangleStrip with vertices varying along a path still needs at least one
    mesh triangle per face."""
    verts = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 2, 0)])
    strip = magpy.current.TriangleStrip(
        vertices=np.array([verts, verts + 0.01]), current=[1, 1]
    )
    assert strip._validate_meshing(3) == 3
    with pytest.raises(ValueError, match="number of faces \\(3\\)"):
        strip._validate_meshing(2)

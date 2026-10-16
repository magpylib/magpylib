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
    magnet.meshing = meshing
    mesh = magnet._generate_mesh()
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
        (magpy.magnet.Cylinder(dimension=(1, 1)), 200, 196, "cylinder_segment"),
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
    magnet.meshing = meshing
    cells = magnet._generate_mesh()["cells"]
    assert cells["type"] == cell_type
    assert len(cells["volumes"]) == n_cells


def test_cylinder_inner_ring_stays_a_cylinder():
    """A full cylinder with at least 3 rings keeps one cell per layer in the middle,
    stored as a segment with r1 = 0 over 360 degrees and centered on the axis."""
    cylinder = magpy.magnet.Cylinder(dimension=(1, 1), polarization=POL, meshing=200)
    cells = cylinder._generate_mesh()["cells"]
    dims = cells["dimensions"]
    inner = (dims[:, 0] == 0) & (dims[:, 4] - dims[:, 3] == 360)
    assert inner.sum() == 7
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
    forces = []
    for meshing in (1, 8, 27, 64, 343, 8000):
        tetra.meshing = meshing
        forces.append(getFT(source, tetra)[0][2])
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
        (magpy.magnet.Cylinder(dimension=(1, 1)), (2, 2, 2)),
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
    """A meshing is a positive integer, not a bool, above the class minimum; a Cuboid
    also takes a triple of them (#1004)."""
    with pytest.raises(ValueError, match="Input meshing of"):
        target.meshing = meshing


@pytest.mark.parametrize(
    ("target", "meshing", "expected"),
    [
        (magpy.magnet.Cuboid(dimension=(1, 1, 1)), np.int64(8), 8),
        (magpy.magnet.Cuboid(dimension=(1, 1, 1)), [2, 2, 2], (2, 2, 2)),
        (magpy.magnet.Cuboid(dimension=(1, 1, 1)), np.array([2, 3, 4]), (2, 3, 4)),
        (magpy.magnet.Cuboid(dimension=(1, 1, 1)), None, None),
        (magpy.current.Circle(diameter=1, current=1), np.int64(4), 4),
    ],
)
def test_meshing_is_stored_as_ints(target, meshing, expected):
    """Valid meshing inputs are stored as int or a tuple of 3 ints."""
    target.meshing = meshing
    assert target.meshing == expected
    assert type(target.meshing) is type(expected)


def test_polyline_meshing_below_segments_warns():
    """A Polyline with fewer mesh points than segments warns and uses one per
    segment."""
    cube = magpy.magnet.Cuboid(dimension=(1, 1, 1), polarization=POL)
    poly = magpy.current.Polyline(
        vertices=[(0, 0, 0), (1, 1, 1), (2, 2, 2)], current=1, meshing=1
    )
    with pytest.warns(UserWarning, match="Setting one point per segment"):
        getFT(cube, poly)

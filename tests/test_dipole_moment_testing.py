import numpy as np
import pytest

import magpylib as magpy


def test_circle_dipole_moment():
    """Test dipole moment calculation for Circle current source."""
    i0 = 1.23
    r = 1.23
    circle = magpy.current.Circle(diameter=2 * r, current=i0)
    moment = np.pi * r**2 * i0 * np.array((0, 0, 1))
    assert np.all(circle.dipole_moment == moment)


def test_polyline_closed_dipole_moment():
    """Test dipole moment calculation for closed Polyline current source."""
    i0 = 1.23
    polyline_vertices = np.array(
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0)]
    )
    polyline = magpy.current.Polyline(current=i0, vertices=polyline_vertices)
    # Area of unit square loop: 1 * 1 = 1, normal in z-direction
    area = 1.0
    moment = np.array(
        [0, 0, area * i0]
    )  # current loop in xy-plane gives z-directed dipole
    assert np.all(polyline.dipole_moment == moment)


def test_cuboid_dipole_moment():
    """Test dipole moment calculation for Cuboid magnet."""
    mag = np.array([1.1e2, 1.23e3, -1e4])
    a, b, c = 1, 0.2, 3.2
    cub = magpy.magnet.Cuboid(dimension=(a, b, c), magnetization=mag)
    moment = a * b * c * mag
    assert np.all(cub.dipole_moment == moment)


def test_cylinder_dipole_moment():
    """Test dipole moment calculation for Cylinder magnet."""
    mag = np.array([1.1e2, 1.23e3, -1e4])
    d, h = 1.12, 2.32
    cyl = magpy.magnet.Cylinder(dimension=(d, h), magnetization=mag)
    moment = np.pi / 4 * d**2 * h * mag
    assert np.all(cyl.dipole_moment == moment)


def test_cylinder_segment_dipole_moment():
    """Test dipole moment calculation for CylinderSegment magnet."""
    mag = np.array([1.1e2, 1.23e3, -1e4])
    r1, r2, height, phi1, phi2 = (
        0.5,
        1.0,
        2.0,
        0,
        90,
    )  # inner radius, outer radius, height, start angle, end angle
    cyl_seg = magpy.magnet.CylinderSegment(
        dimension=(r1, r2, height, phi1, phi2), magnetization=mag
    )
    volume = (r2**2 - r1**2) * np.pi * height * (phi2 - phi1) / 360
    moment = volume * mag
    assert np.all(cyl_seg.dipole_moment == moment)


def test_sphere_dipole_moment():
    """Test dipole moment calculation for Sphere magnet."""
    mag = np.array([1.1e2, 1.23e3, -1e4])
    r = 1.23
    sph = magpy.magnet.Sphere(diameter=2 * r, magnetization=mag)
    moment = 4 * r**3 * np.pi / 3 * mag
    assert np.all(sph.dipole_moment == moment)


def test_tetrahedron_dipole_moment():
    """Test dipole moment calculation for Tetrahedron magnet."""
    mag = np.array([1.1e2, 1.23e3, -1e4])
    vertices = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    tet = magpy.magnet.Tetrahedron(vertices=vertices, magnetization=mag)
    # Volume of tetrahedron: (1/6) * |det([v1-v0, v2-v0, v3-v0])|
    v0, v1, v2, v3 = vertices
    volume = abs(np.linalg.det([v1 - v0, v2 - v0, v3 - v0])) / 6
    moment = volume * mag
    assert np.all(tet.dipole_moment == moment)


def test_triangular_mesh_dipole_moment():
    """Test dipole moment calculation for TriangularMesh magnet."""
    mag = np.array([1.1e2, 1.23e3, -1e4])
    mesh_vertices = np.array(
        [
            [0, 0, 0],  # vertex 0
            [1, 0, 0],  # vertex 1
            [0, 1, 0],  # vertex 2
            [0, 0, 1],  # vertex 3
        ]
    )
    mesh_faces = np.array(
        [
            [0, 1, 2],  # triangle 1
            [0, 1, 3],  # triangle 2
            [0, 2, 3],  # triangle 3
            [1, 2, 3],  # triangle 4
        ]
    )
    trimesh = magpy.magnet.TriangularMesh(
        vertices=mesh_vertices, faces=mesh_faces, magnetization=mag
    )
    # Volume calculation for tetrahedron from vertices
    volume = (
        abs(
            np.linalg.det(
                [
                    mesh_vertices[1] - mesh_vertices[0],
                    mesh_vertices[2] - mesh_vertices[0],
                    mesh_vertices[3] - mesh_vertices[0],
                ]
            )
        )
        / 6
    )
    moment = volume * mag
    assert np.all(trimesh.dipole_moment == moment)


def test_dipole_dipole_moment():
    """Test dipole moment calculation for Dipole source."""
    moment_input = np.array([0.01, 0.02, 0.03])  # Am²
    dipole = magpy.misc.Dipole(moment=moment_input)
    # For a point dipole, the dipole moment is simply the moment itself
    moment = moment_input
    assert np.all(dipole.dipole_moment == moment)


def test_collection_dipole_moment():
    """Test dipole moment calculation for Collection of sources."""
    # Create individual objects
    mag = np.array([1.1e2, 1.23e3, -1e4])
    a, b, c = 1, 0.2, 3.2
    cub = magpy.magnet.Cuboid(dimension=(a, b, c), magnetization=mag)

    d, h = 1.12, 2.32
    cyl = magpy.magnet.Cylinder(dimension=(d, h), magnetization=mag)

    r = 1.23
    sph = magpy.magnet.Sphere(diameter=2 * r, magnetization=mag)

    vertices = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    tet = magpy.magnet.Tetrahedron(vertices=vertices, magnetization=mag)

    coll = magpy.Collection(cub, cyl, sph, tet)
    assert np.all(
        coll.dipole_moment
        == cub.dipole_moment + cyl.dipole_moment + sph.dipole_moment + tet.dipole_moment
    )


def test_polyline_field_accuracy():
    """Test field accuracy comparison between Polyline and equivalent Dipole."""
    i0 = 0.12345
    vertices = np.array(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 3, 4), (-3, -2, 1), (0, 0, 0)]
    )
    curr = magpy.current.Polyline(current=i0, vertices=vertices)

    dipp = magpy.misc.Dipole(moment=curr.dipole_moment)

    obs = np.array([0.5, -1.1, 1.2]) * 1e5
    B1 = curr.getB(obs)
    B2 = dipp.getB(obs)
    err = np.linalg.norm(B1 - B2) / np.linalg.norm(B2)
    assert err < 3e-5


def test_open_polyline_error():
    """Test that open polyline throws error when calling dipole_moment."""
    i0_open = 0.5
    vertices_open = np.array(
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    )  # Open polyline
    curr_open = magpy.current.Polyline(current=i0_open, vertices=vertices_open)

    with pytest.raises(ValueError):  # noqa: PT011
        _ = curr_open.dipole_moment


def test_dipole_moment_triangle_strip():
    """Test dipole moment calculation for TriangleStrip source."""
    i0 = 1.12345
    vertices = np.array(
        [
            (0, 0, 0),
            (0, 0, 1),
            (1.1, 0, 0),
            (1.2, 0, 3),
            (1.3, 1.1, 0.1),
            (1.4, 1.3, 2),
            (0, 1, -1),
            (0, 1, 3),
            (0, 0, 0),
            (0, 0, 1),
        ]
    )
    curr = magpy.current.TriangleStrip(current=i0, vertices=vertices)
    mom = curr.dipole_moment

    dipp = magpy.misc.Dipole(moment=mom, position=curr.centroid)

    obs = np.array([0.5, -1.1, 1.2]) * 1e3
    B1 = curr.getB(obs)
    B2 = dipp.getB(obs)
    err = np.linalg.norm(B1 - B2) / np.linalg.norm(B2)
    assert err < 5e-4


def test_open_triangle_strip_error():
    """Test that open triangle strip throws error when calling dipole_moment."""
    i0_open = 0.5
    # Open triangle strip vertices (not forming a closed surface)
    vertices_open = np.array(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
    )  # Open triangle strip
    curr_open = magpy.current.TriangleStrip(current=i0_open, vertices=vertices_open)
    with pytest.raises(ValueError):  # noqa: PT011
        _ = curr_open.dipole_moment

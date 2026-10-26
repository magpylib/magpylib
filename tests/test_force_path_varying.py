"""Tests for path-varying target properties in getFT.

Each target class has two test functions:
- test_path_varying_<class>_*  : both path properties vary at the same length.
- test_mismatched_paths_<class>_*: one property is scalar (p=1), the other is a
  path array (p>1), exercising _sync_path_lengths in each _generate_mesh.
"""

import numpy as np
import pytest

import magpylib as magpy

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_POSITIONS = np.array([[0, 0, i * 0.01] for i in range(3)])
_MAGS = np.array([[0, 0, 1e6], [0, 0, 1.2e6], [0, 0, 1.5e6]])
_CURRENTS = np.array([100.0, 200.0, 300.0])


def _assert_matches_manual_loop(
    dipole, target_vectorized, target_factory, n_steps, rtol, atol=0, meshing=None
):
    """Compare vectorised getFT against a per-step manual loop."""
    F_vec, T_vec = magpy.getFT(dipole, target_vectorized, meshing=meshing)
    F_man = np.array(
        [
            magpy.getFT(dipole, target_factory(i), meshing=meshing)[0]
            for i in range(n_steps)
        ]
    )
    T_man = np.array(
        [
            magpy.getFT(dipole, target_factory(i), meshing=meshing)[1]
            for i in range(n_steps)
        ]
    )
    np.testing.assert_allclose(F_vec, F_man, rtol=rtol, atol=atol)
    np.testing.assert_allclose(T_vec, T_man, rtol=rtol, atol=atol)


################################################################################
# CURRENT SOURCES
################################################################################


def test_path_varying_circle_current_diameter():
    """Circle: both diameter and current vary at p=4."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -2))

    diameters = np.array([1.0, 2.0, 3.0, 4.0])
    currents = np.array([100.0, 200.0, 300.0, 400.0])
    positions = np.array([[0, 0, i] for i in range(4)])

    circle_varying = magpy.current.Circle(
        diameter=diameters, current=currents, position=positions
    )

    F_vectorized, T_vectorized = magpy.getFT(dipole, circle_varying, meshing=20)

    F_manual, T_manual = [], []
    for i in range(len(diameters)):
        F_i, T_i = magpy.getFT(
            dipole,
            magpy.current.Circle(
                diameter=diameters[i],
                current=currents[i],
                position=positions[i],
            ),
            meshing=20,
        )
        F_manual.append(F_i)
        T_manual.append(T_i)

    np.testing.assert_allclose(F_vectorized, np.array(F_manual), rtol=1e-10)
    np.testing.assert_allclose(T_vectorized, np.array(T_manual), rtol=1e-10)
    assert F_vectorized.shape == (4, 3)
    assert not np.allclose(F_vectorized[0], F_vectorized[1])


def test_mismatched_paths_circle_scalar_diameter():
    """Circle: scalar diameter (p=1), varying current (p=3)."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -5))
    circle = magpy.current.Circle(diameter=1.0, current=_CURRENTS, position=_POSITIONS)
    _assert_matches_manual_loop(
        dipole,
        circle,
        lambda i: magpy.current.Circle(
            diameter=1.0, current=_CURRENTS[i], position=_POSITIONS[i]
        ),
        n_steps=3,
        rtol=1e-10,
        meshing=20,
    )


def test_mismatched_paths_circle_scalar_current():
    """Circle: varying diameter (p=3), scalar current (p=1)."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -5))
    diameters = np.array([1.0, 2.0, 3.0])
    circle = magpy.current.Circle(
        diameter=diameters, current=100.0, position=_POSITIONS
    )
    _assert_matches_manual_loop(
        dipole,
        circle,
        lambda i: magpy.current.Circle(
            diameter=diameters[i], current=100.0, position=_POSITIONS[i]
        ),
        n_steps=3,
        rtol=1e-10,
        meshing=20,
    )


def test_path_varying_polyline_vertices():
    """Polyline: vertices vary at p=3 (expanding square loop), current constant."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -2))

    vertices_path = np.array(
        [
            [
                [-0.5, -0.5, 0],
                [0.5, -0.5, 0],
                [0.5, 0.5, 0],
                [-0.5, 0.5, 0],
                [-0.5, -0.5, 0],
            ],
            [
                [-1.0, -1.0, 0],
                [1.0, -1.0, 0],
                [1.0, 1.0, 0],
                [-1.0, 1.0, 0],
                [-1.0, -1.0, 0],
            ],
            [
                [-1.5, -1.5, 0],
                [1.5, -1.5, 0],
                [1.5, 1.5, 0],
                [-1.5, 1.5, 0],
                [-1.5, -1.5, 0],
            ],
        ]
    )
    currents = np.array([100.0, 100.0, 100.0])
    positions = np.array([[0, 0, i] for i in range(3)])

    polyline_varying = magpy.current.Polyline(
        vertices=vertices_path, current=currents, position=positions
    )

    F_vectorized, T_vectorized = magpy.getFT(dipole, polyline_varying, meshing=20)

    F_manual, T_manual = [], []
    for i in range(len(vertices_path)):
        F_i, T_i = magpy.getFT(
            dipole,
            magpy.current.Polyline(
                vertices=vertices_path[i],
                current=currents[i],
                position=positions[i],
            ),
            meshing=20,
        )
        F_manual.append(F_i)
        T_manual.append(T_i)

    np.testing.assert_allclose(F_vectorized, np.array(F_manual), rtol=1e-10)
    np.testing.assert_allclose(T_vectorized, np.array(T_manual), rtol=1e-10)

    force_magnitudes = np.linalg.norm(F_vectorized, axis=1)
    assert force_magnitudes[1] < force_magnitudes[0]
    assert force_magnitudes[2] < force_magnitudes[1]


def test_mismatched_paths_polyline_scalar_vertices():
    """Polyline: scalar vertices (p=1), varying current (p=3)."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -5))
    verts = np.array(
        [[-0.5, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0], [-0.5, 0, 0]]
    )
    polyline = magpy.current.Polyline(
        vertices=verts, current=_CURRENTS, position=_POSITIONS
    )
    _assert_matches_manual_loop(
        dipole,
        polyline,
        lambda i: magpy.current.Polyline(
            vertices=verts, current=_CURRENTS[i], position=_POSITIONS[i]
        ),
        n_steps=3,
        rtol=1e-10,
        meshing=20,
    )


def test_path_varying_triangle_strip_vertices_current():
    """TriangleStrip: both vertices and current vary at p=3."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -2))

    vertices_path = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [0, 0.5, 0], [1, 0.5, 0]],
            [[0, 0, 0.1], [1, 0, 0.1], [0, 0.5, 0.15], [3, 0.5, 0.15]],
            [[0, 0, 0.2], [1, 0, 0.2], [0, 0.5, 0.3], [0.1, 0.5, 0.3]],
        ]
    )
    currents_path = np.array([50, 100, 150])
    positions = np.array([[0, 0, i * 0.5] for i in range(3)])

    strip_varying = magpy.current.TriangleStrip(
        vertices=vertices_path, current=currents_path, position=positions
    )

    F_vectorized, _ = magpy.getFT(dipole, strip_varying, meshing=10)

    _assert_matches_manual_loop(
        dipole,
        strip_varying,
        lambda i: magpy.current.TriangleStrip(
            vertices=vertices_path[i],
            current=currents_path[i],
            position=positions[i],
        ),
        n_steps=3,
        rtol=1e-8,
        atol=1e-10,
        meshing=10,
    )
    assert F_vectorized.shape == (3, 3)
    assert not np.allclose(F_vectorized[0], F_vectorized[1])


def test_mismatched_paths_triangle_strip_scalar_vertices():
    """TriangleStrip: scalar vertices (p=1), varying current (p=3)."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -5))
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 0.5, 0], [1, 0.5, 0]])
    strip = magpy.current.TriangleStrip(
        vertices=verts, current=_CURRENTS, position=_POSITIONS
    )
    _assert_matches_manual_loop(
        dipole,
        strip,
        lambda i: magpy.current.TriangleStrip(
            vertices=verts, current=_CURRENTS[i], position=_POSITIONS[i]
        ),
        n_steps=3,
        rtol=1e-8,
        meshing=10,
        atol=1e-30,
    )


def test_path_varying_triangle_sheet_vertices_current_densities():
    """TriangleSheet: fixed vertices (p=1), varying current_densities (p=3).

    This was the original failing test that exposed the _sync_path_lengths bug.
    """
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -2))

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])

    angles = np.array([0, np.pi / 4, np.pi / 2])
    current_densities_path = np.zeros((3, 2, 3))
    for i, angle in enumerate(angles):
        current_densities_path[i, :, 0] = 100 * np.cos(angle)
        current_densities_path[i, :, 1] = 100 * np.sin(angle)

    positions = np.array([[0, 0, i * 0.5] for i in range(3)])

    sheet_varying = magpy.current.TriangleSheet(
        vertices=vertices,
        faces=faces,
        current_densities=current_densities_path,
        position=positions,
    )

    F_vectorized, T_vectorized = magpy.getFT(dipole, sheet_varying, meshing=10)

    F_manual, T_manual = [], []
    for i in range(len(current_densities_path)):
        F_i, T_i = magpy.getFT(
            dipole,
            magpy.current.TriangleSheet(
                vertices=vertices,
                faces=faces,
                current_densities=current_densities_path[i],
                position=positions[i],
            ),
            meshing=10,
        )
        F_manual.append(F_i)
        T_manual.append(T_i)

    np.testing.assert_allclose(F_vectorized, np.array(F_manual), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(T_vectorized, np.array(T_manual), rtol=1e-8, atol=1e-10)
    assert F_vectorized.shape == (3, 3)
    assert not np.allclose(F_vectorized[0], F_vectorized[1])
    assert not np.allclose(F_vectorized[1], F_vectorized[2])


def test_mismatched_paths_triangle_sheet_scalar_current_densities():
    """TriangleSheet: varying vertices (p=3), scalar current_densities (p=1)."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -5))
    verts1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    verts2 = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [6, 2, 0]])
    verts_path = np.array([verts1, verts2, verts1])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    cd = np.array([[100.0, 0, 0], [100.0, 0, 0]])
    sheet = magpy.current.TriangleSheet(
        vertices=verts_path,
        faces=faces,
        current_densities=cd,
        position=_POSITIONS,
    )
    _assert_matches_manual_loop(
        dipole,
        sheet,
        lambda i: magpy.current.TriangleSheet(
            vertices=verts_path[i],
            faces=faces,
            current_densities=cd,
            position=_POSITIONS[i],
        ),
        n_steps=3,
        rtol=1e-8,
        meshing=10,
        atol=1e-30,
    )


################################################################################
# MAGNET SOURCES
################################################################################


def test_path_varying_cuboid_dimension_magnetization():
    """Cuboid: both dimension and magnetization vary at p=3."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -5))

    dimensions_varying = np.array(
        [
            [0.001, 0.001, 0.01],
            [0.01, 0.001, 0.001],
            [0.002, 0.002, 0.002],
        ]
    )
    magnetizations_varying = np.array([[0, 0, 1e6], [0, 0, 1.2e6], [0, 0, 1.5e6]])
    positions = np.array([[0, 0, i * 0.01] for i in range(3)])

    cuboid_varying = magpy.magnet.Cuboid(
        dimension=dimensions_varying,
        magnetization=magnetizations_varying,
        position=positions,
    )

    F_vec, T_vec = magpy.getFT(dipole, cuboid_varying, meshing=50)

    F_manual, T_manual = [], []
    for i in range(3):
        F_i, T_i = magpy.getFT(
            dipole,
            magpy.magnet.Cuboid(
                dimension=dimensions_varying[i],
                magnetization=magnetizations_varying[i],
                position=positions[i],
            ),
            meshing=50,
        )
        F_manual.append(F_i)
        T_manual.append(T_i)

    np.testing.assert_allclose(F_vec, np.array(F_manual), rtol=1e-7, atol=1e-23)
    np.testing.assert_allclose(T_vec, np.array(T_manual), rtol=1e-7, atol=1e-27)
    assert F_vec.shape == (3, 3)
    assert not np.allclose(F_vec[0], F_vec[1], rtol=0, atol=0)


def test_mismatched_paths_cuboid_scalar_dimension():
    """Cuboid: scalar dimension (p=1), varying magnetization (p=3)."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -5))
    cuboid = magpy.magnet.Cuboid(
        dimension=(0.002, 0.002, 0.002),
        magnetization=_MAGS,
        position=_POSITIONS,
    )
    _assert_matches_manual_loop(
        dipole,
        cuboid,
        lambda i: magpy.magnet.Cuboid(
            dimension=(0.002, 0.002, 0.002),
            magnetization=_MAGS[i],
            position=_POSITIONS[i],
        ),
        n_steps=3,
        rtol=1e-7,
        meshing=50,
        atol=1e-30,
    )


def test_mismatched_paths_cuboid_scalar_magnetization():
    """Cuboid: varying dimension (p=3), scalar magnetization (p=1)."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -5))
    dims = np.array(
        [[0.001, 0.001, 0.002], [0.002, 0.001, 0.001], [0.001, 0.002, 0.001]]
    )
    cuboid = magpy.magnet.Cuboid(
        dimension=dims,
        magnetization=(0, 0, 1e6),
        position=_POSITIONS,
    )
    _assert_matches_manual_loop(
        dipole,
        cuboid,
        lambda i: magpy.magnet.Cuboid(
            dimension=dims[i],
            magnetization=(0, 0, 1e6),
            position=_POSITIONS[i],
        ),
        n_steps=3,
        rtol=1e-7,
        meshing=50,
        atol=1e-30,
    )


def test_path_varying_cylinder_segment():
    """CylinderSegment: both dimension and magnetization vary at p=3."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -5))

    dimensions_varying = np.array(
        [
            [0.001, 0.003, 0.002, 0, 90],
            [0.002, 0.004, 0.001, 45, 135],
            [0.001, 0.003, 0.002, 0, 90],
        ]
    )
    magnetizations_varying = np.array([[0, 0, 1e6], [0, 0, 1.2e6], [0, 0, 1.5e6]])
    positions = np.array([[0, 0, i * 0.01] for i in range(3)])

    cyl_seg_varying = magpy.magnet.CylinderSegment(
        dimension=dimensions_varying,
        magnetization=magnetizations_varying,
        position=positions,
    )

    F_vec, T_vec = magpy.getFT(dipole, cyl_seg_varying, meshing=50)

    F_manual, T_manual = [], []
    for i in range(3):
        F_i, T_i = magpy.getFT(
            dipole,
            magpy.magnet.CylinderSegment(
                dimension=dimensions_varying[i],
                magnetization=magnetizations_varying[i],
                position=positions[i],
            ),
            meshing=50,
        )
        F_manual.append(F_i)
        T_manual.append(T_i)

    np.testing.assert_allclose(F_vec, np.array(F_manual), rtol=1e-7, atol=1e-23)
    np.testing.assert_allclose(T_vec, np.array(T_manual), rtol=1e-7, atol=1e-27)


def test_mismatched_paths_cylinder_segment_scalar_dimension():
    """CylinderSegment: scalar dimension (p=1), varying magnetization (p=3)."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -5))
    dim = (0.001, 0.003, 0.002, 0, 90)
    cylseg = magpy.magnet.CylinderSegment(
        dimension=dim,
        magnetization=_MAGS,
        position=_POSITIONS,
    )
    _assert_matches_manual_loop(
        dipole,
        cylseg,
        lambda i: magpy.magnet.CylinderSegment(
            dimension=dim,
            magnetization=_MAGS[i],
            position=_POSITIONS[i],
        ),
        n_steps=3,
        rtol=1e-7,
        meshing=30,
        atol=1e-30,
    )


def test_path_varying_sphere_diameter_magnetization():
    """Sphere: both diameter and magnetization vary at p=3; also scalar-diameter case."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -5))

    diameters_varying = np.array([0.001, 0.0015, 0.002])
    magnetizations_varying = np.array([[0, 0, 1e6], [0, 0, 1.5e6], [0, 0, 2e6]])
    positions = np.array([[0, 0, i * 0.5] for i in range(3)])

    # CASE 1: both vary
    sphere_both = magpy.magnet.Sphere(
        diameter=diameters_varying,
        magnetization=magnetizations_varying,
        position=positions,
    )
    F_vec_both, T_vec_both = magpy.getFT(dipole, sphere_both)

    F_manual_both, T_manual_both = [], []
    for i in range(3):
        F_i, T_i = magpy.getFT(
            dipole,
            magpy.magnet.Sphere(
                diameter=diameters_varying[i],
                magnetization=magnetizations_varying[i],
                position=positions[i],
            ),
        )
        F_manual_both.append(F_i)
        T_manual_both.append(T_i)

    np.testing.assert_allclose(F_vec_both, np.array(F_manual_both), rtol=1e-10)
    np.testing.assert_allclose(T_vec_both, np.array(T_manual_both), rtol=1e-10)

    # CASE 2: scalar diameter (p=1), varying magnetization (p=3)
    sphere_mag = magpy.magnet.Sphere(
        diameter=0.001,
        magnetization=magnetizations_varying,
        position=positions,
    )
    F_vec_mag, T_vec_mag = magpy.getFT(dipole, sphere_mag)

    F_manual_mag, T_manual_mag = [], []
    for i in range(3):
        F_i, T_i = magpy.getFT(
            dipole,
            magpy.magnet.Sphere(
                diameter=0.001,
                magnetization=magnetizations_varying[i],
                position=positions[i],
            ),
        )
        F_manual_mag.append(F_i)
        T_manual_mag.append(T_i)

    np.testing.assert_allclose(F_vec_mag, np.array(F_manual_mag), rtol=1e-10)
    np.testing.assert_allclose(T_vec_mag, np.array(T_manual_mag), rtol=1e-10)

    assert F_vec_both.shape == (3, 3)
    assert F_vec_mag.shape == (3, 3)
    assert not np.allclose(F_vec_both[0], F_vec_both[1], rtol=0, atol=0)
    assert not np.allclose(F_vec_mag[0], F_vec_mag[1], rtol=0, atol=0)


def test_mismatched_paths_sphere_scalar_magnetization():
    """Sphere: varying diameter (p=3), scalar magnetization (p=1).

    The reverse direction (scalar diameter + varying magnetization) accidentally
    passed before _sync_path_lengths because sphere mesh pts are always (0,0,0).
    """
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -5))
    diams = np.array([0.001, 0.0015, 0.002])
    sphere = magpy.magnet.Sphere(
        diameter=diams,
        magnetization=(0, 0, 1e6),
        position=_POSITIONS,
    )
    _assert_matches_manual_loop(
        dipole,
        sphere,
        lambda i: magpy.magnet.Sphere(
            diameter=diams[i],
            magnetization=(0, 0, 1e6),
            position=_POSITIONS[i],
        ),
        n_steps=3,
        rtol=1e-10,
    )


def test_path_varying_tetrahedron():
    """Tetrahedron: both vertices and magnetization vary at p=3."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -5))

    vertices_varying = np.array(
        [
            [[0, 0, 0], [0.002, 0, 0], [0, 0.002, 0], [0, 0, 0.002]],
            [[0, 0, 0], [0.003, 0, 0], [0, 0.001, 0], [0, 0, 0.003]],
            [[0, 0, 0], [0.002, 0, 0], [0, 0.002, 0], [0, 0, 0.002]],
        ]
    )
    magnetizations_varying = np.array([[0, 0, 1e6], [0, 0, 1.2e6], [0, 0, 1.5e6]])
    positions = np.array([[0, 0, i * 0.01] for i in range(3)])

    tet_varying = magpy.magnet.Tetrahedron(
        vertices=vertices_varying,
        magnetization=magnetizations_varying,
        position=positions,
    )

    F_vec, T_vec = magpy.getFT(dipole, tet_varying, meshing=50)

    F_manual, T_manual = [], []
    for i in range(3):
        F_i, T_i = magpy.getFT(
            dipole,
            magpy.magnet.Tetrahedron(
                vertices=vertices_varying[i],
                magnetization=magnetizations_varying[i],
                position=positions[i],
            ),
            meshing=50,
        )
        F_manual.append(F_i)
        T_manual.append(T_i)

    np.testing.assert_allclose(F_vec, np.array(F_manual), rtol=1e-7, atol=1e-23)
    np.testing.assert_allclose(T_vec, np.array(T_manual), rtol=1e-7, atol=1e-27)


def test_mismatched_paths_tetrahedron_scalar_vertices():
    """Tetrahedron: scalar vertices (p=1), varying magnetization (p=3)."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -5))
    verts = np.array([[0, 0, 0], [0.002, 0, 0], [0, 0.002, 0], [0, 0, 0.002]])
    tet = magpy.magnet.Tetrahedron(
        vertices=verts,
        magnetization=_MAGS,
        position=_POSITIONS,
    )
    _assert_matches_manual_loop(
        dipole,
        tet,
        lambda i: magpy.magnet.Tetrahedron(
            vertices=verts,
            magnetization=_MAGS[i],
            position=_POSITIONS[i],
        ),
        n_steps=3,
        rtol=1e-7,
        meshing=30,
        atol=1e-30,
    )


def test_path_varying_triangularmesh():
    """TriangularMesh: both vertices and magnetization vary at p=3."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -5))

    vertices1 = np.array(
        [
            [0, 0, 0],
            [0.002, 0, 0],
            [0.002, 0.002, 0],
            [0, 0.002, 0],
            [0, 0, 0.002],
            [0.002, 0, 0.002],
            [0.002, 0.002, 0.002],
            [0, 0.002, 0.002],
        ]
    )
    vertices2 = np.array(
        [
            [0, 0, 0],
            [0.003, 0, 0],
            [0.003, 0.003, 0],
            [0, 0.003, 0],
            [0, 0, 0.003],
            [0.003, 0, 0.003],
            [0.003, 0.003, 0.003],
            [0, 0.003, 0.003],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 3, 7],
            [0, 7, 4],
            [1, 2, 6],
            [1, 6, 5],
        ]
    )
    vertices_varying = np.array([vertices1, vertices2, vertices1])
    magnetizations_varying = np.array([[0, 0, 1e6], [0, 0, 1.2e6], [0, 0, 1.5e6]])
    positions = np.array([[0, 0, i * 0.01] for i in range(3)])

    trimesh_varying = magpy.magnet.TriangularMesh(
        vertices=vertices_varying,
        faces=faces,
        magnetization=magnetizations_varying,
        position=positions,
    )

    F_vec, T_vec = magpy.getFT(dipole, trimesh_varying, meshing=50)

    F_manual, T_manual = [], []
    for i in range(3):
        F_i, T_i = magpy.getFT(
            dipole,
            magpy.magnet.TriangularMesh(
                vertices=vertices_varying[i],
                faces=faces,
                magnetization=magnetizations_varying[i],
                position=positions[i],
            ),
            meshing=50,
        )
        F_manual.append(F_i)
        T_manual.append(T_i)

    np.testing.assert_allclose(F_vec, np.array(F_manual), rtol=1e-7, atol=1e-23)
    np.testing.assert_allclose(T_vec, np.array(T_manual), rtol=1e-7, atol=1e-27)


def test_mismatched_paths_triangularmesh_scalar_vertices():
    """TriangularMesh: scalar vertices (p=1), varying magnetization (p=3)."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -5))
    verts = np.array(
        [
            [0, 0, 0],
            [0.002, 0, 0],
            [0.002, 0.002, 0],
            [0, 0.002, 0],
            [0, 0, 0.002],
            [0.002, 0, 0.002],
            [0.002, 0.002, 0.002],
            [0, 0.002, 0.002],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 3, 7],
            [0, 7, 4],
            [1, 2, 6],
            [1, 6, 5],
        ]
    )
    trimesh = magpy.magnet.TriangularMesh(
        vertices=verts,
        faces=faces,
        magnetization=_MAGS,
        position=_POSITIONS,
    )
    _assert_matches_manual_loop(
        dipole,
        trimesh,
        lambda i: magpy.magnet.TriangularMesh(
            vertices=verts,
            faces=faces,
            magnetization=_MAGS[i],
            position=_POSITIONS[i],
        ),
        n_steps=3,
        rtol=1e-7,
        meshing=30,
        atol=1e-30,
    )


################################################################################
# COLLECTIONS
################################################################################


def test_path_varying_with_collections():
    """Collection: forces summed correctly over path-varying members."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -3))

    circle1 = magpy.current.Circle(diameter=[1.0, 2.0], current=[100.0, 200.0])
    circle2 = magpy.current.Circle(diameter=[1.5, 2.5], current=[150.0, 250.0])

    coll = magpy.Collection(circle1, circle2)
    coll.position = [(0, 0, 0), (0, 0, 1)]

    F_coll, T_coll = magpy.getFT(dipole, coll, meshing=20)

    F1, T1 = magpy.getFT(dipole, [circle1, circle2], meshing=20)
    F_manual = F1[:, 0, :] + F1[:, 1, :]
    T_manual = T1[:, 0, :] + T1[:, 1, :]

    np.testing.assert_allclose(F_coll, F_manual, rtol=1e-10)
    np.testing.assert_allclose(T_coll, T_manual, rtol=1e-10)


################################################################################
# MISC
################################################################################


def test_mismatched_path_lengths():
    """Source path (p=5) longer than target path (p=2): result padded to p=5."""
    dipole = magpy.misc.Dipole(
        moment=(1e3, 0, 0), position=[(0, 0, -2 - i * 0.5) for i in range(5)]
    )
    circle = magpy.current.Circle(
        diameter=[1.0, 2.0],
        current=[100.0, 200.0],
        position=[(0, 0, 0), (0, 0, 1)],
    )

    F, T = magpy.getFT(dipole, circle, meshing=20)

    assert F.shape == (5, 3)
    assert T.shape == (5, 3)

    # Steps 2-4 use edge-padded values (last step of the target path).
    dipole_step2 = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -3))
    circle_step2 = magpy.current.Circle(diameter=2.0, current=200.0, position=(0, 0, 1))
    F_expected, T_expected = magpy.getFT(dipole_step2, circle_step2, meshing=20)

    np.testing.assert_allclose(F[2], F_expected, rtol=1e-10)
    np.testing.assert_allclose(T[2], T_expected, rtol=1e-10)


def test_single_element_array_vs_scalar():
    """Single-element array inputs produce the same result as plain scalars."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -2))

    circle_scalar = magpy.current.Circle(
        diameter=2.0, current=100.0, position=(0, 0, 0)
    )
    circle_array = magpy.current.Circle(
        diameter=[2.0], current=[100.0], position=[(0, 0, 0)]
    )

    F_scalar, T_scalar = magpy.getFT(dipole, circle_scalar, meshing=20)
    F_array, T_array = magpy.getFT(dipole, circle_array, meshing=20)

    np.testing.assert_allclose(F_scalar, F_array, rtol=1e-15)
    np.testing.assert_allclose(T_scalar, T_array, rtol=1e-15)


def test_path_varying_with_centroid_pivot():
    """Default pivot='centroid' matches explicit per-step centroid for varying geometry."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -2))

    diameters = np.array([1.0, 2.0, 3.0])
    currents = np.array([100.0, 100.0, 100.0])
    positions = np.array([[0, 0, i] for i in range(3)])

    circle = magpy.current.Circle(
        diameter=diameters, current=currents, position=positions
    )
    _F_default, T_default = magpy.getFT(dipole, circle, meshing=20)

    T_manual = []
    for i in range(len(diameters)):
        _, T_i = magpy.getFT(
            dipole,
            magpy.current.Circle(
                diameter=diameters[i],
                current=currents[i],
                position=positions[i],
            ),
            pivot="centroid",
            meshing=20,
        )
        T_manual.append(T_i)

    np.testing.assert_allclose(T_default, np.array(T_manual), rtol=1e-10)


def test_collection_source_path_restores_children():
    """A Collection source with a path must not permanently displace its children.

    Regression: _compute_B_with_paths moves a Collection source per path step
    via the public position/orientation setters (which propagate to children),
    but _preserve_paths only restored the Collection's own attributes, leaving
    children permanently displaced (and their paths collapsed).
    """
    src = magpy.magnet.Cuboid(
        polarization=(0, 0, 1), dimension=(1, 1, 1), position=(0, 0, 2)
    )
    col = magpy.Collection(src)
    col.move([(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    pos_before = src.position.copy()
    ori_before = src.orientation.as_quat().copy()

    tgt = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    F_col, T_col = magpy.getFT(col, tgt, meshing=2)

    # children restored exactly
    np.testing.assert_array_equal(src.position, pos_before)
    np.testing.assert_allclose(src.orientation.as_quat(), ori_before)

    # a Collection of one source equals the bare source
    src2 = magpy.magnet.Cuboid(
        polarization=(0, 0, 1), dimension=(1, 1, 1), position=(0, 0, 2)
    )
    src2.move([(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    tgt2 = magpy.magnet.Cuboid(polarization=(0, 0, 1), dimension=(1, 1, 1))
    F_src, T_src = magpy.getFT(src2, tgt2, meshing=2)
    np.testing.assert_allclose(F_col, F_src)
    np.testing.assert_allclose(T_col, T_src)


def test_trimesh_static_geometry_varying_mag_matches_per_step():
    """Deduped path meshing must match independent per-step computation.

    generate_mesh_triangularmesh now meshes once per unique (vertices, volume)
    geometry and applies each step's magnetization, instead of re-meshing every
    step. The result must equal computing each step as a separate object.
    """
    v = (
        np.array(
            [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
            dtype=float,
        )
        * 0.5
    )
    f = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [0, 5, 2],
            [2, 5, 1],
            [1, 5, 3],
            [3, 5, 0],
        ]
    )
    src = magpy.magnet.Cuboid(
        polarization=(0, 0, 1), dimension=(1, 1, 1), position=(0, 0, 3)
    )
    mags = np.array([[0, 0, 1e6], [0, 0, 1.5e6], [1e6, 0, 0]])

    tm = magpy.magnet.TriangularMesh(vertices=v, faces=f, magnetization=mags)
    F, T = magpy.getFT(src, tm, meshing=30)

    for i, mag in enumerate(mags):
        tm_i = magpy.magnet.TriangularMesh(vertices=v, faces=f, magnetization=mag)
        F_i, T_i = magpy.getFT(src, tm_i, meshing=30)
        np.testing.assert_allclose(F[i], F_i, rtol=1e-10)
        np.testing.assert_allclose(T[i], T_i, rtol=1e-10)


def test_trianglesheet_static_geometry_varying_current_matches_per_step():
    """Same invariant for generate_mesh_triangle_current's geometry dedup."""
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0.2]])
    faces = np.array([[0, 1, 2], [1, 3, 2], [0, 2, 4], [1, 2, 4]])
    cds = np.array([[[1, 0, 0]] * 4, [[0, 1, 0]] * 4, [[0.5, 0.5, 0]] * 4], dtype=float)
    src = magpy.magnet.Cuboid(
        polarization=(0, 0, 1), dimension=(1, 1, 1), position=(0, 0, 3)
    )

    ts = magpy.current.TriangleSheet(vertices=verts, faces=faces, current_densities=cds)
    F, T = magpy.getFT(src, ts, meshing=12)

    for i in range(len(cds)):
        ts_i = magpy.current.TriangleSheet(
            vertices=verts, faces=faces, current_densities=cds[i]
        )
        F_i, T_i = magpy.getFT(src, ts_i, meshing=12)
        np.testing.assert_allclose(F[i], F_i, rtol=1e-10)
        np.testing.assert_allclose(T[i], T_i, rtol=1e-10)


_STRIP = np.array([[x, y, 0] for x in range(5) for y in (0, 0.5)]) * 0.01
_SQUARE = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [-1, -1, 0]]) * 0.01
_RINGS = np.array([(0.01, 0.02, 0.01, 0, 360), (0.01, 0.03, 0.01, 0, 360)])


@pytest.mark.parametrize(
    ("cls", "vertices", "meshing"),
    [(magpy.current.TriangleStrip, _STRIP, 40), (magpy.current.Polyline, _SQUARE, 10)],
    ids=["trianglestrip", "polyline"],
)
def test_current_meshed_alike_wherever_it_sits(cls, vertices, meshing):
    """Equal areas, edges and segment lengths tie up to rounding noise. A target moved
    through its vertices must be meshed like the same target moved by position."""
    shifts = np.arange(8)[:, np.newaxis] * [0.0123, 0.0071, 0.0031]
    src = magpy.misc.Dipole(
        moment=(0.1, 0.2, 1), position=np.array([0.021, 0.003, 0.006]) + shifts
    )
    by_vertices = cls(vertices=vertices + shifts[:, np.newaxis], current=1)
    by_position = cls(vertices=vertices, current=1, position=shifts)

    F_vertices, _ = magpy.getFT(src, by_vertices, meshing=meshing)
    F_position, _ = magpy.getFT(src, by_position, meshing=meshing)
    np.testing.assert_allclose(F_vertices, F_position, rtol=1e-10)


@pytest.mark.parametrize(
    ("make_target", "meshing"),
    [
        (
            lambda i: magpy.current.Polyline(
                vertices=np.array([_SQUARE, _SQUARE * [2, 1, 1]])[i], current=1
            ),
            10,
        ),
        (
            lambda i: magpy.magnet.CylinderSegment(
                dimension=_RINGS[i], polarization=(0, 0, 1)
            ),
            30,
        ),
    ],
    ids=["polyline", "cylindersegment"],
)
def test_mesh_padding_avoids_source_at_target_origin(make_target, meshing):
    """Path steps with fewer mesh points are padded. A dipole at the target's local
    origin must give the static result there, not NaN."""
    sizes = [len(make_target(i)._generate_mesh(meshing)["pts"]) for i in range(2)]
    assert sizes[0] != sizes[1]  # one step is padded
    _assert_matches_manual_loop(
        magpy.misc.Dipole(moment=(1, 2, 3)),
        make_target(slice(None)),
        make_target,
        n_steps=2,
        rtol=1e-10,
        atol=1e-18,
        meshing=meshing,
    )


def test_polyline_step_of_zero_length():
    """A path step whose vertices all coincide carries no current: zero force there,
    not 0/0 cast to int."""
    coil = magpy.current.Polyline(vertices=[_SQUARE * 0, _SQUARE], current=1)
    dipole = magpy.misc.Dipole(moment=(1, 2, 3), position=(0, 0, 0.01))
    F, _ = magpy.getFT(dipole, coil, meshing=10)
    np.testing.assert_array_equal(F[0], 0)
    assert np.all(np.isfinite(F[1]))

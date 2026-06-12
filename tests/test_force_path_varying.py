"""Tests for path-varying target properties in getFT.

Each target class has two test functions:
- test_path_varying_<class>_*  : both path properties vary at the same length.
- test_mismatched_paths_<class>_*: one property is scalar (p=1), the other is a
  path array (p>1), exercising _sync_path_lengths in each _generate_mesh.
"""

import numpy as np

import magpylib as magpy

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_POSITIONS = np.array([[0, 0, i * 0.01] for i in range(3)])
_MAGS = np.array([[0, 0, 1e6], [0, 0, 1.2e6], [0, 0, 1.5e6]])
_CURRENTS = np.array([100.0, 200.0, 300.0])


def _assert_matches_manual_loop(
    dipole, target_vectorized, target_factory, n_steps, rtol, atol=0
):
    """Compare vectorised getFT against a per-step manual loop."""
    F_vec, T_vec = magpy.getFT(dipole, target_vectorized)
    F_man = np.array(
        [magpy.getFT(dipole, target_factory(i))[0] for i in range(n_steps)]
    )
    T_man = np.array(
        [magpy.getFT(dipole, target_factory(i))[1] for i in range(n_steps)]
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
        diameter=diameters, current=currents, position=positions, meshing=20
    )

    F_vectorized, T_vectorized = magpy.getFT(dipole, circle_varying)

    F_manual, T_manual = [], []
    for i in range(len(diameters)):
        F_i, T_i = magpy.getFT(
            dipole,
            magpy.current.Circle(
                diameter=diameters[i],
                current=currents[i],
                position=positions[i],
                meshing=20,
            ),
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
    circle = magpy.current.Circle(
        diameter=1.0, current=_CURRENTS, position=_POSITIONS, meshing=20
    )
    _assert_matches_manual_loop(
        dipole,
        circle,
        lambda i: magpy.current.Circle(
            diameter=1.0, current=_CURRENTS[i], position=_POSITIONS[i], meshing=20
        ),
        n_steps=3,
        rtol=1e-10,
    )


def test_mismatched_paths_circle_scalar_current():
    """Circle: varying diameter (p=3), scalar current (p=1)."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -5))
    diameters = np.array([1.0, 2.0, 3.0])
    circle = magpy.current.Circle(
        diameter=diameters, current=100.0, position=_POSITIONS, meshing=20
    )
    _assert_matches_manual_loop(
        dipole,
        circle,
        lambda i: magpy.current.Circle(
            diameter=diameters[i], current=100.0, position=_POSITIONS[i], meshing=20
        ),
        n_steps=3,
        rtol=1e-10,
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
        vertices=vertices_path, current=currents, position=positions, meshing=20
    )

    F_vectorized, T_vectorized = magpy.getFT(dipole, polyline_varying)

    F_manual, T_manual = [], []
    for i in range(len(vertices_path)):
        F_i, T_i = magpy.getFT(
            dipole,
            magpy.current.Polyline(
                vertices=vertices_path[i],
                current=currents[i],
                position=positions[i],
                meshing=20,
            ),
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
        vertices=verts, current=_CURRENTS, position=_POSITIONS, meshing=20
    )
    _assert_matches_manual_loop(
        dipole,
        polyline,
        lambda i: magpy.current.Polyline(
            vertices=verts, current=_CURRENTS[i], position=_POSITIONS[i], meshing=20
        ),
        n_steps=3,
        rtol=1e-10,
    )


def test_path_varying_triangle_strip_vertices_current():
    """TriangleStrip: both vertices and current vary at p=3."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -2))

    vertices_path = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [0, 0.5, 0], [1, 0.5, 0]],
            [[0, 0, 0.1], [1, 0, 0.1], [0, 0.5, 0.15], [1, 0.5, 0.15]],
            [[0, 0, 0.2], [1, 0, 0.2], [0, 0.5, 0.3], [1, 0.5, 0.3]],
        ]
    )
    currents_path = np.array([50, 100, 150])
    positions = np.array([[0, 0, i * 0.5] for i in range(3)])

    strip_varying = magpy.current.TriangleStrip(
        vertices=vertices_path, current=currents_path, position=positions, meshing=10
    )

    F_vectorized, T_vectorized = magpy.getFT(dipole, strip_varying)

    F_manual, T_manual = [], []
    for i in range(len(currents_path)):
        F_i, T_i = magpy.getFT(
            dipole,
            magpy.current.TriangleStrip(
                vertices=vertices_path[i],
                current=currents_path[i],
                position=positions[i],
                meshing=10,
            ),
        )
        F_manual.append(F_i)
        T_manual.append(T_i)

    np.testing.assert_allclose(F_vectorized, np.array(F_manual), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(T_vectorized, np.array(T_manual), rtol=1e-8, atol=1e-10)
    assert F_vectorized.shape == (3, 3)
    assert not np.allclose(F_vectorized[0], F_vectorized[1])


def test_mismatched_paths_triangle_strip_scalar_vertices():
    """TriangleStrip: scalar vertices (p=1), varying current (p=3)."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -5))
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 0.5, 0], [1, 0.5, 0]])
    strip = magpy.current.TriangleStrip(
        vertices=verts, current=_CURRENTS, position=_POSITIONS, meshing=10
    )
    _assert_matches_manual_loop(
        dipole,
        strip,
        lambda i: magpy.current.TriangleStrip(
            vertices=verts, current=_CURRENTS[i], position=_POSITIONS[i], meshing=10
        ),
        n_steps=3,
        rtol=1e-8,
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
        meshing=10,
    )

    F_vectorized, T_vectorized = magpy.getFT(dipole, sheet_varying)

    F_manual, T_manual = [], []
    for i in range(len(current_densities_path)):
        F_i, T_i = magpy.getFT(
            dipole,
            magpy.current.TriangleSheet(
                vertices=vertices,
                faces=faces,
                current_densities=current_densities_path[i],
                position=positions[i],
                meshing=10,
            ),
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
    verts2 = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]])
    verts_path = np.array([verts1, verts2, verts1])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    cd = np.array([[100.0, 0, 0], [100.0, 0, 0]])
    sheet = magpy.current.TriangleSheet(
        vertices=verts_path,
        faces=faces,
        current_densities=cd,
        position=_POSITIONS,
        meshing=10,
    )
    _assert_matches_manual_loop(
        dipole,
        sheet,
        lambda i: magpy.current.TriangleSheet(
            vertices=verts_path[i],
            faces=faces,
            current_densities=cd,
            position=_POSITIONS[i],
            meshing=10,
        ),
        n_steps=3,
        rtol=1e-8,
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
        meshing=50,
    )

    F_vec, T_vec = magpy.getFT(dipole, cuboid_varying)

    F_manual, T_manual = [], []
    for i in range(3):
        F_i, T_i = magpy.getFT(
            dipole,
            magpy.magnet.Cuboid(
                dimension=dimensions_varying[i],
                magnetization=magnetizations_varying[i],
                position=positions[i],
                meshing=50,
            ),
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
        meshing=50,
    )
    _assert_matches_manual_loop(
        dipole,
        cuboid,
        lambda i: magpy.magnet.Cuboid(
            dimension=(0.002, 0.002, 0.002),
            magnetization=_MAGS[i],
            position=_POSITIONS[i],
            meshing=50,
        ),
        n_steps=3,
        rtol=1e-7,
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
        meshing=50,
    )
    _assert_matches_manual_loop(
        dipole,
        cuboid,
        lambda i: magpy.magnet.Cuboid(
            dimension=dims[i],
            magnetization=(0, 0, 1e6),
            position=_POSITIONS[i],
            meshing=50,
        ),
        n_steps=3,
        rtol=1e-7,
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
        meshing=50,
    )

    F_vec, T_vec = magpy.getFT(dipole, cyl_seg_varying)

    F_manual, T_manual = [], []
    for i in range(3):
        F_i, T_i = magpy.getFT(
            dipole,
            magpy.magnet.CylinderSegment(
                dimension=dimensions_varying[i],
                magnetization=magnetizations_varying[i],
                position=positions[i],
                meshing=50,
            ),
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
        meshing=30,
    )
    _assert_matches_manual_loop(
        dipole,
        cylseg,
        lambda i: magpy.magnet.CylinderSegment(
            dimension=dim,
            magnetization=_MAGS[i],
            position=_POSITIONS[i],
            meshing=30,
        ),
        n_steps=3,
        rtol=1e-7,
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
        meshing=50,
    )

    F_vec, T_vec = magpy.getFT(dipole, tet_varying)

    F_manual, T_manual = [], []
    for i in range(3):
        F_i, T_i = magpy.getFT(
            dipole,
            magpy.magnet.Tetrahedron(
                vertices=vertices_varying[i],
                magnetization=magnetizations_varying[i],
                position=positions[i],
                meshing=50,
            ),
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
        meshing=30,
    )
    _assert_matches_manual_loop(
        dipole,
        tet,
        lambda i: magpy.magnet.Tetrahedron(
            vertices=verts,
            magnetization=_MAGS[i],
            position=_POSITIONS[i],
            meshing=30,
        ),
        n_steps=3,
        rtol=1e-7,
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
        meshing=50,
    )

    F_vec, T_vec = magpy.getFT(dipole, trimesh_varying)

    F_manual, T_manual = [], []
    for i in range(3):
        F_i, T_i = magpy.getFT(
            dipole,
            magpy.magnet.TriangularMesh(
                vertices=vertices_varying[i],
                faces=faces,
                magnetization=magnetizations_varying[i],
                position=positions[i],
                meshing=50,
            ),
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
        meshing=30,
    )
    _assert_matches_manual_loop(
        dipole,
        trimesh,
        lambda i: magpy.magnet.TriangularMesh(
            vertices=verts,
            faces=faces,
            magnetization=_MAGS[i],
            position=_POSITIONS[i],
            meshing=30,
        ),
        n_steps=3,
        rtol=1e-7,
        atol=1e-30,
    )


################################################################################
# COLLECTIONS
################################################################################


def test_path_varying_with_collections():
    """Collection: forces summed correctly over path-varying members."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -3))

    circle1 = magpy.current.Circle(
        diameter=[1.0, 2.0], current=[100.0, 200.0], meshing=20
    )
    circle2 = magpy.current.Circle(
        diameter=[1.5, 2.5], current=[150.0, 250.0], meshing=20
    )

    coll = magpy.Collection(circle1, circle2)
    coll.position = [(0, 0, 0), (0, 0, 1)]

    F_coll, T_coll = magpy.getFT(dipole, coll)

    F1, T1 = magpy.getFT(dipole, [circle1, circle2])
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
        meshing=20,
    )

    F, T = magpy.getFT(dipole, circle)

    assert F.shape == (5, 3)
    assert T.shape == (5, 3)

    # Steps 2-4 use edge-padded values (last step of the target path).
    dipole_step2 = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -3))
    circle_step2 = magpy.current.Circle(
        diameter=2.0, current=200.0, position=(0, 0, 1), meshing=20
    )
    F_expected, T_expected = magpy.getFT(dipole_step2, circle_step2)

    np.testing.assert_allclose(F[2], F_expected, rtol=1e-10)
    np.testing.assert_allclose(T[2], T_expected, rtol=1e-10)


def test_single_element_array_vs_scalar():
    """Single-element array inputs produce the same result as plain scalars."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -2))

    circle_scalar = magpy.current.Circle(
        diameter=2.0, current=100.0, position=(0, 0, 0), meshing=20
    )
    circle_array = magpy.current.Circle(
        diameter=[2.0], current=[100.0], position=[(0, 0, 0)], meshing=20
    )

    F_scalar, T_scalar = magpy.getFT(dipole, circle_scalar)
    F_array, T_array = magpy.getFT(dipole, circle_array)

    np.testing.assert_allclose(F_scalar, F_array, rtol=1e-15)
    np.testing.assert_allclose(T_scalar, T_array, rtol=1e-15)


def test_path_varying_with_centroid_pivot():
    """Default pivot='centroid' matches explicit per-step centroid for varying geometry."""
    dipole = magpy.misc.Dipole(moment=(1e3, 0, 0), position=(0, 0, -2))

    diameters = np.array([1.0, 2.0, 3.0])
    currents = np.array([100.0, 100.0, 100.0])
    positions = np.array([[0, 0, i] for i in range(3)])

    circle = magpy.current.Circle(
        diameter=diameters, current=currents, position=positions, meshing=20
    )
    _F_default, T_default = magpy.getFT(dipole, circle)

    T_manual = []
    for i in range(len(diameters)):
        _, T_i = magpy.getFT(
            dipole,
            magpy.current.Circle(
                diameter=diameters[i],
                current=currents[i],
                position=positions[i],
                meshing=20,
            ),
            pivot="centroid",
        )
        T_manual.append(T_i)

    np.testing.assert_allclose(T_default, np.array(T_manual), rtol=1e-10)
